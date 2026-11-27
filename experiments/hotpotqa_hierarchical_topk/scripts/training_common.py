"""Shared experiment paths, frozen-data checks, and scoring for BERT training and evaluation."""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from analyze_labels import Cohort, counts, means, selected_metrics
from prepare_data import DEFAULT_SETTINGS, HERE, K_KEYS, K_VALUES, provenance, read_json, read_rows, require, validate_result
from split_data import SPLITS

CLASSES = (0, 1, 2, 3, 10)
HEAD_CLASSES = {"stage1": (0, 1), "stage2": CLASSES[:4], "flat5": CLASSES}
EXPERIMENTS = {"Hierarchical-CE": ("hierarchical_ce", ("stage1", "stage2"), False),
               "Hierarchical-WCE": ("hierarchical_wce", ("stage1", "stage2"), True),
               "Flat5-CE": ("flat5_ce", ("flat5",), False), "Flat5-WCE": ("flat5_wce", ("flat5",), True)}
RULE = ["score descending", "mean_f1 descending", "mean_context_tokens ascending", "earliest epoch tuple"]


@dataclass(frozen=True)
class Paths:
    metrics: Path
    validation_report: Path
    split_manifest: Path
    label_analysis: Path
    labels: Path
    training: Path
    evaluation: Path
    config_path: Path
    label_config_path: Path
    config: dict
    selection: dict


def entrypoint(function, description, options=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--label-settings", type=Path, default=HERE / "configs/label_settings.json")
    parser.add_argument("--training-settings", type=Path, default=HERE / "configs/training_settings.json")
    if options:
        options(parser)
    args = parser.parse_args()
    try:
        base = read_json(args.settings)["outputs"]
        label = read_json(args.label_settings)
        cfg = read_json(args.training_settings)
        require(cfg["labels"] == {"scheme": "B", "objective": "max_f1"}
                and label["selection"]["stage2_k"] + [label["selection"]["fallback_k"]] == list(CLASSES),
                "Expected fixed Scheme B, maximum F1 and the five deployment candidates")
        paths = Paths(HERE / base["per_question_metrics"], HERE / base["validation_report"],
                      HERE / label["outputs"]["splits"] / "split_manifest.json", HERE / label["outputs"]["analysis"],
                      HERE / "data/labels.jsonl", HERE / "runs/training", HERE / "runs/evaluation",
                      args.training_settings.resolve(), args.label_settings.resolve(), cfg, label["selection"])
        if options:
            function(paths, args)
        else:
            function(paths)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        sys.exit(f"{Path(sys.argv[0]).name}: {error}")


def load_data(paths, splits):
    manifest = read_json(paths.split_manifest)
    report = read_json(paths.validation_report)
    source = provenance(paths.metrics)
    require(report["status"] == "passed" and source == manifest["input"]
            and source["sha256"] == report["output"]["sha256"], "Frozen data hashes differ")
    membership = {}
    for split in SPLITS:
        qids = manifest["splits"][split]["qids"]
        require(len(qids) == manifest["configuration"]["sizes"][split], f"Wrong split size for {split}")
        for qid in qids:
            require(qid not in membership, f"Duplicate split QID {qid}")
            membership[qid] = split
    collected = {split: [] for split in splits}
    seen = set()
    for row in read_rows(paths.metrics, source["sha256"]):
        qid = row["qid"]
        require(qid in membership and qid not in seen, f"Unexpected or duplicate QID {qid}")
        seen.add(qid)
        if membership[qid] not in splits:
            continue
        require(set(row["results"]) == K_KEYS and isinstance(row["question"], str) and row["question"].strip(),
                f"Invalid question or k coverage for {qid}")
        for k in K_VALUES:
            validate_result(row["results"][str(k)], k, qid)
        collected[membership[qid]].append(row)
    require(seen == membership.keys(), "Split QID union differs from metrics")
    result = {}
    for split, rows in collected.items():
        require([row["qid"] for row in rows] == manifest["splits"][split]["qids"], f"Wrong QID order for {split}")
        arrays = [np.array([[row["results"][str(k)][field] for k in K_VALUES] for row in rows])
                  for field in ("f1", "em", "context_tokens")]
        result[split] = rows, Cohort(*arrays, np.array([r["level"] for r in rows]), np.array([r["type"] for r in rows]))
    return result, {"metrics": source, "split_manifest": provenance(paths.split_manifest),
                    "label_settings": provenance(paths.label_config_path)}


def read_labels(paths, datasets):
    expected = {split: rows for split, (rows, _) in datasets.items()}
    summary = read_json(paths.training / "label_summary.json")
    require(summary["sources"]["metrics"] == provenance(paths.metrics)
            and summary["sources"]["split_manifest"] == provenance(paths.split_manifest)
            and summary["sources"]["label_settings"] == provenance(paths.label_config_path), "Label sources have changed")
    digest = summary["sources"]["labels"]["sha256"]
    result = {split: [] for split in expected}
    for label in read_rows(paths.labels, digest):
        split = label["split"]
        if split not in expected:
            continue
        i = len(result[split])
        require(i < len(expected[split]), f"Extra label in {split}")
        row = expected[split][i]
        k = label["optimal_k"]
        require(label["qid"] == row["qid"] and label["question"] == row["question"] and k in CLASSES
                and label["flat5_label"] == CLASSES.index(k) and label["stage1_label"] == int(k == 10)
                and label["stage2_label"] == (None if k == 10 else k), f"Label alignment or encoding differs for {row['qid']}")
        result[split].append(label)
    require(all(len(result[s]) == len(rows) for s, rows in expected.items()), "Missing labels")
    return result


def label_counts(rows, head):
    labels = [row[f"{head}_label"] for row in rows if row[f"{head}_label"] is not None]
    return np.bincount(labels, minlength=len(HEAD_CLASSES[head]))


def head_path(paths, experiment, head):
    base = paths.training / EXPERIMENTS[experiment][0]
    return base if head == "flat5" else base / head


def checkpoint(path, epoch):
    return path / "checkpoints" / f"epoch_{epoch:02d}"


def classifier_metrics(truth, predicted, classes):
    truth, predicted = np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)
    require(truth.shape == predicted.shape and truth.ndim == 1 and truth.size > 0, "Classifier arrays differ or are empty")
    c = len(classes)
    require(np.all((0 <= truth) & (truth < c) & (0 <= predicted) & (predicted < c)), "Class index outside range")
    cm = np.bincount(c * truth + predicted, minlength=c*c).reshape(c, c)
    tp, support, predicted_count = cm.diagonal(), cm.sum(axis=1), cm.sum(axis=0)
    precision = tp / np.maximum(predicted_count, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * tp / np.maximum(support + predicted_count, 1)
    return {"count": truth.size, "classes": list(classes), "accuracy": float(tp.sum() / truth.size),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[support > 0].mean()),
            "confusion_matrix": cm.tolist(), "predicted_distribution": predicted_count.tolist(),
            "per_class": {str(name): {"support": int(support[i]), "precision": float(precision[i]),
                                      "recall": float(recall[i]), "f1": float(f1[i])} for i, name in enumerate(classes)}}


def route(stage1, stage2):
    first, second = np.asarray(stage1), np.asarray(stage2)
    require(first.shape == second.shape and np.all(np.isin(first, [0, 1]))
            and np.all(np.isin(second[first == 0], CLASSES[:4])), "Invalid predicted routing")
    return np.where(first == 1, CLASSES[-1], second)


def e2e(cohort, selected, paths):
    require(selected.shape == (len(cohort.f1),) and np.all(np.isin(selected, K_VALUES)), "Invalid selected k array")
    m = means(selected_metrics(cohort, selected, None), np.ones(len(selected), dtype=bool))
    del m["mean_utility"]
    m["score"] = m["mean_f1"] - paths.config["score_lambda"] * m["mean_context_tokens"] / paths.selection["token_scale"]
    m["selected_k_distribution"] = counts(selected, K_VALUES)
    return m


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, suffix=".tmp") as tmp:
        target = Path(tmp) / path.name
        with target.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        target.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, suffix=".tmp") as tmp:
        target = Path(tmp) / path.name
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        target.replace(path)


def setup_device(paths):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import torch

    require(paths.config["mixed_precision"] == "fp16" and paths.config["device"].startswith("cuda")
            and torch.cuda.is_available(), "Configured CUDA and FP16 are required")
    device = torch.device(paths.config["device"])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return device


def score_key(result, epochs):
    return (result["score"], result["mean_f1"], -result["mean_context_tokens"], *(-epoch for epoch in epochs))


def class_weights(class_sizes):
    require((class_sizes > 0).all(), "Every classifier must have all classes in Train")
    return class_sizes.sum() / (len(class_sizes) * class_sizes)


def check_training_config(paths):
    cfg = paths.config
    for field in ("train_batch_size", "max_sequence_length", "epochs"):
        require(type(cfg[field]) is int and cfg[field] > 0, f"Invalid {field}")
    for field in ("learning_rate", "weight_decay", "warmup_ratio", "gradient_clip", "score_lambda"):
        require(math.isfinite(cfg[field]) and cfg[field] > 0, f"Invalid {field}")
    require(cfg["warmup_ratio"] < 1 and cfg["max_sequence_length"] % 8 == 0, "Invalid warmup or padding configuration")
    seeds = cfg["random_prior_seeds"]
    require(len(set(seeds)) == len(seeds) == 100 and all(type(seed) is int and seed >= 0 for seed in seeds),
            "Expected 100 distinct, predefined Random-Prior seeds")
