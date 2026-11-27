"""Export aligned Validation and held-out Test predictions from the frozen BERT checkpoints."""

import numpy as np

from prepare_data import HERE, provenance, read_json, require
from training_common import (CLASSES, EXPERIMENTS, HEAD_CLASSES, checkpoint, entrypoint, head_path, load_data,
                             read_labels, route, setup_device, write_jsonl)
from analysis.hotpot_formal_runner.storage import atomic_json  # noqa: E402

PROTOCOL_FILE = "evaluation_protocol.json"
RESULT_DIRS = {"val": "validation", "test": "test"}


def protocol(paths, create=False):
    selected_path = paths.training / "frozen_selection.json"
    frozen = read_json(selected_path)
    require(frozen["selection_split"] == "val" and set(frozen["choices"]) == set(EXPERIMENTS)
            and frozen["plan"]["configuration"] == paths.config and frozen["token_scale"] == paths.selection["token_scale"],
            "Checkpoints and evaluation configuration must be frozen on Validation first")
    source_paths = {"metrics": paths.metrics, "split_manifest": paths.split_manifest, "labels": paths.labels,
                    "training_config": paths.config_path, "label_settings": paths.label_config_path}
    require(all(provenance(path) == frozen["plan"]["sources"][key] for key, path in source_paths.items()),
            "Frozen evaluation data or configuration changed")
    val_classes = {}
    for experiment, (_, heads, _) in EXPERIMENTS.items():
        require(set(frozen["choices"][experiment]["heads"]) == set(heads), f"Missing selected heads for {experiment}")
        val_classes[experiment] = {}
        for head in heads:
            selected_head = frozen["choices"][experiment]["heads"][head]
            directory = checkpoint(head_path(paths, experiment, head), selected_head["epoch"])
            val_classes[experiment][head] = provenance(directory / "validation_classes.json")
    code = ("analyze_labels.py", "build_labels.py", "training_common.py", "train.py", "predict.py", "evaluate.py")
    expected = {"frozen_selection": provenance(selected_path), "code": {name: provenance(HERE / "scripts" / name) for name in code},
                "validation_predictions": val_classes, "population": "all frozen QIDs in each evaluated split",
                "fixed_ks": list(CLASSES), "oracle_selection": "max_f1", "routing": "stage1 argmax, predicted low only to stage2",
                "score_lambda": frozen["score_lambda"], "token_scale": frozen["token_scale"],
                "random_prior_seeds": frozen["random_prior_seeds"], "random_prior_distribution": "Train flat5 frequencies",
                "random_prior_std": "sample standard deviation across independent seeds"}
    protocol_path = paths.evaluation / PROTOCOL_FILE
    if create and not protocol_path.exists():
        atomic_json(protocol_path, expected)
    require(read_json(protocol_path) == expected, "Frozen evaluation protocol differs")
    return frozen, provenance(protocol_path)


def model_predictions(paths, experiment, head, chosen, labels, mask, tokenizer, device):
    import torch
    from experiments.hotpotqa_hierarchical_topk.scripts.train import batches, dataset_for, new_model, run_epoch

    pred = np.full(len(labels), -1, dtype=int)
    if mask.any():
        directory = checkpoint(head_path(paths, experiment, head), chosen["epoch"])
        require(provenance(directory / "pytorch_model.bin") == chosen["weights"]
                and provenance(directory / "config.json") == chosen["model_config"], f"Frozen model changed in {directory}")
        questions = [{"question": row["question"], f"{head}_label": None} for row, use in zip(labels, mask) if use]
        data = dataset_for(questions, head, tokenizer, paths.config["max_sequence_length"])
        model = new_model(paths, head, device, directory)
        _, _, pred[mask] = run_epoch(model, batches(data, paths.config["train_batch_size"]), device, None)
        del model
        torch.cuda.empty_cache()
    require(np.all(np.isin(pred[mask], np.arange(len(HEAD_CLASSES[head])))), f"Invalid {head} prediction")
    return pred


def validation_predictions(paths, experiment, head, chosen, labels):
    directory = checkpoint(head_path(paths, experiment, head), chosen["epoch"])
    doc = read_json(directory / "validation_classes.json")
    expected = [row["qid"] for row in labels]
    require(doc["qids"] == expected and len(doc["class_indices"]) == len(expected), "Frozen Validation predictions misaligned")
    return np.asarray(doc["class_indices"], dtype=int)


def predict(paths, args):
    split = args.split
    frozen, protocol_source = protocol(paths, create=split == "val")
    if split == "test":
        val = paths.evaluation / RESULT_DIRS["val"]
        completed = read_json(val / "complete.json")
        require(completed == {"protocol": protocol_source, "results": provenance(val / "results.csv"),
                              "split": "val", "count": 500}, "Complete the frozen Validation evaluation before Test")
        device = setup_device(paths)
        from experiments.hotpotqa_hierarchical_topk.scripts.train import tokenizer_for

        tokenizer = tokenizer_for(paths)
    datasets, _ = load_data(paths, (split,))
    labels = read_labels(paths, datasets)[split]
    all_rows = np.ones(len(labels), dtype=bool)
    true_low = np.asarray([row["stage1_label"] == 0 for row in labels])
    directory = paths.evaluation / RESULT_DIRS[split] / "predictions"
    for experiment, (slug, heads, _) in EXPERIMENTS.items():
        chosen = frozen["choices"][experiment]["heads"]

        def infer(head, mask):
            if split == "test":
                return model_predictions(paths, experiment, head, chosen[head], labels, mask, tokenizer, device)
            candidates = validation_predictions(paths, experiment, head, chosen[head], labels)
            require(np.all(np.isin(candidates, np.arange(len(HEAD_CLASSES[head])))), "Invalid class in Validation cache")
            return np.where(mask, candidates, -1)

        first = infer(heads[0], all_rows)
        if len(heads) == 2:
            routed = infer("stage2", first == 0)
            standalone = infer("stage2", true_low)
            selected = route(first, routed)
            prediction = [{"stage1_pred": int(a), "stage2_routed_pred": int(b) if b >= 0 else None,
                           "stage2_standalone_pred": int(c) if c >= 0 else None} for a, b, c in zip(first, routed, standalone)]
        else:
            selected = np.asarray(CLASSES)[first]
            prediction = [{"flat5_pred": int(cls)} for cls in first]
        output = directory / f"{slug}.jsonl"
        write_jsonl(output, ({"qid": row["qid"], "selected_k": int(k), **pred}
                             for row, k, pred in zip(labels, selected, prediction)))
        atomic_json(directory / f"{slug}.json", {"protocol": protocol_source, "split": split, "experiment": experiment,
                                                  "predictions": provenance(output), "count": len(labels)})
        print(f"PREDICTIONS {split}/{experiment}: {len(labels)}", flush=True)


def options(parser):
    parser.add_argument("--split", choices=tuple(RESULT_DIRS), required=True)


if __name__ == "__main__":
    entrypoint(predict, __doc__, options)
