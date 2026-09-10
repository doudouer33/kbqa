#!/usr/bin/env python3
"""Evaluate a Phase 2 ordinal checkpoint against hard Oracle labels."""

# ruff: noqa: E402 -- project root must be added before local package imports.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from new_onehop.src.dataset import (
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VAL_COUNT,
    QuestionBatchCollator,
    build_train_val_datasets,
)
from new_onehop.src.metrics import (
    compute_always_k1_predictor_baseline,
    compute_grouped_predictor_metrics,
)
from new_onehop.src.models import OrdinalTopKPredictor
from new_onehop.src.utils import (
    ConfigError,
    load_yaml_config,
    predict_ordinal,
    project_path,
    require_mapping,
    select_device,
    set_random_seed,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_CONFIG = PROJECT_ROOT / "new_onehop/configs/train_ordinal.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--oracle", type=Path, default=None)
    parser.add_argument("--train-ids", type=Path, default=None)
    parser.add_argument("--val-ids", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    return parser.parse_args()


def resolve_checkpoint_dir(path: Path) -> Path:
    checkpoint = project_path(path)
    if checkpoint.is_file():
        if checkpoint.name != "model.pt":
            raise ConfigError(
                f"Checkpoint file must be named model.pt, got {checkpoint}"
            )
        checkpoint = checkpoint.parent
    if (checkpoint / "model.pt").is_file():
        return checkpoint
    available = [
        name
        for name in ("best_val_loss", "best_rag_utility")
        if (checkpoint / name / "model.pt").is_file()
    ]
    if available:
        raise ConfigError(
            "Checkpoint run contains multiple selection targets; choose one of: "
            + ", ".join(str(checkpoint / name) for name in available)
        )
    raise ConfigError(f"Could not find model.pt in {checkpoint}")


def load_effective_config(checkpoint_dir: Path, fallback_path: Path) -> dict[str, Any]:
    checkpoint_config = checkpoint_dir / "config.yaml"
    return load_yaml_config(
        checkpoint_config if checkpoint_config.is_file() else fallback_path
    )


def load_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"Cannot read checkpoint metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ConfigError(f"Checkpoint metadata must be an object: {metadata_path}")
    return metadata


def main() -> None:
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    config = load_effective_config(checkpoint_dir, args.config)
    data_config = require_mapping(config, "data")
    model_config = require_mapping(config, "model")
    training_config = require_mapping(config, "training")
    output_config = require_mapping(config, "output")
    if model_config.get("type") != "ordinal":
        raise ConfigError("Only a Phase 2 ordinal checkpoint is supported")
    if model_config.get("num_labels") != 15 or model_config.get("num_thresholds") != 14:
        raise ConfigError("Ordinal evaluator requires 15 labels and 14 thresholds")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"seed must be an integer, got {seed!r}")
    set_random_seed(seed)
    device = select_device()
    mixed_precision = bool(training_config.get("mixed_precision", False))

    oracle_path = args.oracle or project_path(data_config["train_oracle"])
    train_ids_path = args.train_ids or project_path(data_config["train_ids"])
    val_ids_path = args.val_ids or project_path(data_config["val_ids"])
    tokenizer_path = checkpoint_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path.is_dir() else model_config["name"],
        use_fast=True,
    )
    train_dataset, val_dataset = build_train_val_datasets(
        oracle_path,
        train_ids_path,
        val_ids_path,
        tokenizer,
        max_length=model_config["max_length"],
    )
    dataset = train_dataset if args.split == "train" else val_dataset
    expected_count = (
        EXPECTED_TRAIN_COUNT if args.split == "train" else EXPECTED_VAL_COUNT
    )
    collator = QuestionBatchCollator(
        tokenizer,
        pad_to_multiple_of=(8 if mixed_precision and device.type == "cuda" else None),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = OrdinalTopKPredictor(
        model_name=model_config["name"],
        num_labels=model_config["num_labels"],
        num_thresholds=model_config["num_thresholds"],
        dropout=model_config["dropout"],
        threshold_init_min=model_config["threshold_init_min"],
        threshold_init_max=model_config["threshold_init_max"],
    )
    try:
        checkpoint_payload = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"Cannot load checkpoint {checkpoint_dir / 'model.pt'}: {exc}"
        ) from exc
    if (
        not isinstance(checkpoint_payload, dict)
        or "model_state_dict" not in checkpoint_payload
    ):
        raise ConfigError(f"Invalid checkpoint payload: {checkpoint_dir / 'model.pt'}")
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    model.to(device)
    prediction = predict_ordinal(
        model,
        data_loader,
        device,
        mixed_precision,
        num_thresholds=model_config["num_thresholds"],
    )
    if len(prediction.rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {args.split} predictions, got {len(prediction.rows)}"
        )

    grouped_metrics = compute_grouped_predictor_metrics(
        prediction.predicted_ks,
        prediction.oracle_ks,
        prediction.dataset_names,
    )
    always_k1 = compute_always_k1_predictor_baseline(
        prediction.oracle_ks, prediction.dataset_names
    )
    metadata = load_metadata(checkpoint_dir)
    selection_name = checkpoint_dir.name
    if selection_name not in ("best_val_loss", "best_rag_utility"):
        selection_name = str(metadata.get("selection_metric", "checkpoint"))
    run_name = output_config["run_name"]
    artifact_name = f"{run_name}_{selection_name}"
    output_root = project_path(output_config["root"])
    metrics_path = (
        project_path(args.output)
        if args.output
        else (output_root / "metrics" / args.split / f"{artifact_name}_predictor.json")
    )
    predictions_path = (
        project_path(args.predictions_output)
        if args.predictions_output
        else (output_root / "predictions" / args.split / f"{artifact_name}.jsonl")
    )
    payload: dict[str, Any] = {
        "split": args.split,
        "run_name": run_name,
        "model_type": "ordinal",
        "checkpoint": str(checkpoint_dir),
        "checkpoint_selection_metric": metadata.get("selection_metric"),
        "checkpoint_selection_value": metadata.get("selection_value"),
        "loss": prediction.loss,
        "decoding": "categorical_argmax_from_ordinal_distribution",
        "thresholds": prediction.thresholds,
        "numerical_sanity": prediction.numerical_sanity,
        **grouped_metrics,
        "always_k_1_predictor_baseline": always_k1,
    }
    write_jsonl_atomic(predictions_path, prediction.rows)
    write_json_atomic(metrics_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
