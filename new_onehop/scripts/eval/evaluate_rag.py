#!/usr/bin/env python3
"""Replay validation predictions and fixed-k baselines on stored RAG curves."""

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

from new_onehop.src.dataset import (
    EXPECTED_ORACLE_COUNT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VAL_COUNT,
    load_curve_records,
    load_prediction_records,
    load_split_ids,
)
from new_onehop.src.metrics import (
    compute_dynamic_rag_utility,
    compute_dynamic_rag_metrics,
    compute_fixed_k_rag_metrics,
    compute_global_max_context_tokens,
)
from new_onehop.src.utils import (
    ConfigError,
    load_yaml_config,
    project_path,
    require_mapping,
    write_json_atomic,
)


DEFAULT_CONFIG = PROJECT_ROOT / "new_onehop/configs/train_multiclass.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--curves", type=Path, default=None)
    parser.add_argument("--train-ids", type=Path, default=None)
    parser.add_argument("--val-ids", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    data_config = require_mapping(config, "data")
    output_config = require_mapping(config, "output")
    model_config = require_mapping(config, "model")
    model_type = model_config.get("type", "multiclass")
    if model_type not in ("multiclass", "ordinal"):
        raise ConfigError(
            f"model.type must be 'multiclass' or 'ordinal', got {model_type!r}"
        )
    selection_config = config.get("selection", {})
    if not isinstance(selection_config, dict):
        raise ConfigError("Config field 'selection' must be a mapping when present")
    lambda_T = selection_config.get("rag_utility_lambda_T", 0.1)
    curves_path = args.curves or project_path(data_config["train_curves"])
    split_path = (
        args.train_ids or project_path(data_config["train_ids"])
        if args.split == "train"
        else args.val_ids or project_path(data_config["val_ids"])
    )
    expected_count = (
        EXPECTED_TRAIN_COUNT if args.split == "train" else EXPECTED_VAL_COUNT
    )
    split_ids = load_split_ids(split_path, expected_count)
    curves = load_curve_records(curves_path, expected_count=EXPECTED_ORACLE_COUNT)
    predictions = load_prediction_records(args.predictions)

    prediction_ids = set(predictions)
    expected_ids = set(split_ids)
    missing = expected_ids - prediction_ids
    extra = prediction_ids - expected_ids
    if missing or extra:
        raise ValueError(
            f"Prediction/split ID mismatch: missing={sorted(missing)[:10]}, "
            f"extra={sorted(extra)[:10]}"
        )
    for question_id in split_ids:
        curve = curves.get(question_id)
        if curve is None:
            raise ValueError(f"Missing curve for split ID {question_id!r}")
        prediction = predictions[question_id]
        if prediction.dataset != curve.dataset:
            raise ValueError(
                f"{question_id}: prediction dataset {prediction.dataset!r} "
                f"does not match curve dataset {curve.dataset!r}"
            )

    predicted_k_by_id = {
        question_id: predictions[question_id].predicted_k for question_id in split_ids
    }
    dynamic_metrics = compute_dynamic_rag_metrics(predicted_k_by_id, curves, split_ids)
    fixed_k_baselines = compute_fixed_k_rag_metrics(curves, split_ids)
    t_max = compute_global_max_context_tokens(curves)
    rag_utility = compute_dynamic_rag_utility(
        predicted_k_by_id,
        curves,
        split_ids,
        lambda_T=lambda_T,
        t_max=t_max,
    )
    prediction_path = project_path(args.predictions)
    run_name = prediction_path.stem
    output_root = project_path(output_config["root"])
    output_path = (
        project_path(args.output)
        if args.output
        else (output_root / "metrics" / args.split / f"{run_name}_rag.json")
    )
    payload: dict[str, Any] = {
        "split": args.split,
        "run_name": run_name,
        "model_type": model_type,
        "num_predictions": len(predictions),
        "dynamic_model": dynamic_metrics,
        "rag_utility_lambda_T": lambda_T,
        "rag_utility_t_max": t_max,
        "val_rag_utility" if args.split == "val" else "rag_utility": rag_utility,
        "fixed_k_baselines": fixed_k_baselines,
        "always_k_1_rag_baseline": fixed_k_baselines["fixed_k_1"],
        "baseline_note": (
            "always_k_1_rag_baseline is the RAG replay of fixed k=1; "
            "predictor majority-class accuracy is reported by evaluate_predictor.py"
        ),
    }
    # Preserve the Phase 1 field for existing consumers while exposing the
    # model-agnostic dynamic_model field for Phase 2 and later predictors.
    if model_type == "multiclass":
        payload["dynamic_multiclass"] = dynamic_metrics
    write_json_atomic(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved RAG replay metrics: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
