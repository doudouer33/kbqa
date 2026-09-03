#!/usr/bin/env python3
"""Build hard and soft Oracle labels from query-level retrieval curves."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "new_onehop/configs/data_config.yaml"


class DataError(RuntimeError):
    """Raised when curve data or Oracle configuration is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise DataError(f"Missing config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise DataError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DataError(f"Config must contain a YAML object: {path}")
    return config


def read_curves(
    path: Path,
    expected_dataset: str,
    expected_count: int | None,
    k_values: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_k = {str(k) for k in k_values}
    try:
        file = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DataError(f"Missing curve file: {path}") from exc

    with file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise DataError(f"Expected an object at {path}:{line_number}")

            qid = row.get("question_id")
            if not isinstance(qid, str) or not qid:
                raise DataError(f"Invalid question_id at {path}:{line_number}")
            if qid in seen:
                raise DataError(f"Duplicate question_id {qid!r} in {path}")
            seen.add(qid)
            if row.get("dataset") != expected_dataset:
                raise DataError(
                    f"{qid}: expected dataset {expected_dataset!r}, got {row.get('dataset')!r}"
                )
            if not isinstance(row.get("question"), str) or not row["question"]:
                raise DataError(f"{qid}: missing question")

            results = row.get("results")
            if not isinstance(results, dict) or set(results) != expected_k:
                raise DataError(f"{qid}: results must contain exactly k={k_values}")
            for k in k_values:
                result = results[str(k)]
                if not isinstance(result, dict):
                    raise DataError(f"{qid} k={k}: result is not an object")
                f1 = result.get("f1")
                tokens = result.get("context_tokens")
                if (
                    isinstance(f1, bool)
                    or not isinstance(f1, (int, float))
                    or not math.isfinite(f1)
                    or not 0.0 <= f1 <= 1.0
                ):
                    raise DataError(f"{qid} k={k}: invalid F1 {f1!r}")
                if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                    raise DataError(f"{qid} k={k}: invalid context_tokens {tokens!r}")
                if result.get("context_tokens_validated") is not True:
                    raise DataError(
                        f"{qid} k={k}: Oracle labels require validated context tokens"
                    )
            rows.append(row)

    if expected_count is not None and len(rows) != expected_count:
        raise DataError(
            f"{expected_dataset}: expected {expected_count} curves, found {len(rows)}"
        )
    return rows


def require_number(value: Any, name: str, minimum: float, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"oracle.{name} must be numeric")
    number = float(value)
    valid = number > minimum if strict else number >= minimum
    if not math.isfinite(number) or not valid:
        operator = ">" if strict else ">="
        raise DataError(f"oracle.{name} must be finite and {operator} {minimum}")
    return number


def token_max(curves_by_dataset: dict[str, list[dict[str, Any]]]) -> int:
    try:
        return max(
            result["context_tokens"]
            for rows in curves_by_dataset.values()
            for row in rows
            for result in row["results"].values()
        )
    except ValueError as exc:
        raise DataError("Cannot compute T_max from empty curves") from exc


def softmax(utilities: list[float], temperature: float) -> list[float]:
    logits = [utility / temperature for utility in utilities]
    maximum = max(logits)
    weights = [math.exp(logit - maximum) for logit in logits]
    denominator = sum(weights)
    return [weight / denominator for weight in weights]


def build_oracle_row(
    curve: dict[str, Any],
    k_values: list[int],
    lambda_t: float,
    temperature: float,
    t_max: int,
) -> dict[str, Any]:
    results = curve["results"]
    maximum_f1 = max(results[str(k)]["f1"] for k in k_values)
    best_f1_k = [k for k in k_values if results[str(k)]["f1"] == maximum_f1]
    hard_oracle_k = min(
        best_f1_k,
        key=lambda k: (results[str(k)]["context_tokens"], k),
    )

    utilities = [
        results[str(k)]["f1"]
        - lambda_t * results[str(k)]["context_tokens"] / t_max
        for k in k_values
    ]
    probabilities = softmax(utilities, temperature)
    if any(not math.isfinite(probability) or probability < 0.0 for probability in probabilities):
        raise DataError(f"{curve['question_id']}: invalid soft Oracle probability")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise DataError(f"{curve['question_id']}: soft Oracle probabilities do not sum to 1")

    return {
        "question_id": curve["question_id"],
        "dataset": curve["dataset"],
        "question": curve["question"],
        "hard_oracle_k": hard_oracle_k,
        "soft_oracle_distribution": {
            str(k): probability for k, probability in zip(k_values, probabilities)
        },
    }


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def build_id_splits(
    curves_by_dataset: dict[str, list[dict[str, Any]]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < validation_fraction < 1.0:
        raise DataError("id_splits.validation_fraction must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataError("id_splits.seed must be an integer")

    train_ids: list[str] = []
    val_ids: list[str] = []
    for dataset, curves in curves_by_dataset.items():
        dataset_ids = [curve["question_id"] for curve in curves]
        random.Random(f"{seed}:{dataset}").shuffle(dataset_ids)
        validation_count = round(len(dataset_ids) * validation_fraction)
        val_ids.extend(dataset_ids[:validation_count])
        train_ids.extend(dataset_ids[validation_count:])
    return train_ids, val_ids


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    k_values = config.get("k_values")
    if k_values != list(range(1, 16)):
        raise DataError(f"k_values must be exactly 1..15, got {k_values!r}")

    oracle_config = config.get("oracle")
    if not isinstance(oracle_config, dict):
        raise DataError("Missing oracle configuration")
    lambda_t = require_number(oracle_config.get("lambda_T"), "lambda_T", 0.0)
    temperature = require_number(
        oracle_config.get("temperature"), "temperature", 0.0, strict=True
    )

    curve_dir = project_path(oracle_config["curve_input_dir"])
    output_dir = project_path(oracle_config["output_dir"])
    expected_count = oracle_config.get("expected_questions_per_dataset")
    curves_by_dataset = {
        dataset: read_curves(
            curve_dir / f"{dataset}.jsonl",
            expected_dataset=dataset,
            expected_count=expected_count,
            k_values=k_values,
        )
        for dataset in config["datasets"]
    }
    t_max = token_max(curves_by_dataset)

    all_oracle_rows: list[dict[str, Any]] = []
    oracle_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset, curves in curves_by_dataset.items():
        oracle_rows = [
            build_oracle_row(curve, k_values, lambda_t, temperature, t_max)
            for curve in curves
        ]
        oracle_by_dataset[dataset] = oracle_rows
        all_oracle_rows.extend(oracle_rows)
        print(f"Built {dataset}: {len(oracle_rows)} Oracle labels")

    for dataset, rows in oracle_by_dataset.items():
        write_jsonl_atomic(output_dir / f"{dataset}.jsonl", rows)
    write_jsonl_atomic(output_dir / "all.jsonl", all_oracle_rows)
    print(
        f"Parameters: lambda_T={lambda_t:g}, temperature={temperature:g}, "
        f"T_max={t_max} (global curve maximum)"
    )
    print(f"Wrote {output_dir / 'all.jsonl'}: {len(all_oracle_rows)} labels")

    split_config = config.get("id_splits")
    if not isinstance(split_config, dict):
        raise DataError("Missing id_splits configuration")
    validation_fraction = require_number(
        split_config.get("validation_fraction"),
        "id_splits.validation_fraction",
        0.0,
        strict=True,
    )
    train_ids, val_ids = build_id_splits(
        curves_by_dataset,
        validation_fraction=validation_fraction,
        seed=split_config.get("seed"),
    )
    train_ids_path = project_path(split_config["train_ids_file"])
    val_ids_path = project_path(split_config["val_ids_file"])
    write_json_atomic(train_ids_path, train_ids)
    write_json_atomic(val_ids_path, val_ids)
    print(
        f"Wrote stratified ID split: train={len(train_ids)}, val={len(val_ids)}, "
        f"seed={split_config['seed']}"
    )


if __name__ == "__main__":
    try:
        main()
    except (DataError, KeyError, TypeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
