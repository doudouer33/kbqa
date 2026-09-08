#!/usr/bin/env python3
"""Validate Oracle splits and report Hard/Soft Oracle statistics.

This script is read-only: it only loads the generated Oracle and split files and
prints validation results and descriptive statistics to stdout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "new_onehop/configs/data_config.yaml"
EXPECTED_ORACLE_COUNT = 9000
EXPECTED_TRAIN_COUNT = 8100
EXPECTED_VAL_COUNT = 900


class DataError(RuntimeError):
    """Raised when an input file or Oracle row is invalid."""


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


def read_id_list(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8") as file:
            values = json.load(file)
    except FileNotFoundError as exc:
        raise DataError(f"Missing ID file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(values, list):
        raise DataError(f"ID file must contain a JSON array: {path}")
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise DataError(f"Invalid ID at {path}[{index}]: {value!r}")
    return values


def read_oracle_rows(
    path: Path,
    datasets: list[str],
    k_values: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_soft_keys = {str(k) for k in k_values}
    try:
        file = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DataError(f"Missing Oracle file: {path}") from exc

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
            dataset = row.get("dataset")
            if dataset not in datasets:
                raise DataError(
                    f"{qid}: dataset must be one of {datasets}, got {dataset!r}"
                )
            hard_k = row.get("hard_oracle_k")
            if isinstance(hard_k, bool) or not isinstance(hard_k, int):
                raise DataError(f"{qid}: invalid hard_oracle_k {hard_k!r}")
            if hard_k not in k_values:
                raise DataError(f"{qid}: hard_oracle_k is outside k={k_values}")

            distribution = row.get("soft_oracle_distribution")
            if not isinstance(distribution, dict):
                raise DataError(f"{qid}: soft_oracle_distribution is not an object")
            if set(distribution) != expected_soft_keys:
                missing = sorted(expected_soft_keys - set(distribution), key=int)
                extra = sorted(set(distribution) - expected_soft_keys)
                raise DataError(
                    f"{qid}: invalid soft distribution keys; missing={missing}, extra={extra}"
                )
            probabilities = []
            for k in k_values:
                probability = distribution[str(k)]
                if (
                    isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not math.isfinite(probability)
                    or probability < 0.0
                ):
                    raise DataError(f"{qid}: invalid soft probability at k={k}: {probability!r}")
                probabilities.append(float(probability))
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise DataError(
                    f"{qid}: soft probabilities sum to {sum(probabilities):.12g}, not 1"
                )
            rows.append(row)
    return rows


def duplicate_ids(values: list[str]) -> set[str]:
    return {value for value, count in Counter(values).items() if count > 1}


def check_line(label: str, actual: int, expected: int) -> bool:
    passed = actual == expected
    print(f"{label}: {actual} (expected {expected}) [{'PASS' if passed else 'FAIL'}]")
    return passed


def preview_ids(values: set[str], limit: int = 10) -> str:
    if not values:
        return "none"
    ordered = sorted(values)
    preview = ", ".join(ordered[:limit])
    if len(ordered) > limit:
        preview += f", ... (+{len(ordered) - limit} more)"
    return preview


def print_set_check(label: str, values: set[str]) -> bool:
    passed = not values
    print(f"{label}: {len(values)} [{'PASS' if passed else 'FAIL'}]")
    if values:
        print(f"  IDs: {preview_ids(values)}")
    return passed


def print_distribution(counter: Counter[int], total: int, k_values: list[int]) -> None:
    print("k      count    percent")
    for k in k_values:
        count = counter[k]
        percent = 100.0 * count / total if total else 0.0
        print(f"{k:>2}  {count:>9}  {percent:>8.3f}%")


def soft_argmax(distribution: dict[str, float], k_values: list[int]) -> int:
    # The second tuple item gives deterministic smallest-k tie-breaking.
    return max(k_values, key=lambda k: (distribution[str(k)], -k))


def entropy(distribution: dict[str, float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in distribution.values()
        if probability > 0.0
    )


def main() -> None:
    args = parse_args()
    config = read_config(args.config)

    datasets_config = config.get("datasets")
    if not isinstance(datasets_config, dict) or not datasets_config:
        raise DataError("datasets must be a non-empty mapping")
    datasets = list(datasets_config)
    k_values = config.get("k_values")
    if k_values != list(range(1, 16)):
        raise DataError(f"k_values must be exactly 1..15, got {k_values!r}")

    oracle_config = config.get("oracle")
    split_config = config.get("id_splits")
    if not isinstance(oracle_config, dict):
        raise DataError("Missing oracle configuration")
    if not isinstance(split_config, dict):
        raise DataError("Missing id_splits configuration")

    oracle_path = project_path(oracle_config["output_dir"]) / "all.jsonl"
    train_path = project_path(split_config["train_ids_file"])
    val_path = project_path(split_config["val_ids_file"])
    rows = read_oracle_rows(oracle_path, datasets, k_values)
    train_ids = read_id_list(train_path)
    val_ids = read_id_list(val_path)

    oracle_ids_list = [row["question_id"] for row in rows]
    oracle_ids = set(oracle_ids_list)
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    split_ids = train_id_set | val_id_set
    checks: list[bool] = []

    print("1. Count validation")
    checks.append(check_line("Oracle rows", len(rows), EXPECTED_ORACLE_COUNT))
    checks.append(check_line("Train IDs", len(train_ids), EXPECTED_TRAIN_COUNT))
    checks.append(check_line("Validation IDs", len(val_ids), EXPECTED_VAL_COUNT))
    print()

    print("2. Train/validation overlap")
    checks.append(print_set_check("Overlapping IDs", train_id_set & val_id_set))
    print()

    print("3. Missing and duplicate IDs")
    checks.append(print_set_check("Train IDs missing from Oracle", train_id_set - oracle_ids))
    checks.append(print_set_check("Validation IDs missing from Oracle", val_id_set - oracle_ids))
    checks.append(print_set_check("Oracle IDs missing from splits", oracle_ids - split_ids))
    checks.append(print_set_check("Duplicate Oracle IDs", duplicate_ids(oracle_ids_list)))
    checks.append(print_set_check("Duplicate train IDs", duplicate_ids(train_ids)))
    checks.append(print_set_check("Duplicate validation IDs", duplicate_ids(val_ids)))
    print()

    hard_distribution = Counter(row["hard_oracle_k"] for row in rows)
    print("4. Hard Oracle k distribution (overall)")
    print_distribution(hard_distribution, len(rows), k_values)
    print()

    print("5. Hard Oracle k distribution by dataset")
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        print(f"Dataset: {dataset} (n={len(dataset_rows)})")
        print_distribution(
            Counter(row["hard_oracle_k"] for row in dataset_rows),
            len(dataset_rows),
            k_values,
        )
        print()

    hard_values = [row["hard_oracle_k"] for row in rows]
    soft_argmax_values = [
        soft_argmax(row["soft_oracle_distribution"], k_values) for row in rows
    ]
    distances = [
        abs(hard_k - soft_k)
        for hard_k, soft_k in zip(hard_values, soft_argmax_values)
    ]
    conflicts = sum(distance != 0 for distance in distances)

    print("6. Hard/Soft conflict rate")
    print(f"Conflicts: {conflicts}/{len(rows)}")
    print(f"Conflict rate: {conflicts / len(rows):.6f} ({100.0 * conflicts / len(rows):.3f}%)")
    print()

    print("7. Hard/Soft argmax distance")
    print(f"Mean absolute k distance: {statistics.fmean(distances):.6f}")
    print(f"Median absolute k distance: {statistics.median(distances):.6f}")
    print()

    entropies = [entropy(row["soft_oracle_distribution"]) for row in rows]
    print("8. Soft Oracle entropy (natural log, nats)")
    print(f"Mean: {statistics.fmean(entropies):.6f}")
    print(f"Median: {statistics.median(entropies):.6f}")
    print(f"Min: {min(entropies):.6f}")
    print(f"Max: {max(entropies):.6f}")
    print()

    max_soft_probabilities = [
        max(row["soft_oracle_distribution"].values()) for row in rows
    ]
    print("9. Maximum Soft Oracle probability")
    print(f"Mean: {statistics.fmean(max_soft_probabilities):.6f}")
    print(f"Max: {max(max_soft_probabilities):.6f}")
    print(f"Min: {min(max_soft_probabilities):.6f}")
    print()

    if all(checks):
        print("Validation result: PASS")
    else:
        print("Validation result: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (DataError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
