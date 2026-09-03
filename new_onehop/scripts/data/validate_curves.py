#!/usr/bin/env python3
"""Validate train or test query-level retrieval curve JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "new_onehop/configs/data_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_score(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def validate_file(
    path: Path,
    expected_dataset: str | None,
    expected_count: int,
    expected_k: set[str],
    allow_unvalidated: bool,
) -> tuple[list[dict[str, Any]], list[str], int, int]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    empty_predictions = 0
    unvalidated_contexts = 0

    try:
        file = path.open(encoding="utf-8")
    except FileNotFoundError:
        return [], [f"missing file: {path}"], 0, 0

    with file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_number}: row is not an object")
                continue
            rows.append(row)
            qid = row.get("question_id")
            prefix = f"line {line_number} ({qid!r})"
            if not isinstance(qid, str) or not qid:
                errors.append(f"{prefix}: invalid question_id")
            elif qid in seen:
                errors.append(f"{prefix}: duplicate question_id")
            else:
                seen.add(qid)

            if expected_dataset is not None and row.get("dataset") != expected_dataset:
                errors.append(f"{prefix}: expected dataset {expected_dataset!r}")
            if not isinstance(row.get("question"), str) or not row["question"]:
                errors.append(f"{prefix}: missing question")
            gold = row.get("gold_answer")
            if (
                not isinstance(gold, list)
                or not gold
                or any(not isinstance(answer, str) or not answer for answer in gold)
            ):
                errors.append(f"{prefix}: invalid gold_answer")

            results = row.get("results")
            if not isinstance(results, dict):
                errors.append(f"{prefix}: results is not an object")
                continue
            actual_k = set(results)
            if actual_k != expected_k:
                errors.append(
                    f"{prefix}: missing k={sorted(expected_k - actual_k)}, "
                    f"extra k={sorted(actual_k - expected_k)}"
                )
            for k, result in results.items():
                if not isinstance(result, dict):
                    errors.append(f"{prefix}: result k={k} is not an object")
                    continue
                prediction = result.get("prediction")
                if not isinstance(prediction, str):
                    errors.append(f"{prefix}: missing prediction at k={k}")
                elif prediction == "":
                    empty_predictions += 1
                if not is_score(result.get("em")):
                    errors.append(f"{prefix}: invalid EM at k={k}")
                if not is_score(result.get("f1")):
                    errors.append(f"{prefix}: invalid F1 at k={k}")
                tokens = result.get("context_tokens")
                if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                    errors.append(f"{prefix}: invalid context_tokens at k={k}")
                validated = result.get("context_tokens_validated")
                if not isinstance(validated, bool):
                    errors.append(f"{prefix}: invalid context_tokens_validated at k={k}")
                elif not validated:
                    unvalidated_contexts += 1
                    if not allow_unvalidated:
                        errors.append(f"{prefix}: unvalidated context tokens at k={k}")

    if len(rows) != expected_count:
        errors.append(f"expected {expected_count} rows, found {len(rows)}")
    return rows, errors, empty_predictions, unvalidated_contexts


def main() -> None:
    args = parse_args()
    try:
        with args.config.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as exc:
        raise SystemExit(f"ERROR: cannot read config {args.config}: {exc}") from exc

    split_config = config.get("splits", {}).get(args.split)
    if not isinstance(split_config, dict):
        raise SystemExit(f"ERROR: missing splits.{args.split} configuration")
    curve_dir = project_path(split_config["curve_output_dir"])
    expected_per_dataset = split_config["expected_questions_per_dataset"]
    allow_unvalidated = bool(split_config.get("allow_unvalidated_context_tokens", False))
    datasets = list(config["datasets"])
    expected_k = {str(k) for k in config["k_values"]}
    all_dataset_rows: list[dict[str, Any]] = []
    any_errors = False

    for dataset in datasets:
        rows, errors, empty, unvalidated = validate_file(
            curve_dir / f"{dataset}.jsonl",
            expected_dataset=dataset,
            expected_count=expected_per_dataset,
            expected_k=expected_k,
            allow_unvalidated=allow_unvalidated,
        )
        all_dataset_rows.extend(rows)
        print(f"Dataset: {args.split}/{dataset}")
        print(f"Questions: {len(rows)}")
        print(f"Empty (but present) predictions: {empty}")
        print(f"Unvalidated context-token observations: {unvalidated}")
        print(f"Errors: {len(errors)}")
        for error in errors[:20]:
            print(f"  - {error}")
        print("PASS" if not errors else "FAIL")
        print()
        any_errors |= bool(errors)

    all_rows, all_errors, all_empty, all_unvalidated = validate_file(
        curve_dir / "all.jsonl",
        expected_dataset=None,
        expected_count=expected_per_dataset * len(datasets),
        expected_k=expected_k,
        allow_unvalidated=allow_unvalidated,
    )
    if all_rows != all_dataset_rows:
        all_errors.append("all.jsonl is not the exact dataset-file concatenation")
    if any(row.get("dataset") not in datasets for row in all_rows):
        all_errors.append("all.jsonl contains an unexpected dataset")
    print(f"Dataset: {args.split}/all")
    print(f"Questions: {len(all_rows)}")
    print(f"Empty (but present) predictions: {all_empty}")
    print(f"Unvalidated context-token observations: {all_unvalidated}")
    print(f"Errors: {len(all_errors)}")
    for error in all_errors[:20]:
        print(f"  - {error}")
    print("PASS" if not all_errors else "FAIL")
    any_errors |= bool(all_errors)

    if any_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
