#!/usr/bin/env python3
"""Build train or test query-level retrieval curves from fixed-k RAG outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "new_onehop/configs/data_config.yaml"


class DataError(RuntimeError):
    """Raised when an input violates the curve data contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise DataError(f"Missing input file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataError(f"Invalid JSON in {path}: {exc}") from exc


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


def read_source(path: Path, dataset: str) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    questions: dict[str, str] = {}
    try:
        file = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DataError(f"Missing source file: {path}") from exc

    with file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

            qid = row.get("question_id")
            question = row.get("question_text")
            if row.get("dataset") != dataset:
                raise DataError(
                    f"Dataset mismatch at {path}:{line_number}: "
                    f"expected {dataset!r}, got {row.get('dataset')!r}"
                )
            if not isinstance(qid, str) or not qid:
                raise DataError(f"Invalid question_id at {path}:{line_number}")
            if qid in questions:
                raise DataError(f"Duplicate question_id {qid!r} in {path}")
            if not isinstance(question, str) or not question:
                raise DataError(f"Missing question text for {qid!r} in {path}")
            order.append(qid)
            questions[qid] = question

    return order, questions


def require_score(value: Any, field: str, dataset: str, k: int, qid: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{dataset} k={k} {qid}: {field} is not numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise DataError(f"{dataset} k={k} {qid}: invalid {field}={value!r}")
    return score


def index_evaluations(path: Path, dataset: str, k: int) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise DataError(f"Expected a JSON array in {path}")

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise DataError(f"{dataset} k={k}: evaluation row is not an object")
        qid = row.get("id")
        if not isinstance(qid, str) or not qid:
            raise DataError(f"{dataset} k={k}: invalid evaluation id")
        if qid in indexed:
            raise DataError(f"{dataset} k={k}: duplicate evaluation id {qid!r}")
        indexed[qid] = row
    return indexed


def find_reconstruction(experiment_dir: Path, payload: dict[str, Any]) -> Path:
    configured = payload.get("source_reconstruction", {}).get("path")
    if isinstance(configured, str) and Path(configured).is_file():
        return Path(configured)
    matches = sorted(
        path
        for path in experiment_dir.glob("reconstructed_inputs__*.json")
        if "__limit_" not in path.stem
    )
    if len(matches) != 1:
        raise DataError(
            f"Expected one complete reconstruction in {experiment_dir}, found {len(matches)}"
        )
    return matches[0]


def recover_unvalidated_token_counts(
    experiment_dir: Path,
    payload: dict[str, Any],
    missing_ids: set[str],
    dataset: str,
    k: int,
) -> dict[str, int]:
    if not missing_ids:
        return {}
    try:
        import tiktoken
    except ImportError as exc:
        raise DataError(
            "tiktoken is required to retain unvalidated test examples"
        ) from exc

    encoding_name = payload.get("encoding")
    if not isinstance(encoding_name, str) or not encoding_name:
        raise DataError(f"{dataset} k={k}: token-count file has no encoding")
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError as exc:
        raise DataError(f"{dataset} k={k}: unknown encoding {encoding_name!r}") from exc

    reconstruction_path = find_reconstruction(experiment_dir, payload)
    reconstruction = read_json(reconstruction_path)
    items = reconstruction.get("items") if isinstance(reconstruction, dict) else None
    if not isinstance(items, list):
        raise DataError(f"Invalid reconstruction file: {reconstruction_path}")
    indexed = {item.get("qid"): item for item in items if isinstance(item, dict)}

    recovered: dict[str, int] = {}
    for qid in missing_ids:
        item = indexed.get(qid)
        rendered = item.get("rendered_input") if isinstance(item, dict) else None
        context = rendered.get("context_text") if isinstance(rendered, dict) else None
        if not isinstance(context, str):
            raise DataError(f"{dataset} k={k} {qid}: missing reconstructed context")
        recovered[qid] = len(encoding.encode(context, disallowed_special=()))
    return recovered


def read_context_token_data(
    path: Path,
    experiment_dir: Path,
    dataset: str,
    k: int,
    allow_unvalidated: bool,
) -> tuple[dict[str, int], dict[str, bool]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("token_counts"), dict):
        raise DataError(f"Missing token_counts object in {path}")
    if payload.get("count_type") != "retrieved_context_tokens":
        raise DataError(
            f"{dataset} k={k}: expected retrieved_context_tokens, "
            f"got {payload.get('count_type')!r}"
        )

    counts: dict[str, int] = {}
    validity: dict[str, bool] = {}
    for qid, value in payload["token_counts"].items():
        if not isinstance(qid, str) or not qid:
            raise DataError(f"{dataset} k={k}: invalid token-count id")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DataError(f"{dataset} k={k} {qid}: invalid context_tokens={value!r}")
        counts[qid] = value
        validity[qid] = True

    skipped = payload.get("skipped")
    if not isinstance(skipped, list):
        raise DataError(f"{dataset} k={k}: skipped must be a list")
    skipped_ids: set[str] = set()
    for entry in skipped:
        qid = entry.get("qid") if isinstance(entry, dict) else None
        if not isinstance(qid, str) or not qid or qid in skipped_ids or qid in counts:
            raise DataError(f"{dataset} k={k}: invalid or duplicate skipped ID {qid!r}")
        skipped_ids.add(qid)

    if skipped_ids and not allow_unvalidated:
        raise DataError(
            f"{dataset} k={k}: {len(skipped_ids)} contexts failed reconstruction validation"
        )
    recovered = recover_unvalidated_token_counts(
        experiment_dir, payload, skipped_ids, dataset, k
    )
    counts.update(recovered)
    validity.update({qid: False for qid in recovered})
    return counts, validity


def require_same_ids(
    expected: set[str], actual: set[str], dataset: str, k: int, source_name: str
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise DataError(
            f"{dataset} k={k}: {source_name} IDs do not match source IDs; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def artifact_path(
    experiment_dir: Path, prefix: str, dataset: str, artifact_suffix: str
) -> Path:
    return experiment_dir / f"{prefix}__{dataset}_to_{dataset}__{artifact_suffix}.json"


def build_dataset(
    dataset: str,
    spec: dict[str, Any],
    split_config: dict[str, Any],
    k_values: list[int],
) -> list[dict[str, Any]]:
    prediction_root = project_path(split_config["prediction_root"])
    source_root = project_path(split_config["source_root"])
    source_file = split_config["source_file_template"].format(dataset=dataset)
    artifact_suffix = split_config["artifact_suffix"]
    expected_count = split_config.get("expected_questions_per_dataset")
    allow_unvalidated = bool(split_config.get("allow_unvalidated_context_tokens", False))

    order, questions = read_source(source_root / source_file, dataset)
    if expected_count is not None and len(order) != expected_count:
        raise DataError(
            f"{dataset}: expected {expected_count} source questions, found {len(order)}"
        )

    source_ids = set(order)
    records: dict[str, dict[str, Any]] = {
        qid: {
            "question_id": qid,
            "dataset": dataset,
            "question": questions[qid],
            "gold_answer": None,
            "results": {},
        }
        for qid in order
    }

    for k in k_values:
        experiment_dir = prediction_root / spec["dir_pattern"].format(k=k)
        evaluation_path = artifact_path(
            experiment_dir, "per_question_eval", dataset, artifact_suffix
        )
        token_path = artifact_path(
            experiment_dir, "retrieval_context_token_counts", dataset, artifact_suffix
        )
        evaluations = index_evaluations(evaluation_path, dataset, k)
        token_counts, token_validity = read_context_token_data(
            token_path, experiment_dir, dataset, k, allow_unvalidated
        )
        require_same_ids(source_ids, set(evaluations), dataset, k, "evaluation")
        require_same_ids(source_ids, set(token_counts), dataset, k, "token-count")

        for qid in order:
            row = evaluations[qid]
            prediction = row.get("prediction")
            if not isinstance(prediction, str):
                raise DataError(f"{dataset} k={k} {qid}: missing prediction")
            if "predicted_answer" in row and row["predicted_answer"] != prediction:
                raise DataError(f"{dataset} k={k} {qid}: prediction fields disagree")

            gold_answers = row.get("gold_answers")
            if (
                not isinstance(gold_answers, list)
                or not gold_answers
                or any(not isinstance(answer, str) or not answer for answer in gold_answers)
            ):
                raise DataError(f"{dataset} k={k} {qid}: invalid gold_answers")
            if records[qid]["gold_answer"] is None:
                records[qid]["gold_answer"] = gold_answers
            elif records[qid]["gold_answer"] != gold_answers:
                raise DataError(f"{dataset} k={k} {qid}: gold answers changed across k")

            records[qid]["results"][str(k)] = {
                "prediction": prediction,
                "em": require_score(row.get("em"), "em", dataset, k, qid),
                "f1": require_score(row.get("f1"), "f1", dataset, k, qid),
                "context_tokens": token_counts[qid],
                "context_tokens_validated": token_validity[qid],
            }

    return [records[qid] for qid in order]


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


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    k_values = config.get("k_values")
    if k_values != list(range(1, 16)):
        raise DataError(f"k_values must be exactly 1..15, got {k_values!r}")
    split_config = config.get("splits", {}).get(args.split)
    if not isinstance(split_config, dict):
        raise DataError(f"Missing splits.{args.split} configuration")

    output_dir = project_path(split_config["curve_output_dir"])
    all_rows: list[dict[str, Any]] = []
    per_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset, spec in config["datasets"].items():
        rows = build_dataset(dataset, spec, split_config, k_values)
        per_dataset[dataset] = rows
        all_rows.extend(rows)
        unvalidated = sum(
            not result["context_tokens_validated"]
            for row in rows
            for result in row["results"].values()
        )
        print(
            f"Built {args.split}/{dataset}: {len(rows)} questions x {len(k_values)} k values; "
            f"unvalidated contexts={unvalidated}"
        )

    for dataset, rows in per_dataset.items():
        write_jsonl_atomic(output_dir / f"{dataset}.jsonl", rows)
    write_jsonl_atomic(output_dir / "all.jsonl", all_rows)
    print(f"Wrote {output_dir / 'all.jsonl'}: {len(all_rows)} questions")


if __name__ == "__main__":
    try:
        main()
    except (DataError, KeyError, TypeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
