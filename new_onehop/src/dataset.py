"""Strict data loading for the question-only Top-k prediction task."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from .utils import project_path


DATASETS = ("nq", "squad", "trivia")
NUM_LABELS = 15
EXPECTED_ORACLE_COUNT = 9000
EXPECTED_TRAIN_COUNT = 8100
EXPECTED_VAL_COUNT = 900


class DataValidationError(RuntimeError):
    """Raised when a data artifact violates the Phase 1 contract."""


@dataclass(frozen=True)
class OracleRecord:
    question_id: str
    dataset: str
    question: str
    hard_oracle_k: int


@dataclass(frozen=True)
class CurveResult:
    f1: float
    context_tokens: int
    context_tokens_validated: bool


@dataclass(frozen=True)
class CurveRecord:
    question_id: str
    dataset: str
    question: str
    results: dict[int, CurveResult]


@dataclass(frozen=True)
class PredictionRecord:
    question_id: str
    dataset: str
    predicted_k: int
    probabilities: tuple[float, ...]
    hard_oracle_k: int | None


def _read_jsonl_objects(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    input_path = project_path(path)
    rows: list[tuple[int, dict[str, Any]]] = []
    try:
        file = input_path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DataValidationError(f"Missing JSONL file: {input_path}") from exc
    with file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"Invalid JSON at {input_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise DataValidationError(
                    f"Expected a JSON object at {input_path}:{line_number}"
                )
            rows.append((line_number, row))
    return rows


def _require_nonempty_string(value: Any, field: str, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{location}: {field} must be a non-empty string")
    return value


def _require_dataset(value: Any, location: str) -> str:
    dataset = _require_nonempty_string(value, "dataset", location)
    if dataset not in DATASETS:
        raise DataValidationError(
            f"{location}: dataset must be one of {list(DATASETS)}, got {dataset!r}"
        )
    return dataset


def _require_k(value: Any, field: str, location: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= NUM_LABELS
    ):
        raise DataValidationError(
            f"{location}: {field} must be an integer in [1, {NUM_LABELS}], got {value!r}"
        )
    return value


def _require_score(value: Any, field: str, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise DataValidationError(
            f"{location}: {field} must be a finite number in [0, 1], got {value!r}"
        )
    return float(value)


def _validate_soft_distribution(value: Any, location: str) -> None:
    if not isinstance(value, dict):
        raise DataValidationError(
            f"{location}: soft_oracle_distribution must be an object"
        )
    expected_keys = {str(k) for k in range(1, NUM_LABELS + 1)}
    if set(value) != expected_keys:
        missing = sorted(expected_keys - set(value), key=int)
        extra = sorted(set(value) - expected_keys)
        raise DataValidationError(
            f"{location}: invalid soft distribution keys; missing={missing}, extra={extra}"
        )
    probabilities = [
        _require_score(value[str(k)], f"soft probability k={k}", location)
        for k in range(1, NUM_LABELS + 1)
    ]
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DataValidationError(
            f"{location}: soft probabilities sum to {sum(probabilities):.12g}, not 1"
        )


def load_oracle_records(
    path: str | Path,
    expected_count: int = EXPECTED_ORACLE_COUNT,
) -> dict[str, OracleRecord]:
    """Load Oracle rows while retaining only question text, label, and metadata."""
    input_path = project_path(path)
    records: dict[str, OracleRecord] = {}
    raw_rows = _read_jsonl_objects(input_path)
    if len(raw_rows) != expected_count:
        raise DataValidationError(
            f"{input_path}: expected {expected_count} Oracle rows, found {len(raw_rows)}"
        )
    for line_number, row in raw_rows:
        location = f"{input_path}:{line_number}"
        question_id = _require_nonempty_string(
            row.get("question_id"), "question_id", location
        )
        if question_id in records:
            raise DataValidationError(
                f"{location}: duplicate question_id {question_id!r}"
            )
        dataset = _require_dataset(row.get("dataset"), location)
        question = _require_nonempty_string(row.get("question"), "question", location)
        hard_oracle_k = _require_k(row.get("hard_oracle_k"), "hard_oracle_k", location)
        # Validate the complete source schema but deliberately do not retain the
        # soft distribution in this Phase 1 dataset.
        _validate_soft_distribution(row.get("soft_oracle_distribution"), location)
        records[question_id] = OracleRecord(
            question_id=question_id,
            dataset=dataset,
            question=question,
            hard_oracle_k=hard_oracle_k,
        )
    return records


def load_split_ids(path: str | Path, expected_count: int) -> list[str]:
    input_path = project_path(path)
    try:
        with input_path.open(encoding="utf-8") as file:
            values = json.load(file)
    except FileNotFoundError as exc:
        raise DataValidationError(f"Missing split file: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"Invalid JSON in {input_path}: {exc}") from exc
    if not isinstance(values, list):
        raise DataValidationError(f"{input_path}: split file must contain a JSON array")
    if len(values) != expected_count:
        raise DataValidationError(
            f"{input_path}: expected {expected_count} IDs, found {len(values)}"
        )
    ids: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        question_id = _require_nonempty_string(
            value, "split ID", f"{input_path}[{index}]"
        )
        if question_id in seen:
            raise DataValidationError(
                f"{input_path}[{index}]: duplicate split ID {question_id!r}"
            )
        seen.add(question_id)
        ids.append(question_id)
    return ids


def load_oracle_splits(
    oracle_path: str | Path,
    train_ids_path: str | Path,
    val_ids_path: str | Path,
) -> tuple[list[OracleRecord], list[OracleRecord]]:
    """Validate the complete 9000/8100/900 topology and preserve split order."""
    oracle = load_oracle_records(oracle_path)
    train_ids = load_split_ids(train_ids_path, EXPECTED_TRAIN_COUNT)
    val_ids = load_split_ids(val_ids_path, EXPECTED_VAL_COUNT)
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        preview = ", ".join(sorted(overlap)[:10])
        raise DataValidationError(
            f"Train/validation split overlap contains {len(overlap)} IDs: {preview}"
        )
    split_ids = set(train_ids) | set(val_ids)
    missing_from_oracle = split_ids - set(oracle)
    if missing_from_oracle:
        preview = ", ".join(sorted(missing_from_oracle)[:10])
        raise DataValidationError(
            f"{len(missing_from_oracle)} split IDs are missing from Oracle: {preview}"
        )
    missing_from_splits = set(oracle) - split_ids
    if missing_from_splits:
        preview = ", ".join(sorted(missing_from_splits)[:10])
        raise DataValidationError(
            f"{len(missing_from_splits)} Oracle IDs are missing from splits: {preview}"
        )
    return ([oracle[qid] for qid in train_ids], [oracle[qid] for qid in val_ids])


class OracleQuestionDataset(Dataset):
    """Tokenize question text and expose the zero-based hard Oracle label."""

    def __init__(
        self,
        records: Sequence[OracleRecord],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        if not records:
            raise DataValidationError(
                "Cannot initialize a dataset with no Oracle records"
            )
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length <= 0
        ):
            raise DataValidationError(
                f"max_length must be a positive integer, got {max_length!r}"
            )
        ids = [record.question_id for record in records]
        if len(ids) != len(set(ids)):
            raise DataValidationError("Dataset records contain duplicate question IDs")
        for record in records:
            _require_k(record.hard_oracle_k, "hard_oracle_k", record.question_id)
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        encoded = self.tokenizer(
            record.question,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_token_type_ids=False,
        )
        return {
            "question_id": record.question_id,
            "dataset": record.dataset,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "label": torch.tensor(record.hard_oracle_k - 1, dtype=torch.long),
        }


class QuestionBatchCollator:
    """Use HuggingFace dynamic padding while preserving string metadata."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.padding_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        token_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = dict(self.padding_collator(token_features))
        batch["labels"] = torch.stack([feature["label"] for feature in features])
        batch["question_ids"] = [feature["question_id"] for feature in features]
        batch["datasets"] = [feature["dataset"] for feature in features]
        return batch


def build_train_val_datasets(
    oracle_path: str | Path,
    train_ids_path: str | Path,
    val_ids_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> tuple[OracleQuestionDataset, OracleQuestionDataset]:
    train_records, val_records = load_oracle_splits(
        oracle_path, train_ids_path, val_ids_path
    )
    train_dataset = OracleQuestionDataset(train_records, tokenizer, max_length)
    val_dataset = OracleQuestionDataset(val_records, tokenizer, max_length)
    if (
        len(train_dataset) != EXPECTED_TRAIN_COUNT
        or len(val_dataset) != EXPECTED_VAL_COUNT
    ):
        raise DataValidationError(
            f"Final dataset sizes must be {EXPECTED_TRAIN_COUNT}/{EXPECTED_VAL_COUNT}, "
            f"got {len(train_dataset)}/{len(val_dataset)}"
        )
    return train_dataset, val_dataset


def load_curve_records(
    path: str | Path,
    expected_count: int = EXPECTED_ORACLE_COUNT,
) -> dict[str, CurveRecord]:
    input_path = project_path(path)
    raw_rows = _read_jsonl_objects(input_path)
    if len(raw_rows) != expected_count:
        raise DataValidationError(
            f"{input_path}: expected {expected_count} curve rows, found {len(raw_rows)}"
        )
    expected_result_keys = {str(k) for k in range(1, NUM_LABELS + 1)}
    records: dict[str, CurveRecord] = {}
    for line_number, row in raw_rows:
        location = f"{input_path}:{line_number}"
        question_id = _require_nonempty_string(
            row.get("question_id"), "question_id", location
        )
        if question_id in records:
            raise DataValidationError(
                f"{location}: duplicate question_id {question_id!r}"
            )
        dataset = _require_dataset(row.get("dataset"), location)
        question = _require_nonempty_string(row.get("question"), "question", location)
        gold_answer = row.get("gold_answer")
        if (
            not isinstance(gold_answer, list)
            or not gold_answer
            or any(not isinstance(answer, str) or not answer for answer in gold_answer)
        ):
            raise DataValidationError(f"{location}: invalid gold_answer")
        raw_results = row.get("results")
        if (
            not isinstance(raw_results, dict)
            or set(raw_results) != expected_result_keys
        ):
            raise DataValidationError(
                f"{location}: results must contain exactly string keys 1..{NUM_LABELS}"
            )
        results: dict[int, CurveResult] = {}
        for k in range(1, NUM_LABELS + 1):
            result_location = f"{location} results[{k}]"
            result = raw_results[str(k)]
            if not isinstance(result, dict):
                raise DataValidationError(
                    f"{result_location}: result must be an object"
                )
            if not isinstance(result.get("prediction"), str):
                raise DataValidationError(
                    f"{result_location}: prediction must be a string"
                )
            _require_score(result.get("em"), "em", result_location)
            f1 = _require_score(result.get("f1"), "f1", result_location)
            tokens = result.get("context_tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                raise DataValidationError(
                    f"{result_location}: context_tokens must be a positive integer"
                )
            validated = result.get("context_tokens_validated")
            if not isinstance(validated, bool):
                raise DataValidationError(
                    f"{result_location}: context_tokens_validated must be boolean"
                )
            results[k] = CurveResult(
                f1=f1,
                context_tokens=tokens,
                context_tokens_validated=validated,
            )
        records[question_id] = CurveRecord(
            question_id=question_id,
            dataset=dataset,
            question=question,
            results=results,
        )
    return records


def load_prediction_records(path: str | Path) -> dict[str, PredictionRecord]:
    input_path = project_path(path)
    records: dict[str, PredictionRecord] = {}
    for line_number, row in _read_jsonl_objects(input_path):
        location = f"{input_path}:{line_number}"
        question_id = _require_nonempty_string(
            row.get("question_id"), "question_id", location
        )
        if question_id in records:
            raise DataValidationError(
                f"{location}: duplicate question_id {question_id!r}"
            )
        dataset = _require_dataset(row.get("dataset"), location)
        predicted_k = _require_k(row.get("predicted_k"), "predicted_k", location)
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != NUM_LABELS:
            raise DataValidationError(
                f"{location}: probabilities must be a {NUM_LABELS}-element array"
            )
        parsed_probabilities = tuple(
            _require_score(probability, f"probabilities[{index}]", location)
            for index, probability in enumerate(probabilities)
        )
        probability_sum = sum(parsed_probabilities)
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise DataValidationError(
                f"{location}: probabilities sum to {probability_sum:.12g}, not 1"
            )
        argmax_k = max(
            range(1, NUM_LABELS + 1),
            key=lambda k: (parsed_probabilities[k - 1], -k),
        )
        if predicted_k != argmax_k:
            raise DataValidationError(
                f"{location}: predicted_k={predicted_k} differs from probability argmax={argmax_k}"
            )
        hard_value = row.get("hard_oracle_k")
        hard_oracle_k = (
            None
            if hard_value is None
            else _require_k(hard_value, "hard_oracle_k", location)
        )
        records[question_id] = PredictionRecord(
            question_id=question_id,
            dataset=dataset,
            predicted_k=predicted_k,
            probabilities=parsed_probabilities,
            hard_oracle_k=hard_oracle_k,
        )
    if not records:
        raise DataValidationError(f"Prediction file contains no rows: {input_path}")
    return records
