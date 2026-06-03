"""Shared utilities for the new two-stage KBQA classifier trainers."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TRAIN_FILE = "kbqa_classifier_new/train_data/train/train.json"
DEFAULT_VALID_FILE = "kbqa_classifier_new/train_data/valid/valid.json"
JSONL_SUFFIXES = {".jsonl", ".jsonlines"}

QUESTION_FIELD_CANDIDATES = ("question", "question_text", "query", "text", "prompt")
LABEL_FIELD_CANDIDATES = ("label", "labels", "topk_label", "top_k_label", "target")
ID_FIELD_CANDIDATES = ("id", "qid", "question_id", "example_id")


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret {value!r} as a boolean.")


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / path).resolve()


def ensure_output_dir(path_value: str) -> Path:
    output_dir = resolve_path(path_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() in JSONL_SUFFIXES:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL line {line_number} in {path} is not an object.")
                records.append(record)
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("data", "records", "examples", "items", "train", "validation", "valid"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            records = next((value for value in payload.values() if isinstance(value, list)), None)
        if records is None:
            raise ValueError(f"Could not find a record list in {path}.")
    else:
        raise ValueError(f"Unsupported JSON root type in {path}: {type(payload).__name__}")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every record in {path} must be a JSON object.")
    return list(records)


def normalize_field_name(field_name: str) -> str:
    return "".join(char.lower() for char in field_name if char.isalnum())


def infer_field_name(
    records: Sequence[Dict[str, Any]],
    explicit_field: Optional[str],
    candidates: Iterable[str],
    field_type: str,
) -> str:
    if explicit_field:
        return explicit_field

    available_keys: List[str] = []
    seen = set()
    for record in records[:50]:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                available_keys.append(key)

    normalized_to_key = {normalize_field_name(key): key for key in available_keys}
    for candidate in candidates:
        if candidate in available_keys:
            return candidate
    for candidate in candidates:
        normalized = normalize_field_name(candidate)
        if normalized in normalized_to_key:
            return normalized_to_key[normalized]

    raise KeyError(
        f"Could not infer {field_type} field. Available keys: {available_keys}. "
        f"Set --{field_type}_field explicitly."
    )


def infer_optional_field_name(
    records: Sequence[Dict[str, Any]],
    explicit_field: Optional[str],
    candidates: Iterable[str],
) -> Optional[str]:
    if explicit_field:
        return explicit_field

    available_keys: List[str] = []
    seen = set()
    for record in records[:50]:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                available_keys.append(key)

    normalized_to_key = {normalize_field_name(key): key for key in available_keys}
    for candidate in candidates:
        if candidate in available_keys:
            return candidate
    for candidate in candidates:
        normalized = normalize_field_name(candidate)
        if normalized in normalized_to_key:
            return normalized_to_key[normalized]
    return None


def parse_topk_label(raw_label: Any) -> int:
    if isinstance(raw_label, bool):
        label = int(raw_label)
    elif isinstance(raw_label, int):
        label = raw_label
    elif isinstance(raw_label, float):
        if not raw_label.is_integer():
            raise ValueError(f"Expected integer-like label, got {raw_label!r}.")
        label = int(raw_label)
    elif isinstance(raw_label, str):
        stripped = raw_label.strip()
        if not stripped:
            raise ValueError("Label string is empty.")
        label = int(float(stripped)) if "." in stripped else int(stripped)
    else:
        raise TypeError(f"Unsupported label type: {type(raw_label).__name__}")

    if label < 0:
        raise ValueError(f"Top-k label must be non-negative, got {label}.")
    return label


def stage1_label_from_topk(topk_label: int) -> int:
    return 0 if topk_label == 0 else 1


def stage2_bucket_from_topk(topk_label: int) -> int:
    if 1 <= topk_label <= 2:
        return 0
    if 3 <= topk_label <= 5:
        return 1
    if 6 <= topk_label <= 9:
        return 2
    if 10 <= topk_label <= 15:
        return 3
    raise ValueError(f"Stage 2 only supports positive labels in [1, 15], got {topk_label}.")


def clean_text_value(record: Dict[str, Any], field_name: str, index: int) -> str:
    if field_name not in record:
        raise KeyError(f"Missing field {field_name!r} at record index {index}.")

    value = record[field_name]
    if value is None or isinstance(value, (list, dict)):
        raise ValueError(f"Field {field_name!r} at record index {index} must be a scalar text value.")

    text = str(value).strip()
    if not text:
        raise ValueError(f"Field {field_name!r} at record index {index} is empty.")
    return text


def load_stage_examples(
    data_file: Path,
    stage: str,
    question_field: Optional[str] = None,
    label_field: Optional[str] = None,
    id_field: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = load_records(data_file)
    if not records:
        raise ValueError(f"No records found in {data_file}.")

    resolved_question_field = infer_field_name(
        records,
        explicit_field=question_field,
        candidates=QUESTION_FIELD_CANDIDATES,
        field_type="question",
    )
    resolved_label_field = infer_field_name(
        records,
        explicit_field=label_field,
        candidates=LABEL_FIELD_CANDIDATES,
        field_type="label",
    )
    resolved_id_field = infer_optional_field_name(records, id_field, ID_FIELD_CANDIDATES)

    examples: List[Dict[str, Any]] = []
    raw_distribution: Counter = Counter()
    task_distribution: Counter = Counter()
    skipped_count = 0

    for index, record in enumerate(records):
        question = clean_text_value(record, resolved_question_field, index)
        if resolved_label_field not in record:
            raise KeyError(f"Missing label field {resolved_label_field!r} at record index {index}.")

        topk_label = parse_topk_label(record[resolved_label_field])
        raw_distribution[topk_label] += 1

        if stage == "stage1":
            task_label = stage1_label_from_topk(topk_label)
        elif stage == "stage2":
            if topk_label == 0:
                skipped_count += 1
                continue
            task_label = stage2_bucket_from_topk(topk_label)
        else:
            raise ValueError(f"Unknown stage: {stage}")

        sample_id = record.get(resolved_id_field) if resolved_id_field else None
        examples.append(
            {
                "id": str(sample_id).strip() if sample_id not in (None, "") else f"sample_{index}",
                "question": question,
                "label": int(task_label),
                "raw_topk_label": int(topk_label),
            }
        )
        task_distribution[task_label] += 1

    if not examples:
        raise ValueError(f"No usable {stage} examples found in {data_file}.")

    metadata = {
        "data_file": str(data_file),
        "stage": stage,
        "total_records": len(records),
        "usable_records": len(examples),
        "skipped_records": skipped_count,
        "question_field": resolved_question_field,
        "label_field": resolved_label_field,
        "id_field": resolved_id_field,
        "raw_label_distribution": {str(key): value for key, value in sorted(raw_distribution.items())},
        "task_label_distribution": {str(key): value for key, value in sorted(task_distribution.items())},
    }
    return examples, metadata


class QuestionClassificationDataset(Dataset):
    """Tokenized question-only dataset for sequence classification."""

    def __init__(self, examples: Sequence[Dict[str, Any]], tokenizer, max_length: int):
        self.examples = list(examples)
        self.encodings = tokenizer(
            [example["question"] for example in self.examples],
            truncation=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(int(self.examples[index]["label"]), dtype=torch.long)
        return item


def compute_class_weights(labels: Sequence[int], num_labels: int) -> torch.Tensor:
    distribution = Counter(int(label) for label in labels)
    present_class_count = sum(1 for label_id in range(num_labels) if distribution.get(label_id, 0) > 0)
    if present_class_count == 0:
        raise ValueError("Cannot compute class weights for an empty label set.")

    total_count = float(sum(distribution.values()))
    weights = []
    for label_id in range(num_labels):
        count = distribution.get(label_id, 0)
        weights.append(total_count / (present_class_count * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float)


class WeightedLossTrainer:  # pragma: no cover - runtime adapter around transformers.Trainer
    """Create a Trainer subclass that uses optional weighted cross-entropy."""

    def __new__(cls, *args, **kwargs):
        from transformers import Trainer

        class _WeightedLossTrainer(Trainer):
            def __init__(self, *trainer_args, class_weights: Optional[torch.Tensor] = None, **trainer_kwargs):
                super().__init__(*trainer_args, **trainer_kwargs)
                self.class_weights = class_weights

            def compute_loss(self, model, inputs, return_outputs=False, **unused_kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fn = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
                )
                loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        return _WeightedLossTrainer(*args, **kwargs)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_classification_metrics(
    gold_labels: Sequence[int],
    pred_labels: Sequence[int],
    label_ids: Sequence[int],
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    if len(gold_labels) != len(pred_labels):
        raise ValueError("gold_labels and pred_labels must have the same length.")
    if not gold_labels:
        raise ValueError("Cannot compute metrics on an empty evaluation set.")

    label_to_index = {label_id: index for index, label_id in enumerate(label_ids)}
    confusion = np.zeros((len(label_ids), len(label_ids)), dtype=np.int64)
    for gold_label, pred_label in zip(gold_labels, pred_labels):
        confusion[label_to_index[int(gold_label)], label_to_index[int(pred_label)]] += 1

    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    per_class: List[Dict[str, Any]] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []

    for index, label_id in enumerate(label_ids):
        true_positive = float(confusion[index, index])
        precision = safe_divide(true_positive, float(predicted_support[index]))
        recall = safe_divide(true_positive, float(support[index]))
        f1 = safe_divide(2.0 * precision * recall, precision + recall) if precision + recall else 0.0
        label_name = id2label.get(label_id, str(label_id))
        per_class.append(
            {
                "label_id": int(label_id),
                "label_name": label_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support[index]),
                "predicted_count": int(predicted_support[index]),
            }
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    support_float = support.astype(np.float64)
    total_support = float(support_float.sum())
    total_samples = int(confusion.sum())

    return {
        "sample_count": total_samples,
        "accuracy": safe_divide(float(np.trace(confusion)), float(total_samples)),
        "precision_macro": float(np.mean(precision_values)),
        "recall_macro": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "precision_weighted": safe_divide(float(np.dot(precision_values, support_float)), total_support),
        "recall_weighted": safe_divide(float(np.dot(recall_values, support_float)), total_support),
        "weighted_f1": safe_divide(float(np.dot(f1_values, support_float)), total_support),
        "per_class": per_class,
        "confusion_matrix": {
            "label_ids": [int(label_id) for label_id in label_ids],
            "label_names": [id2label.get(label_id, str(label_id)) for label_id in label_ids],
            "matrix": confusion.tolist(),
        },
    }


def make_trainer_metrics_fn(label_ids: Sequence[int], id2label: Dict[int, str]):
    def compute_metrics(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1).astype(int).tolist()
        gold_labels = eval_prediction.label_ids.astype(int).tolist()
        metrics = compute_classification_metrics(gold_labels, predictions, label_ids, id2label)
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
        }

    return compute_metrics


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def build_prediction_rows(
    examples: Sequence[Dict[str, Any]],
    logits: np.ndarray,
    id2label: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    probabilities = softmax(logits)
    pred_labels = np.argmax(logits, axis=-1).astype(int).tolist()
    gold_labels = [int(example["label"]) for example in examples]

    rows: List[Dict[str, Any]] = []
    for example, logit_row, probability_row, pred_label in zip(examples, logits, probabilities, pred_labels):
        gold_label = int(example["label"])
        pred_label = int(pred_label)
        rows.append(
            {
                "id": example["id"],
                "question": example["question"],
                "raw_topk_label": int(example["raw_topk_label"]),
                "gold_label": gold_label,
                "gold_label_name": id2label.get(gold_label, str(gold_label)),
                "pred_label": pred_label,
                "pred_label_name": id2label.get(pred_label, str(pred_label)),
                "logits": [float(value) for value in logit_row.tolist()],
                "probabilities": [float(value) for value in probability_row.tolist()],
            }
        )

    return rows, gold_labels, pred_labels


def sanitize_name(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return candidate.strip("._-") or "model"

