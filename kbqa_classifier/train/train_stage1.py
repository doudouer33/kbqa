#!/usr/bin/env python
"""Train and evaluate the stage-1 binary classifier for retrieval necessity."""

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


DEFAULT_TRAIN_FILE = "kbqa_classifier/data/merged/dev_500_topk_train.json"
DEFAULT_EVAL_OUTPUT_DIR = "kbqa_classifier/output/classifier_eval"
QUESTION_FIELD_CANDIDATES: Tuple[str, ...] = (
    "question",
    "question_text",
    "questiontext",
    "query",
    "query_text",
    "text",
    "input",
    "prompt",
)
LABEL_FIELD_CANDIDATES: Tuple[str, ...] = (
    "label",
    "labels",
    "topk_label",
    "top_k_label",
    "topklabel",
    "topk",
    "class_label",
    "target",
)
DATASET_NAME_FIELD_CANDIDATES: Tuple[str, ...] = (
    "dataset_name",
    "dataset",
    "source_dataset",
    "source",
)
ID_FIELD_CANDIDATES: Tuple[str, ...] = (
    "id",
    "example_id",
    "question_id",
    "qid",
)
JSONL_SUFFIXES = {".jsonl", ".jsonlines"}


class Stage1ClassificationDataset(Dataset):
    """Tokenized dataset for binary sequence classification."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class WeightedLossTrainer:  # pragma: no cover - thin runtime wrapper around transformers.Trainer
    """Wrap transformers.Trainer with class-weighted cross-entropy."""

    def __new__(cls, *args, **kwargs):
        from transformers import Trainer

        class _WeightedLossTrainer(Trainer):
            def __init__(self, *trainer_args, class_weights: torch.Tensor = None, **trainer_kwargs):
                super().__init__(*trainer_args, **trainer_kwargs)
                self.class_weights = class_weights

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs["labels"]
                model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                outputs = model(**model_inputs)
                logits = outputs.logits

                if self.class_weights is None:
                    loss_fct = torch.nn.CrossEntropyLoss()
                else:
                    loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))

                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        return _WeightedLossTrainer(*args, **kwargs)


def str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the stage-1 retrieval classifier.")
    parser.add_argument(
        "--train_file",
        type=str,
        default=DEFAULT_TRAIN_FILE,
        help="Path to the JSON or JSONL training file.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to the JSON or JSONL evaluation file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="BERT checkpoint name, saved model directory, or checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/stage1",
        help="Directory used to save the fine-tuned HuggingFace model.",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default=DEFAULT_EVAL_OUTPUT_DIR,
        help="Root directory used to save classifier evaluation artifacts.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional dataset tag used in evaluation artifact names.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help="Optional split tag used in evaluation artifact names.",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Optional model tag used in evaluation artifact names.",
    )
    parser.add_argument("--max_length", type=int, default=128, help="Maximum tokenized question length.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Per-device batch size used for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size used for classifier evaluation.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging interval in steps.")
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Checkpoint save strategy passed to TrainingArguments.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Used when --save_strategy=steps.",
    )
    parser.add_argument(
        "--do_train",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to run training. Supports --do_train, --do_train True, or --do_train False.",
    )
    parser.add_argument(
        "--do_eval",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to run classifier-level evaluation on --eval_file.",
    )
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_input_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def resolve_output_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def normalize_field_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def sanitize_artifact_component(value: Optional[str], fallback: str) -> str:
    if value is None:
        return fallback

    candidate = str(value).strip()
    if not candidate:
        return fallback

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    sanitized = sanitized.strip("._-")
    return sanitized or fallback


def can_cast_to_int(value: Any) -> bool:
    try:
        parse_raw_label(value)
    except (TypeError, ValueError):
        return False
    return True


def parse_raw_label(value: Any) -> int:
    if isinstance(value, bool):
        raw_label = int(value)
    elif isinstance(value, int):
        raw_label = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Expected an integer-like label, but got {value!r}.")
        raw_label = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Label string is empty.")
        if "." in stripped:
            float_value = float(stripped)
            if not float_value.is_integer():
                raise ValueError(f"Expected an integer-like label string, but got {value!r}.")
            raw_label = int(float_value)
        else:
            raw_label = int(stripped)
    else:
        raise TypeError(f"Unsupported label type: {type(value).__name__}")

    if raw_label < 0:
        raise ValueError(f"Original top-k label must be non-negative, but got {raw_label}.")
    return raw_label


def stage1_label_from_raw(raw_label: Any) -> int:
    # Stage-1 label mapping:
    # original top-k label == 0 -> 0 (no retrieval)
    # original top-k label > 0  -> 1 (needs retrieval)
    raw_label = parse_raw_label(raw_label)
    return 0 if raw_label == 0 else 1


def extract_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        preferred_container_keys = ("data", "records", "examples", "items", "train", "dataset")
        for key in preferred_container_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value

        for _, value in payload.items():
            if isinstance(value, list):
                return value

    raise ValueError(
        "Unsupported JSON structure. Expected a list of records or a dict containing a list of records."
    )


def load_records_from_file(data_file: Path) -> List[Dict[str, Any]]:
    if not data_file.exists():
        raise FileNotFoundError(f"Data file does not exist: {data_file}")

    if data_file.suffix.lower() in JSONL_SUFFIXES:
        records: List[Dict[str, Any]] = []
        with data_file.open("r", encoding="utf-8") as handle:
            for line_index, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Every JSONL line must be a JSON object, but line {line_index} in {data_file} is not."
                    )
                records.append(record)
        return records

    with data_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = extract_records(payload)
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every record must be a JSON object.")
    return records


def get_available_keys(records: Sequence[Dict[str, Any]], limit: int = 50) -> List[str]:
    ordered_keys: List[str] = []
    seen = set()
    for record in records[:limit]:
        if not isinstance(record, dict):
            continue
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)
    return ordered_keys


def has_text_value(records: Sequence[Dict[str, Any]], key: str) -> bool:
    for record in records:
        value = record.get(key)
        if value is None or isinstance(value, (list, dict)):
            continue
        if str(value).strip():
            return True
    return False


def has_numeric_value(records: Sequence[Dict[str, Any]], key: str) -> bool:
    for record in records:
        if key in record and can_cast_to_int(record[key]):
            return True
    return False


def infer_field_name(records: Sequence[Dict[str, Any]], candidates: Iterable[str], field_type: str) -> str:
    available_keys = get_available_keys(records)
    normalized_to_key = {normalize_field_name(key): key for key in available_keys}

    for candidate in candidates:
        if candidate in available_keys:
            return candidate

    for candidate in candidates:
        normalized_candidate = normalize_field_name(candidate)
        if normalized_candidate in normalized_to_key:
            return normalized_to_key[normalized_candidate]

    if field_type == "question":
        semantic_tokens = ("question", "query", "prompt", "text")
        for key in available_keys:
            normalized_key = normalize_field_name(key)
            if any(token in normalized_key for token in semantic_tokens) and has_text_value(records, key):
                return key
    elif field_type == "label":
        semantic_tokens = ("label", "topk", "class", "target")
        for key in available_keys:
            normalized_key = normalize_field_name(key)
            if any(token in normalized_key for token in semantic_tokens) and has_numeric_value(records, key):
                return key

    raise KeyError(
        f"Could not infer the {field_type} field. Available keys: {available_keys}. "
        f"Tried preferred candidates: {list(candidates)}."
    )


def infer_optional_field_name(
    records: Sequence[Dict[str, Any]],
    candidates: Iterable[str],
    field_type: str,
) -> Optional[str]:
    available_keys = get_available_keys(records)
    normalized_to_key = {normalize_field_name(key): key for key in available_keys}

    for candidate in candidates:
        if candidate in available_keys:
            return candidate

    for candidate in candidates:
        normalized_candidate = normalize_field_name(candidate)
        if normalized_candidate in normalized_to_key:
            return normalized_to_key[normalized_candidate]

    semantic_tokens_by_type = {
        "dataset_name": ("dataset", "source"),
        "id": ("id", "qid"),
    }
    for key in available_keys:
        normalized_key = normalize_field_name(key)
        if any(token in normalized_key for token in semantic_tokens_by_type.get(field_type, ())):
            return key

    return None


def extract_question_text(record: Dict[str, Any], question_field: str, record_index: int) -> str:
    if question_field not in record:
        raise KeyError(f"Missing question field {question_field!r} at record index {record_index}.")

    raw_question = record[question_field]
    if raw_question is None or isinstance(raw_question, (list, dict)):
        raise ValueError(
            f"Question field {question_field!r} must be a non-empty string-like value at record index {record_index}."
        )

    question = str(raw_question).strip()
    if not question:
        raise ValueError(f"Question text is empty at record index {record_index}.")
    return question


def extract_optional_scalar(record: Dict[str, Any], field_name: Optional[str]) -> Optional[Any]:
    if not field_name or field_name not in record:
        return None

    value = record[field_name]
    if value is None or isinstance(value, (list, dict)):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    return value


def make_default_sample_id(record_index: int) -> str:
    return f"sample_{record_index}"


def load_labeled_records(
    data_file: Path,
) -> Tuple[List[Dict[str, Any]], str, str, Optional[str], Optional[str]]:
    records = load_records_from_file(data_file)
    if not records:
        raise ValueError(f"No records found in {data_file}.")

    question_field = infer_field_name(records, QUESTION_FIELD_CANDIDATES, field_type="question")
    label_field = infer_field_name(records, LABEL_FIELD_CANDIDATES, field_type="label")
    id_field = infer_optional_field_name(records, ID_FIELD_CANDIDATES, field_type="id")
    dataset_field = infer_optional_field_name(records, DATASET_NAME_FIELD_CANDIDATES, field_type="dataset_name")
    return records, question_field, label_field, id_field, dataset_field


def load_training_examples(train_file: Path) -> Tuple[List[str], List[int], Counter, str, str]:
    records, question_field, label_field, _, _ = load_labeled_records(train_file)

    texts: List[str] = []
    stage1_labels: List[int] = []
    original_label_distribution: Counter = Counter()

    for index, record in enumerate(records):
        question = extract_question_text(record, question_field, index)
        if label_field not in record:
            raise KeyError(f"Missing label field {label_field!r} at record index {index}.")

        raw_label = parse_raw_label(record[label_field])
        stage1_label = stage1_label_from_raw(raw_label)

        texts.append(question)
        stage1_labels.append(stage1_label)
        original_label_distribution[raw_label] += 1

    if not texts:
        raise ValueError(f"No usable training examples found in {train_file}.")

    return texts, stage1_labels, original_label_distribution, question_field, label_field


def load_stage1_eval_examples(
    eval_file: Path,
) -> Tuple[List[Dict[str, Any]], Counter, Counter, str, str, Optional[str], Optional[str]]:
    records, question_field, label_field, id_field, dataset_field = load_labeled_records(eval_file)

    examples: List[Dict[str, Any]] = []
    raw_label_distribution: Counter = Counter()
    binary_label_distribution: Counter = Counter()

    for index, record in enumerate(records):
        question = extract_question_text(record, question_field, index)
        if label_field not in record:
            raise KeyError(f"Missing label field {label_field!r} at record index {index}.")

        raw_topk_label = parse_raw_label(record[label_field])
        binary_label = stage1_label_from_raw(raw_topk_label)

        sample_id = extract_optional_scalar(record, id_field)
        dataset_name = extract_optional_scalar(record, dataset_field)

        examples.append(
            {
                "id": sample_id if sample_id is not None else make_default_sample_id(index),
                "question": question,
                "dataset_name": dataset_name,
                "raw_topk_label": raw_topk_label,
                "gold_label": binary_label,
            }
        )
        raw_label_distribution[raw_topk_label] += 1
        binary_label_distribution[binary_label] += 1

    if not examples:
        raise ValueError(f"No usable evaluation examples found in {eval_file}.")

    return (
        examples,
        raw_label_distribution,
        binary_label_distribution,
        question_field,
        label_field,
        id_field,
        dataset_field,
    )


def compute_class_weights(labels: Sequence[int]) -> torch.Tensor:
    label_distribution = Counter(labels)
    class_counts = [label_distribution.get(0, 0), label_distribution.get(1, 0)]
    if 0 in class_counts:
        raise ValueError(
            f"Both stage-1 classes must be present for weighted training, but got distribution {label_distribution}."
        )

    total_count = float(sum(class_counts))
    # Inverse-frequency class weights reduce the dominance of label 0 in this dataset.
    weights = [total_count / (len(class_counts) * class_count) for class_count in class_counts]
    return torch.tensor(weights, dtype=torch.float)


def save_json(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_id2label_mapping(raw_mapping: Any, fallback: Dict[int, str]) -> Dict[int, str]:
    if not isinstance(raw_mapping, dict):
        return dict(fallback)

    normalized: Dict[int, str] = {}
    for raw_key, raw_value in raw_mapping.items():
        try:
            label_id = int(raw_key)
        except (TypeError, ValueError):
            continue

        label_name = str(raw_value)
        normalized[label_id] = label_name

    return normalized or dict(fallback)


def normalize_label2id_mapping(raw_mapping: Any, fallback: Dict[str, int]) -> Dict[str, int]:
    if not isinstance(raw_mapping, dict):
        return dict(fallback)

    normalized: Dict[str, int] = {}
    for raw_key, raw_value in raw_mapping.items():
        try:
            normalized[str(raw_key)] = int(raw_value)
        except (TypeError, ValueError):
            continue

    return normalized or dict(fallback)


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_dataset_name(explicit_dataset_name: Optional[str], examples: Sequence[Dict[str, Any]], eval_file: Path) -> str:
    if explicit_dataset_name:
        return str(explicit_dataset_name)

    for example in examples:
        dataset_name = example.get("dataset_name")
        if dataset_name is None:
            continue
        dataset_name_str = str(dataset_name).strip()
        if dataset_name_str:
            return dataset_name_str

    return eval_file.parent.name


def infer_split_name(explicit_split_name: Optional[str], eval_file: Path) -> str:
    if explicit_split_name:
        return str(explicit_split_name)

    filename = eval_file.name
    for suffix in (".jsonl", ".jsonlines", ".json"):
        if filename.endswith(suffix):
            filename = filename[: -len(suffix)]
            break

    if "_topk" in filename:
        candidate = filename.split("_topk", 1)[0]
        if candidate:
            return candidate

    if "_" in filename:
        return filename.split("_", 1)[0]

    return filename


def infer_model_tag(explicit_model_tag: Optional[str], model_name_or_path: str) -> str:
    if explicit_model_tag:
        return str(explicit_model_tag)

    stripped = str(model_name_or_path).rstrip("/").rstrip("\\")
    basename = Path(stripped).name if stripped else ""
    return basename or "model"


def build_eval_artifact_paths(
    stage_name: str,
    eval_output_dir: str,
    dataset_name: str,
    split_name: str,
    model_tag: str,
) -> Tuple[Path, Path, Path]:
    root_dir = resolve_output_path(eval_output_dir)
    stage_dir = root_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    safe_dataset_name = sanitize_artifact_component(dataset_name, fallback="dataset")
    safe_split_name = sanitize_artifact_component(split_name, fallback="split")
    safe_model_tag = sanitize_artifact_component(model_tag, fallback="model")

    file_stem = f"{stage_name}_classifier__{safe_dataset_name}__{safe_split_name}__{safe_model_tag}"
    metrics_path = stage_dir / f"metrics__{file_stem}.json"
    predictions_path = stage_dir / f"predictions__{file_stem}.jsonl"
    return stage_dir, metrics_path, predictions_path


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def predict_logits(
    texts: Sequence[str],
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, but got {batch_size}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    logits_batches: List[np.ndarray] = []
    with torch.inference_mode():
        for start_index in range(0, len(texts), batch_size):
            batch_texts = list(texts[start_index : start_index + batch_size])
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            batch_logits = model(**encoded).logits.detach().cpu().numpy()
            logits_batches.append(batch_logits)

    logits = np.concatenate(logits_batches, axis=0) if logits_batches else np.zeros((0, 0), dtype=np.float32)
    probabilities = softmax(logits) if logits.size else np.zeros_like(logits)
    predictions = np.argmax(logits, axis=-1).astype(int).tolist() if logits.size else []
    return logits, probabilities, predictions


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_classification_metrics(
    gold_labels: Sequence[int],
    pred_labels: Sequence[int],
    label_ids: Sequence[int],
    id2label: Dict[int, str],
    positive_label_id: Optional[int] = None,
) -> Dict[str, Any]:
    if len(gold_labels) != len(pred_labels):
        raise ValueError("gold_labels and pred_labels must have the same length.")
    if not gold_labels:
        raise ValueError("Cannot compute metrics on an empty evaluation set.")

    label_to_index = {label_id: index for index, label_id in enumerate(label_ids)}
    confusion = np.zeros((len(label_ids), len(label_ids)), dtype=np.int64)

    for gold_label, pred_label in zip(gold_labels, pred_labels):
        if gold_label not in label_to_index:
            raise ValueError(f"Encountered gold label {gold_label} outside the declared label space {list(label_ids)}.")
        if pred_label not in label_to_index:
            raise ValueError(f"Encountered predicted label {pred_label} outside the declared label space {list(label_ids)}.")
        confusion[label_to_index[gold_label], label_to_index[pred_label]] += 1

    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)

    per_class: List[Dict[str, Any]] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []

    for index, label_id in enumerate(label_ids):
        tp = float(confusion[index, index])
        precision = safe_divide(tp, float(predicted_support[index]))
        recall = safe_divide(tp, float(support[index]))
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

    total_samples = int(confusion.sum())
    support_array = support.astype(np.float64)
    total_support = float(support_array.sum())

    accuracy = safe_divide(float(np.trace(confusion)), float(total_samples))
    macro_precision = float(np.mean(precision_values))
    macro_recall = float(np.mean(recall_values))
    macro_f1 = float(np.mean(f1_values))
    weighted_precision = safe_divide(float(np.dot(precision_values, support_array)), total_support)
    weighted_recall = safe_divide(float(np.dot(recall_values, support_array)), total_support)
    weighted_f1 = safe_divide(float(np.dot(f1_values, support_array)), total_support)

    metrics: Dict[str, Any] = {
        "sample_count": total_samples,
        "evaluated_sample_count": total_samples,
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "macro_f1": macro_f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "support": {entry["label_name"]: entry["support"] for entry in per_class},
        "confusion_matrix": {
            "label_ids": [int(label_id) for label_id in label_ids],
            "label_names": [id2label.get(label_id, str(label_id)) for label_id in label_ids],
            "matrix": confusion.tolist(),
        },
    }

    if positive_label_id is not None:
        positive_metrics = next((entry for entry in per_class if entry["label_id"] == positive_label_id), None)
        if positive_metrics is None:
            raise ValueError(f"Positive label id {positive_label_id} is not part of the label space {list(label_ids)}.")
        metrics.update(
            {
                "precision": positive_metrics["precision"],
                "recall": positive_metrics["recall"],
                "f1": positive_metrics["f1"],
                "precision_binary": positive_metrics["precision"],
                "recall_binary": positive_metrics["recall"],
                "f1_binary": positive_metrics["f1"],
            }
        )

    return metrics


def load_config_label_mapping(
    model_name_or_path: str,
    fallback_id2label: Dict[int, str],
    fallback_label2id: Dict[str, int],
) -> Tuple[Dict[int, str], Dict[str, int]]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path)
    id2label = normalize_id2label_mapping(getattr(config, "id2label", None), fallback_id2label)
    label2id = normalize_label2id_mapping(getattr(config, "label2id", None), fallback_label2id)
    return id2label, label2id


def train_stage1_classifier(args: argparse.Namespace) -> Path:
    train_file = resolve_input_path(args.train_file)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, stage1_labels, original_label_distribution, question_field, label_field = load_training_examples(train_file)
    stage1_label_distribution = Counter(stage1_labels)
    class_weights = compute_class_weights(stage1_labels)

    from transformers import DataCollatorWithPadding, TrainingArguments

    from kbqa_classifier.model.stage1_bert_classifier import (
        STAGE1_ID2LABEL,
        STAGE1_LABEL2ID,
        build_stage1_model,
        load_stage1_tokenizer,
    )

    tokenizer = load_stage1_tokenizer(args.model_name_or_path)
    model = build_stage1_model(args.model_name_or_path)
    train_dataset = Stage1ClassificationDataset(
        texts=texts,
        labels=stage1_labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=False,
        evaluation_strategy="no",
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        remove_unused_columns=True,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        class_weights=class_weights,
    )

    print(f"Total training samples: {len(texts)}")
    print(f"Stage-1 label distribution: {dict(sorted(stage1_label_distribution.items()))}")
    print(f"Training output directory: {output_dir}")

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()

    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(texts)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    save_json(
        output_dir / "label_mapping.json",
        {
            "id2label": {str(key): value for key, value in STAGE1_ID2LABEL.items()},
            "label2id": dict(STAGE1_LABEL2ID),
            "stage1_label_definition": {
                "0": "original top-k label == 0 -> no retrieval",
                "1": "original top-k label > 0 -> retrieval required",
            },
        },
    )
    save_json(
        output_dir / "stage1_training_config.json",
        {
            "train_file": str(train_file),
            "eval_file": args.eval_file,
            "model_name_or_path": args.model_name_or_path,
            "question_field": question_field,
            "raw_label_field": label_field,
            "max_length": args.max_length,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "logging_steps": args.logging_steps,
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "do_train": args.do_train,
            "do_eval": args.do_eval,
            "dataset_name": args.dataset_name,
            "split_name": args.split_name,
            "model_tag": args.model_tag,
            "total_samples": len(texts),
            "stage1_label_distribution": {str(key): value for key, value in sorted(stage1_label_distribution.items())},
            "original_label_distribution": {
                str(key): value for key, value in sorted(original_label_distribution.items())
            },
            "class_weights": class_weights.tolist(),
        },
    )
    save_json(output_dir / "training_args.json", training_args.to_dict())

    print("Stage-1 training finished.")
    print(f"Saved stage-1 model to: {output_dir}")
    return output_dir


def evaluate_stage1_classifier(args: argparse.Namespace, model_source: str) -> Tuple[Path, Path, Dict[str, Any]]:
    eval_file = resolve_input_path(args.eval_file)
    (
        examples,
        raw_label_distribution,
        binary_label_distribution,
        question_field,
        label_field,
        id_field,
        dataset_field,
    ) = load_stage1_eval_examples(eval_file)

    from kbqa_classifier.model.stage1_bert_classifier import (
        STAGE1_ID2LABEL,
        STAGE1_LABEL2ID,
        build_stage1_model,
        load_stage1_tokenizer,
    )

    id2label, label2id = load_config_label_mapping(model_source, STAGE1_ID2LABEL, STAGE1_LABEL2ID)
    tokenizer = load_stage1_tokenizer(model_source)
    model = build_stage1_model(model_source)

    logits, probabilities, pred_labels = predict_logits(
        texts=[example["question"] for example in examples],
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )

    gold_labels = [int(example["gold_label"]) for example in examples]
    metrics = compute_classification_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        label_ids=sorted(id2label.keys()),
        id2label=id2label,
        positive_label_id=label2id.get("needs_retrieval", 1),
    )

    dataset_name = infer_dataset_name(args.dataset_name, examples, eval_file)
    split_name = infer_split_name(args.split_name, eval_file)
    model_tag = infer_model_tag(args.model_tag, model_source)
    _, metrics_path, predictions_path = build_eval_artifact_paths(
        stage_name="stage1",
        eval_output_dir=args.eval_output_dir,
        dataset_name=dataset_name,
        split_name=split_name,
        model_tag=model_tag,
    )

    prediction_rows: List[Dict[str, Any]] = []
    for example, logit_row, probability_row, pred_label in zip(examples, logits, probabilities, pred_labels):
        gold_label = int(example["gold_label"])
        resolved_dataset_name = example["dataset_name"] if example["dataset_name"] is not None else dataset_name
        prediction_rows.append(
            {
                "id": example["id"],
                "question": example.get("question"),
                "dataset_name": resolved_dataset_name,
                "raw_topk_label": int(example["raw_topk_label"]),
                "binary_label": gold_label,
                "gold_label": gold_label,
                "gold_label_name": id2label.get(gold_label, str(gold_label)),
                "pred_label": int(pred_label),
                "pred_label_name": id2label.get(int(pred_label), str(pred_label)),
                "logits": [float(value) for value in logit_row.tolist()],
                "probabilities": [float(value) for value in probability_row.tolist()],
            }
        )

    metrics.update(
        {
            "stage": "stage1",
            "task": "classifier_eval",
            "dataset_name": dataset_name,
            "split_name": split_name,
            "model_tag": model_tag,
            "eval_file": str(eval_file),
            "model_name_or_path": str(model_source),
            "question_field": question_field,
            "raw_label_field": label_field,
            "id_field": id_field,
            "dataset_name_field": dataset_field,
            "raw_label_distribution": {str(key): value for key, value in sorted(raw_label_distribution.items())},
            "binary_label_distribution": {
                str(key): value for key, value in sorted(binary_label_distribution.items())
            },
            "label_mapping": {
                "id2label": {str(key): value for key, value in sorted(id2label.items())},
                "label2id": label2id,
            },
            "prediction_file": str(predictions_path),
        }
    )

    save_json(metrics_path, metrics)
    write_jsonl(predictions_path, prediction_rows)

    print("Stage-1 classifier evaluation finished.")
    print(f"Samples: {metrics['sample_count']}")
    print(
        "Binary precision/recall/f1 (positive class = needs_retrieval): "
        f"{metrics['precision_binary']:.4f} / {metrics['recall_binary']:.4f} / {metrics['f1_binary']:.4f}"
    )
    print(
        f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | "
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )
    print(f"Confusion matrix (gold rows, pred cols): {metrics['confusion_matrix']['matrix']}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")

    return metrics_path, predictions_path, metrics


def main() -> None:
    args = parse_args()

    if not args.do_train and not args.do_eval:
        raise ValueError("Nothing to do. At least one of --do_train or --do_eval must be True.")
    if args.do_eval and not args.eval_file:
        raise ValueError("--eval_file must be provided when --do_eval is True.")

    set_reproducible_seed(args.seed)

    eval_model_source = args.model_name_or_path

    if args.do_train:
        eval_model_source = str(train_stage1_classifier(args))
    else:
        print("Skipping stage-1 training because --do_train is False.")

    if args.do_eval:
        evaluate_stage1_classifier(args, eval_model_source)
    else:
        print("Skipping stage-1 evaluation because --do_eval is False.")


if __name__ == "__main__":
    main()
