#!/usr/bin/env python
"""Offline end-to-end replay evaluation for kbqa_classifier_mutihop.

The classifier predicts one retrieval count label in [1, 8] for each question.
This script looks up the precomputed QA result for that label and aggregates
EM/F1 without rerunning retrieval or generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kbqa_classifier_mutihop.model.classifier import (  # noqa: E402
    CLASS_ID2LABEL,
    load_classifier_model,
    load_classifier_tokenizer,
)


DEFAULT_MODEL_DIR = "kbqa_classifier_mutihop/outputs/classifier/model/bert-base-uncased/epoch/40"
DEFAULT_TEST_FILE = "kbqa_classifier_mutihop/data/prediction.json"
DEFAULT_REPLAY_LABEL_FILES = (
    "kbqa_classifier_mutihop/data/test/2wikimultihopqa/test_topk_labels.json",
    "kbqa_classifier_mutihop/data/test/hotpotqa/test_topk_labels.json",
    "kbqa_classifier_mutihop/data/test/musique/test_topk_labels.json",
)
DEFAULT_PREDICTIONS_ROOT = "predictions/test"
DEFAULT_OUTPUT_DIR = "kbqa_classifier_mutihop/eval/results"
DEFAULT_FIXED_KS = tuple(range(1, 9))
DATASET_NAMES = ("2wikimultihopqa", "hotpotqa", "musique")
JSONL_SUFFIXES = {".jsonl", ".jsonlines"}

QUESTION_FIELD_CANDIDATES = ("question", "question_text", "query", "text", "prompt")
LABEL_FIELD_CANDIDATES = ("label", "labels", "topk_label", "top_k_label", "target", "best_k")
ID_FIELD_CANDIDATES = ("id", "qid", "question_id", "example_id")
DATASET_FIELD_CANDIDATES = ("dataset_name", "dataset", "source_dataset", "source")
SCORES_FIELD_CANDIDATES = (
    "scores",
    "score_by_k",
    "score_map",
    "k_scores",
    "topk_scores",
    "per_k_scores",
    "results_by_k",
)
BEST_K_FIELD_CANDIDATES = ("best_k", "bestk", "oracle_best_k", "oracle_k")
PREDICTIONS_FIELD_CANDIDATES = ("predictions", "prediction_by_k", "answers_by_k")
EM_FIELD_CANDIDATES = ("em", "exact_match", "exact")
F1_FIELD_CANDIDATES = ("f1", "f1_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate kbqa_classifier_mutihop by offline replay lookup over labels 1-8."
    )
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test_file", default=DEFAULT_TEST_FILE)
    parser.add_argument(
        "--replay_label_files",
        nargs="*",
        default=None,
        help=(
            "Prepared replay label JSON files. Defaults to kbqa_classifier_mutihop/data/test/*/test_topk_labels.json. "
            "Use --replay_source predictions_root to rebuild lookup tables directly from predictions/test."
        ),
    )
    parser.add_argument("--predictions_root", default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument(
        "--replay_source",
        choices=("auto", "prepared", "predictions_root"),
        default="auto",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--fixed_ks", default=",".join(str(k) for k in DEFAULT_FIXED_KS))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_save_probabilities", type=str2bool, nargs="?", const=True, default=True)
    return parser.parse_args()


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


def resolve_output_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def has_model_files(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and (
        (path / "pytorch_model.bin").is_file()
        or (path / "model.safetensors").is_file()
        or any(path.glob("*.safetensors"))
    )


def resolve_model_dir(path_value: str) -> Path:
    path = resolve_path(path_value)
    if has_model_files(path):
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}")

    candidates = [candidate for candidate in path.rglob("*") if has_model_files(candidate)]
    if not candidates:
        raise FileNotFoundError(
            f"No model files found in {path}; expected config.json and pytorch_model.bin/model.safetensors."
        )
    candidates.sort(
        key=lambda candidate: (
            1 if not candidate.name.startswith("checkpoint-") else 0,
            candidate.stat().st_mtime,
            str(candidate),
        ),
        reverse=True,
    )
    return candidates[0]


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
        for key in ("data", "records", "examples", "items", "prediction", "predictions"):
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


def infer_field_name(records: Sequence[Dict[str, Any]], candidates: Iterable[str], field_type: str) -> str:
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
    raise KeyError(f"Could not infer {field_type} field. Available keys: {available_keys}.")


def infer_optional_field_name(records: Sequence[Dict[str, Any]], candidates: Iterable[str]) -> Optional[str]:
    try:
        return infer_field_name(records, candidates, field_type="optional")
    except KeyError:
        return None


def extract_optional_scalar(record: Dict[str, Any], field_name: Optional[str]) -> Optional[str]:
    if field_name is None or field_name not in record:
        return None
    value = record[field_name]
    if value is None or isinstance(value, (list, dict)):
        return None
    text = str(value).strip()
    return text if text else None


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
            raise ValueError("Top-k label string is empty.")
        label = int(float(stripped)) if "." in stripped else int(stripped)
    else:
        raise TypeError(f"Unsupported top-k label type: {type(raw_label).__name__}")
    if label < 1 or label > 8:
        raise ValueError(f"Multihop replay top-k label must be in [1, 8], got {label}.")
    return label


def safe_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} is an empty string.")
        return float(stripped)
    raise TypeError(f"{field_name} must be numeric-like, got {type(value).__name__}.")


def lookup_value_by_candidates(mapping: Dict[str, Any], candidates: Iterable[str]) -> Any:
    normalized_to_key = {normalize_field_name(key): key for key in mapping.keys()}
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    for candidate in candidates:
        normalized = normalize_field_name(candidate)
        if normalized in normalized_to_key:
            return mapping[normalized_to_key[normalized]]
    return None


def normalize_replay_scores(raw_scores: Any, record_index: int) -> Dict[int, Dict[str, float]]:
    normalized_scores: Dict[int, Dict[str, float]] = {}
    if isinstance(raw_scores, dict):
        iterable = raw_scores.items()
    elif isinstance(raw_scores, list):
        pairs = []
        for item_index, item in enumerate(raw_scores):
            if not isinstance(item, dict):
                raise ValueError(f"Replay score item {item_index} at record {record_index} is not an object.")
            raw_k = lookup_value_by_candidates(item, ("k", "topk", "top_k"))
            if raw_k is None:
                raise KeyError(f"Replay score item {item_index} at record {record_index} is missing k.")
            pairs.append((raw_k, item))
        iterable = pairs
    else:
        raise TypeError(f"Replay scores at record {record_index} must be a dict or list.")

    for raw_k, raw_value in iterable:
        k_value = parse_topk_label(raw_k)
        if not isinstance(raw_value, dict):
            raise TypeError(f"Replay score entry for k={k_value} at record {record_index} is not an object.")
        em_value = lookup_value_by_candidates(raw_value, EM_FIELD_CANDIDATES)
        f1_value = lookup_value_by_candidates(raw_value, F1_FIELD_CANDIDATES)
        if em_value is None:
            raise KeyError(f"Replay score entry for k={k_value} at record {record_index} is missing EM.")
        if f1_value is None:
            raise KeyError(f"Replay score entry for k={k_value} at record {record_index} is missing F1.")
        normalized_scores[k_value] = {
            "em": safe_float(em_value, field_name=f"record {record_index} k={k_value} em"),
            "f1": safe_float(f1_value, field_name=f"record {record_index} k={k_value} f1"),
        }

    if not normalized_scores:
        raise ValueError(f"Replay label record {record_index} contains no usable k scores.")
    return normalized_scores


def normalize_replay_predictions(raw_predictions: Any) -> Dict[int, str]:
    normalized: Dict[int, str] = {}
    if not isinstance(raw_predictions, dict):
        return normalized
    for raw_k, prediction in raw_predictions.items():
        try:
            k_value = parse_topk_label(raw_k)
        except (TypeError, ValueError):
            continue
        if prediction is not None:
            normalized[k_value] = str(prediction)
    return normalized


def select_best_k(scores: Dict[int, Dict[str, float]]) -> Tuple[int, Dict[str, float]]:
    return max(scores.items(), key=lambda item: (item[1]["em"], item[1]["f1"], -item[0]))


def parse_fixed_ks(raw_value: str) -> List[int]:
    fixed_ks = [parse_topk_label(chunk.strip()) for chunk in str(raw_value).split(",") if chunk.strip()]
    if not fixed_ks:
        raise ValueError("At least one fixed k must be provided.")
    return fixed_ks


def normalize_question_key(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def infer_dataset_from_id(sample_id: str) -> Optional[str]:
    for dataset_name in DATASET_NAMES:
        if f"_{dataset_name}_" in sample_id or sample_id.startswith(f"{dataset_name}_"):
            return dataset_name
    return None


def load_test_examples(test_file: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Counter]:
    records = load_records(test_file)
    if not records:
        raise ValueError(f"No records found in {test_file}.")

    question_field = infer_field_name(records, QUESTION_FIELD_CANDIDATES, field_type="question")
    label_field = infer_optional_field_name(records, LABEL_FIELD_CANDIDATES)
    id_field = infer_optional_field_name(records, ID_FIELD_CANDIDATES)
    dataset_field = infer_optional_field_name(records, DATASET_FIELD_CANDIDATES)
    raw_label_distribution: Counter = Counter()

    examples: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        question = record.get(question_field)
        if question is None or isinstance(question, (list, dict)) or not str(question).strip():
            raise ValueError(f"Question field {question_field!r} at record {index} must be non-empty scalar text.")
        sample_id = extract_optional_scalar(record, id_field) or f"sample_{index}"
        dataset_name = extract_optional_scalar(record, dataset_field) or infer_dataset_from_id(sample_id)

        raw_topk_label = None
        if label_field and label_field in record and record[label_field] is not None:
            raw_topk_label = parse_topk_label(record[label_field])
            raw_label_distribution[raw_topk_label] += 1

        examples.append(
            {
                "example_index": index,
                "id": str(sample_id),
                "question": str(question).strip(),
                "dataset_name": dataset_name,
                "raw_topk_label": raw_topk_label,
            }
        )

    metadata = {
        "question_field": question_field,
        "label_field": label_field,
        "id_field": id_field,
        "dataset_field": dataset_field,
    }
    return examples, metadata, raw_label_distribution


def infer_scores_field(records: Sequence[Dict[str, Any]]) -> str:
    return infer_field_name(records, SCORES_FIELD_CANDIDATES, field_type="replay scores")


def normalize_prepared_replay_records(
    records: Sequence[Dict[str, Any]],
    source_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    question_field = infer_optional_field_name(records, QUESTION_FIELD_CANDIDATES)
    id_field = infer_optional_field_name(records, ID_FIELD_CANDIDATES)
    dataset_field = infer_optional_field_name(records, DATASET_FIELD_CANDIDATES)
    scores_field = infer_scores_field(records)
    best_k_field = infer_optional_field_name(records, BEST_K_FIELD_CANDIDATES)
    predictions_field = infer_optional_field_name(records, PREDICTIONS_FIELD_CANDIDATES)
    stats: Counter = Counter()
    normalized_records: List[Dict[str, Any]] = []

    for index, record in enumerate(records):
        sample_id = extract_optional_scalar(record, id_field)
        question = extract_optional_scalar(record, question_field)
        dataset_name = extract_optional_scalar(record, dataset_field)
        if sample_id is None and question is None:
            raise ValueError(f"Replay record {index} in {source_path} has neither id nor question.")

        scores = normalize_replay_scores(record.get(scores_field), record_index=index)
        raw_predictions = record.get(predictions_field) if predictions_field else None
        predictions = normalize_replay_predictions(raw_predictions)
        if best_k_field and best_k_field in record:
            best_k = parse_topk_label(record[best_k_field])
            stats["explicit_best_k_count"] += 1
        else:
            best_k, _ = select_best_k(scores)
            stats["computed_best_k_count"] += 1

        normalized_records.append(
            {
                "record_index": index,
                "id": str(sample_id) if sample_id is not None else None,
                "question": str(question) if question is not None else None,
                "dataset_name": dataset_name,
                "best_k": int(best_k),
                "scores": scores,
                "predictions": predictions,
                "source": str(source_path),
            }
        )
    return normalized_records, dict(stats)


def load_prepared_replay_records(replay_label_files: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    source_files: List[str] = []
    for replay_path in replay_label_files:
        records = load_records(replay_path)
        normalized_records, file_stats = normalize_prepared_replay_records(records, replay_path)
        all_records.extend(normalized_records)
        stats.update(file_stats)
        source_files.append(str(replay_path))
    metadata = {"source": "prepared", "source_files": source_files, **dict(stats)}
    return all_records, metadata


def find_single_per_question_file(directory: Path) -> Path:
    matches = sorted(directory.glob("per_question_eval__*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one per_question_eval__*.json in {directory}, found {len(matches)}.")
    return matches[0]


def load_per_question_eval_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    records = load_records(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        sample_id = extract_optional_scalar(record, "id")
        if sample_id is None:
            raise KeyError(f"Missing id at item {index} in {path}.")
        em_value = lookup_value_by_candidates(record, EM_FIELD_CANDIDATES)
        f1_value = lookup_value_by_candidates(record, F1_FIELD_CANDIDATES)
        if em_value is None or f1_value is None:
            raise KeyError(f"Missing EM/F1 for id {sample_id!r} in {path}.")
        prediction = extract_optional_scalar(record, "prediction")
        by_id[str(sample_id)] = {
            "em": safe_float(em_value, f"{path} {sample_id} em"),
            "f1": safe_float(f1_value, f"{path} {sample_id} f1"),
            "prediction": prediction,
        }
    return by_id


def load_replay_records_from_predictions_root(
    predictions_root: Path,
    examples: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not predictions_root.is_dir():
        raise FileNotFoundError(f"Predictions root not found: {predictions_root}")

    example_by_id = {str(example["id"]): example for example in examples}
    scores_by_id: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    predictions_by_id: Dict[str, Dict[int, str]] = defaultdict(dict)
    source_dirs: Dict[str, Dict[int, str]] = defaultdict(dict)

    for dataset_name in DATASET_NAMES:
        for k_value in range(1, 9):
            directory = (
                predictions_root
                / f"ircot_qa_qwen_{dataset_name}____prompt_set_1___bm25_retrieval_count__{k_value}___distractor_count__1"
            )
            per_question_file = find_single_per_question_file(directory)
            for sample_id, result in load_per_question_eval_by_id(per_question_file).items():
                if sample_id not in example_by_id:
                    continue
                scores_by_id[sample_id][int(k_value)] = {
                    "em": float(result["em"]),
                    "f1": float(result["f1"]),
                }
                if result.get("prediction") is not None:
                    predictions_by_id[sample_id][int(k_value)] = str(result["prediction"])
                source_dirs[dataset_name][int(k_value)] = str(directory)

    replay_records: List[Dict[str, Any]] = []
    for index, example in enumerate(examples):
        sample_id = str(example["id"])
        scores = scores_by_id.get(sample_id)
        if not scores:
            continue
        best_k, _ = select_best_k(scores)
        replay_records.append(
            {
                "record_index": index,
                "id": sample_id,
                "question": example["question"],
                "dataset_name": example.get("dataset_name"),
                "best_k": int(best_k),
                "scores": scores,
                "predictions": predictions_by_id.get(sample_id, {}),
                "source": str(predictions_root),
            }
        )

    metadata = {
        "source": "predictions_root",
        "predictions_root": str(predictions_root),
        "source_dirs_by_dataset": source_dirs,
        "computed_best_k_count": len(replay_records),
    }
    return replay_records, metadata


def build_replay_indices(
    replay_records: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Counter]:
    by_id: Dict[str, Dict[str, Any]] = {}
    question_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stats: Counter = Counter()

    for record in replay_records:
        sample_id = record.get("id")
        if sample_id is not None:
            sample_id = str(sample_id)
            if sample_id in by_id:
                raise ValueError(f"Duplicate replay id detected: {sample_id}")
            by_id[sample_id] = record
            stats["unique_ids"] += 1
        question = record.get("question")
        if question is not None:
            question_to_records[normalize_question_key(str(question))].append(record)

    by_question: Dict[str, Dict[str, Any]] = {}
    for question_key, question_records in question_to_records.items():
        if len(question_records) == 1:
            by_question[question_key] = question_records[0]
        else:
            stats["ambiguous_question_keys"] += 1
    stats["unique_question_keys"] = len(by_question)
    return by_id, by_question, stats


def align_examples_with_replay(
    examples: Sequence[Dict[str, Any]],
    replay_by_id: Dict[str, Dict[str, Any]],
    replay_by_question: Dict[str, Dict[str, Any]],
) -> Counter:
    stats: Counter = Counter()
    for example in examples:
        aligned_record = None
        matched_by = None
        sample_id = str(example["id"])
        if sample_id in replay_by_id:
            aligned_record = replay_by_id[sample_id]
            matched_by = "id"
        else:
            question_key = normalize_question_key(example["question"])
            if question_key in replay_by_question:
                aligned_record = replay_by_question[question_key]
                matched_by = "question"

        example["aligned_replay"] = aligned_record
        example["aligned"] = aligned_record is not None
        example["matched_by"] = matched_by

        if aligned_record is None:
            stats["missing_alignment_count"] += 1
            continue

        stats[f"matched_by_{matched_by}_count"] += 1
        if example.get("raw_topk_label") is None:
            example["raw_topk_label"] = parse_topk_label(aligned_record["best_k"])
        if example.get("dataset_name") is None:
            example["dataset_name"] = aligned_record.get("dataset_name") or infer_dataset_from_id(sample_id)

    stats["aligned_count"] = stats["matched_by_id_count"] + stats["matched_by_question_count"]
    return stats


def choose_device(raw_device: Optional[str]) -> torch.device:
    if raw_device:
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def predict_logits_on_device(
    texts: Sequence[str],
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

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
            logits = model(**encoded).logits.detach().cpu().numpy()
            logits_batches.append(logits)

    if not logits_batches:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), []
    logits = np.concatenate(logits_batches, axis=0)
    probabilities = softmax(logits)
    predictions = np.argmax(logits, axis=-1).astype(int).tolist()
    return logits, probabilities, predictions


def compute_label_accuracy(predictions: Sequence[int], gold_labels: Sequence[int]) -> Dict[str, Any]:
    if len(predictions) != len(gold_labels):
        raise ValueError("predictions and gold_labels must have the same length.")
    label_ids = list(range(1, 9))
    label_to_index = {label_id: index for index, label_id in enumerate(label_ids)}
    confusion = np.zeros((len(label_ids), len(label_ids)), dtype=np.int64)
    correct_count = 0
    for pred_label, gold_label in zip(predictions, gold_labels):
        confusion[label_to_index[int(gold_label)], label_to_index[int(pred_label)]] += 1
        correct_count += int(pred_label == gold_label)
    sample_count = len(gold_labels)
    return {
        "sample_count": sample_count,
        "accuracy": float(correct_count / sample_count) if sample_count else None,
        "correct_count": correct_count,
        "confusion_matrix": {
            "label_ids": label_ids,
            "matrix": confusion.tolist(),
        },
    }


def evaluate_replay_system(
    system_name: str,
    examples: Sequence[Dict[str, Any]],
    requested_k_resolver,
    total_input_sample_count: int,
) -> Dict[str, Any]:
    k_sum = 0
    k_determined_sample_count = 0
    aligned_sample_count = 0
    evaluated_sample_count = 0
    missing_alignment_count = 0
    missing_score_count = 0
    em_sum = 0.0
    f1_sum = 0.0

    for example in examples:
        proposed_k = requested_k_resolver(example)
        if proposed_k is not None:
            proposed_k = int(proposed_k)
            k_determined_sample_count += 1
            k_sum += proposed_k

        replay_record = example.get("aligned_replay")
        if replay_record is None:
            missing_alignment_count += 1
            continue
        aligned_sample_count += 1
        if proposed_k is None:
            missing_score_count += 1
            continue
        score_entry = replay_record["scores"].get(int(proposed_k))
        if score_entry is None:
            missing_score_count += 1
            continue

        evaluated_sample_count += 1
        em_sum += float(score_entry["em"])
        f1_sum += float(score_entry["f1"])

    return {
        "system_name": system_name,
        "input_sample_count": total_input_sample_count,
        "k_determined_sample_count": k_determined_sample_count,
        "aligned_sample_count": aligned_sample_count,
        "evaluated_sample_count": evaluated_sample_count,
        "skipped_missing_alignment_count": missing_alignment_count,
        "skipped_missing_score_count": missing_score_count,
        "avg_em": float(em_sum / evaluated_sample_count) if evaluated_sample_count else None,
        "avg_f1": float(f1_sum / evaluated_sample_count) if evaluated_sample_count else None,
        "avg_topk": float(k_sum / k_determined_sample_count) if k_determined_sample_count else None,
    }


def write_summary_csv(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "system_name",
        "avg_em",
        "avg_f1",
        "avg_topk",
        "classifier_acc",
        "input_sample_count",
        "k_determined_sample_count",
        "aligned_sample_count",
        "evaluated_sample_count",
        "skipped_missing_alignment_count",
        "skipped_missing_score_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def float_to_display(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def print_comparison_table(rows: Sequence[Dict[str, Any]]) -> None:
    columns = [
        ("system_name", "system"),
        ("avg_em", "avg_em"),
        ("avg_f1", "avg_f1"),
        ("avg_topk", "avg_topk"),
        ("classifier_acc", "clf_acc"),
        ("evaluated_sample_count", "eval_n"),
        ("skipped_missing_alignment_count", "missing_align"),
        ("skipped_missing_score_count", "missing_score"),
    ]
    formatted_rows: List[List[str]] = []
    for row in rows:
        formatted_row = []
        for field_name, _ in columns:
            value = row.get(field_name)
            if field_name.startswith("avg_") or field_name == "classifier_acc":
                formatted_row.append(float_to_display(value))
            else:
                formatted_row.append(str(value))
        formatted_rows.append(formatted_row)

    widths = []
    for column_index, (_, header) in enumerate(columns):
        widths.append(max([len(header)] + [len(row[column_index]) for row in formatted_rows]))
    print(" | ".join(header.ljust(widths[index]) for index, (_, header) in enumerate(columns)))
    print("-+-".join("-" * width for width in widths))
    for formatted_row in formatted_rows:
        print(" | ".join(formatted_row[index].ljust(widths[index]) for index in range(len(columns))))


def sanitize_component(value: Optional[str], fallback: str) -> str:
    if value is None or not str(value).strip():
        return fallback
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return sanitized or fallback


def dataset_subset_rows(system_rows: Sequence[Dict[str, Any]], examples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_dataset[str(example.get("dataset_name") or "unknown")].append(example)

    dataset_results: Dict[str, List[Dict[str, Any]]] = {}
    for dataset_name, dataset_examples in sorted(by_dataset.items()):
        rows = []
        total = len(dataset_examples)
        for row in system_rows:
            system_name = row["system_name"]
            if system_name.startswith("fixed_k_"):
                fixed_k = int(system_name.rsplit("_", 1)[-1])
                resolver = lambda _example, chosen_k=fixed_k: chosen_k
            elif system_name == "classifier_dynamic_topk":
                resolver = lambda example: example.get("pred_topk")
            elif system_name == "oracle_best_k":
                resolver = lambda example: example.get("oracle_best_k")
            else:
                continue
            rows.append(evaluate_replay_system(system_name, dataset_examples, resolver, total))
        dataset_results[dataset_name] = rows
    return dataset_results


def get_replay_label_files(args: argparse.Namespace) -> List[Path]:
    raw_files = args.replay_label_files if args.replay_label_files else list(DEFAULT_REPLAY_LABEL_FILES)
    return [resolve_path(path_value) for path_value in raw_files]


def load_replay_records(args: argparse.Namespace, examples: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    replay_label_files = get_replay_label_files(args)
    prepared_files_exist = all(path.is_file() for path in replay_label_files)

    if args.replay_source == "prepared":
        return load_prepared_replay_records(replay_label_files)
    if args.replay_source == "predictions_root":
        return load_replay_records_from_predictions_root(resolve_path(args.predictions_root), examples)
    if prepared_files_exist:
        return load_prepared_replay_records(replay_label_files)
    return load_replay_records_from_predictions_root(resolve_path(args.predictions_root), examples)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_dir = resolve_model_dir(args.model_dir)
    test_file = resolve_path(args.test_file)
    output_root = resolve_output_path(args.output_dir)
    run_name = sanitize_component(
        args.run_name,
        fallback=f"end2end_replay_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}",
    )
    run_output_dir = output_root / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    fixed_ks = parse_fixed_ks(args.fixed_ks)
    device = choose_device(args.device)

    test_examples, test_metadata, raw_label_distribution = load_test_examples(test_file)
    replay_records, replay_metadata = load_replay_records(args, test_examples)
    replay_by_id, replay_by_question, replay_index_stats = build_replay_indices(replay_records)
    alignment_stats = align_examples_with_replay(test_examples, replay_by_id, replay_by_question)

    if not replay_records:
        raise ValueError("No replay records were loaded.")

    model = load_classifier_model(str(model_dir))
    tokenizer = load_classifier_tokenizer(str(model_dir))
    logits, probabilities, pred_class_ids = predict_logits_on_device(
        texts=[example["question"] for example in test_examples],
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    for example, logit_row, probability_row, pred_class_id in zip(
        test_examples, logits, probabilities, pred_class_ids
    ):
        pred_class_id = int(pred_class_id)
        pred_topk = pred_class_id + 1
        replay_record = example.get("aligned_replay")
        replay_score = replay_record["scores"].get(pred_topk) if replay_record is not None else None
        example["pred_class_id"] = pred_class_id
        example["pred_label_name"] = CLASS_ID2LABEL.get(pred_class_id, str(pred_class_id))
        example["pred_topk"] = int(pred_topk)
        example["classifier_logits"] = [float(value) for value in logit_row.tolist()]
        example["classifier_probabilities"] = [float(value) for value in probability_row.tolist()]
        example["oracle_best_k"] = int(replay_record["best_k"]) if replay_record is not None else None
        example["available_score_ks"] = sorted(replay_record["scores"].keys()) if replay_record is not None else []
        example["replay_score_available"] = replay_score is not None
        example["replay_em"] = float(replay_score["em"]) if replay_score is not None else None
        example["replay_f1"] = float(replay_score["f1"]) if replay_score is not None else None
        example["selected_prediction"] = (
            replay_record.get("predictions", {}).get(pred_topk) if replay_record is not None else None
        )

    gold_labels = [int(example["raw_topk_label"]) for example in test_examples if example.get("raw_topk_label") is not None]
    pred_labels_for_acc = [
        int(example["pred_topk"]) for example in test_examples if example.get("raw_topk_label") is not None
    ]
    classifier_metrics = compute_label_accuracy(pred_labels_for_acc, gold_labels)

    total_input_sample_count = len(test_examples)
    supported_ks = sorted({int(k) for record in replay_records for k in record["scores"].keys()})
    system_rows: List[Dict[str, Any]] = []
    for fixed_k in fixed_ks:
        system_rows.append(
            evaluate_replay_system(
                system_name=f"fixed_k_{fixed_k}",
                examples=test_examples,
                requested_k_resolver=lambda _example, chosen_k=fixed_k: chosen_k,
                total_input_sample_count=total_input_sample_count,
            )
        )

    classifier_row = evaluate_replay_system(
        system_name="classifier_dynamic_topk",
        examples=test_examples,
        requested_k_resolver=lambda example: example.get("pred_topk"),
        total_input_sample_count=total_input_sample_count,
    )
    classifier_row["classifier_acc"] = classifier_metrics["accuracy"]
    system_rows.append(classifier_row)
    system_rows.append(
        evaluate_replay_system(
            system_name="oracle_best_k",
            examples=test_examples,
            requested_k_resolver=lambda example: example.get("oracle_best_k"),
            total_input_sample_count=total_input_sample_count,
        )
    )

    for row in system_rows:
        row.setdefault("classifier_acc", None)

    summary_json_path = run_output_dir / "summary.json"
    summary_csv_path = run_output_dir / "summary.csv"
    per_example_path = run_output_dir / "per_example_predictions.jsonl"
    config_json_path = run_output_dir / "config.json"

    per_example_rows: List[Dict[str, Any]] = []
    for example in test_examples:
        replay_record = example.get("aligned_replay")
        row = {
            "example_index": example["example_index"],
            "id": example["id"],
            "question": example["question"],
            "dataset_name": example.get("dataset_name"),
            "aligned": bool(example.get("aligned")),
            "matched_by": example.get("matched_by"),
            "replay_record_id": replay_record.get("id") if replay_record is not None else None,
            "gold_topk_label": example.get("raw_topk_label"),
            "pred_topk": example.get("pred_topk"),
            "pred_class_id": example.get("pred_class_id"),
            "pred_label_name": example.get("pred_label_name"),
            "oracle_best_k": example.get("oracle_best_k"),
            "available_score_ks": example.get("available_score_ks"),
            "replay_score_available": example.get("replay_score_available"),
            "replay_em": example.get("replay_em"),
            "replay_f1": example.get("replay_f1"),
            "selected_prediction": example.get("selected_prediction"),
        }
        if args.do_save_probabilities:
            row["classifier_logits"] = example.get("classifier_logits")
            row["classifier_probabilities"] = example.get("classifier_probabilities")
        per_example_rows.append(row)

    dataset_results = dataset_subset_rows(system_rows, test_examples)
    summary_payload = {
        "task": "kbqa_classifier_mutihop_end2end_replay_eval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "model_dir": str(model_dir),
            "test_file": str(test_file),
            "output_dir": str(run_output_dir),
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "per_example_predictions": str(per_example_path),
            "config_json": str(config_json_path),
        },
        "test_file_metadata": {
            **test_metadata,
            "raw_label_distribution": {str(k): v for k, v in sorted(raw_label_distribution.items())},
        },
        "replay_label_metadata": replay_metadata,
        "data_stats": {
            "input_sample_count": total_input_sample_count,
            "replay_record_count": len(replay_records),
            "aligned_sample_count": alignment_stats["aligned_count"],
            "matched_by_id_count": alignment_stats["matched_by_id_count"],
            "matched_by_question_count": alignment_stats["matched_by_question_count"],
            "missing_alignment_count": alignment_stats["missing_alignment_count"],
            "replay_unique_id_count": replay_index_stats["unique_ids"],
            "replay_unique_question_key_count": replay_index_stats["unique_question_keys"],
            "replay_ambiguous_question_key_count": replay_index_stats["ambiguous_question_keys"],
        },
        "selector_metadata": {
            "device": str(device),
            "fixed_k_values": fixed_ks,
            "supported_k_values": supported_ks,
            "supported_k_min": supported_ks[0] if supported_ks else None,
            "supported_k_max": supported_ks[-1] if supported_ks else None,
            "class_id_to_topk_rule": "pred_topk = pred_class_id + 1",
            "id2label": {str(key): value for key, value in CLASS_ID2LABEL.items()},
        },
        "classifier_metrics": classifier_metrics,
        "warning_counts": {
            "missing_alignment_count": alignment_stats["missing_alignment_count"],
            "missing_score_count_by_system": {
                row["system_name"]: row["skipped_missing_score_count"] for row in system_rows
            },
        },
        "system_results": system_rows,
        "dataset_results": dataset_results,
    }
    config_payload = {
        "seed": args.seed,
        "model_dir": str(model_dir),
        "test_file": str(test_file),
        "replay_source": replay_metadata.get("source"),
        "predictions_root": str(resolve_path(args.predictions_root)),
        "replay_label_files": [str(path) for path in get_replay_label_files(args)],
        "output_dir": str(run_output_dir),
        "fixed_k_values": fixed_ks,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": str(device),
        "supported_k_values": supported_ks,
        "do_save_probabilities": args.do_save_probabilities,
    }

    save_json(summary_json_path, summary_payload)
    save_json(config_json_path, config_payload)
    write_summary_csv(summary_csv_path, system_rows)
    write_jsonl(per_example_path, per_example_rows)

    print("Offline end-to-end replay evaluation finished.")
    print(
        f"Input samples: {total_input_sample_count} | "
        f"Aligned samples: {alignment_stats['aligned_count']} | "
        f"Missing alignment: {alignment_stats['missing_alignment_count']}"
    )
    print(
        f"Classifier Avg EM: {float_to_display(classifier_row['avg_em'])} | "
        f"Avg F1: {float_to_display(classifier_row['avg_f1'])} | "
        f"Avg Top-k: {float_to_display(classifier_row['avg_topk'])} | "
        f"Classifier Acc: {float_to_display(classifier_metrics['accuracy'])}"
    )
    print(f"Summary JSON saved to: {summary_json_path}")
    print(f"Summary CSV saved to: {summary_csv_path}")
    print(f"Per-example predictions saved to: {per_example_path}")
    print("")
    print_comparison_table(system_rows)


if __name__ == "__main__":
    main()
