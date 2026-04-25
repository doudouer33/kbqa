#!/usr/bin/env python
"""Offline replay evaluation for the two-stage dynamic top-k selector."""

import argparse
import csv
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


from kbqa_classifier.topk_data_builder import select_best_k
from kbqa_classifier.train.train_stage1 import (
    DATASET_NAME_FIELD_CANDIDATES,
    ID_FIELD_CANDIDATES,
    LABEL_FIELD_CANDIDATES,
    QUESTION_FIELD_CANDIDATES,
    extract_optional_scalar,
    extract_question_text,
    get_available_keys,
    infer_dataset_name,
    infer_field_name,
    infer_model_tag,
    infer_optional_field_name,
    infer_split_name,
    load_records_from_file,
    make_default_sample_id,
    normalize_field_name,
    parse_raw_label,
    resolve_input_path,
    resolve_output_path,
    sanitize_artifact_component,
    save_json,
    set_reproducible_seed,
    softmax,
    stage1_label_from_raw,
    str2bool,
    write_jsonl,
)
from kbqa_classifier.train.train_stage2 import load_stage2_label_metadata


DEFAULT_OUTPUT_ROOT = "kbqa_classifier/output/end2end_replay"
DEFAULT_FIXED_KS = (0, 1, 3, 5, 10, 15)
BEST_K_FIELD_CANDIDATES: Tuple[str, ...] = (
    "best_k",
    "bestk",
    "oracle_best_k",
    "oracle_k",
)
SCORES_FIELD_CANDIDATES: Tuple[str, ...] = (
    "scores",
    "score_by_k",
    "score_map",
    "k_scores",
    "topk_scores",
    "per_k_scores",
    "results_by_k",
)
EM_FIELD_CANDIDATES: Tuple[str, ...] = ("em", "exact_match", "exact")
F1_FIELD_CANDIDATES: Tuple[str, ...] = ("f1", "f1_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline replay evaluation for fixed-k, oracle, and two-stage dynamic top-k selector."
    )
    parser.add_argument("--stage1_model_dir", type=str, required=True, help="Stage-1 checkpoint or model directory.")
    parser.add_argument("--stage2_model_dir", type=str, required=True, help="Stage-2 checkpoint or model directory.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the evaluation questions file.")
    parser.add_argument(
        "--replay_label_file",
        type=str,
        required=True,
        help="Path to the detailed top-k replay label file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory used to save replay evaluation artifacts.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional dataset tag. If omitted, infer from test file / records.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help="Optional split tag. If omitted, infer from the test filename.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional subdirectory name inside output_dir/<dataset_name>/.",
    )
    parser.add_argument(
        "--fixed_ks",
        type=str,
        default=",".join(str(k) for k in DEFAULT_FIXED_KS),
        help="Comma-separated fixed-k baselines to replay. Default: 0,1,3,5,10,15",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Default batch size for both stages unless stage-specific overrides are provided.",
    )
    parser.add_argument(
        "--stage1_batch_size",
        type=int,
        default=None,
        help="Optional batch size override for stage-1 inference.",
    )
    parser.add_argument(
        "--stage2_batch_size",
        type=int,
        default=None,
        help="Optional batch size override for stage-2 inference.",
    )
    parser.add_argument(
        "--stage1_max_length",
        type=int,
        default=128,
        help="Maximum tokenized question length for stage-1 inference.",
    )
    parser.add_argument(
        "--stage2_max_length",
        type=int,
        default=64,
        help="Maximum tokenized question length for stage-2 inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string such as cpu, cuda, or cuda:0. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--bucket_to_topk_strategy",
        type=str,
        default="lower_bound",
        choices=["lower_bound", "upper_bound", "midpoint"],
        help=(
            "How to turn a predicted stage-2 bucket into a replay top-k. "
            "Default uses the lower bound of each bucket to stay aligned with the minimal-k label preference."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic inference order.")
    parser.add_argument(
        "--do_save_stage_probabilities",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to store stage-1 / stage-2 probabilities in per_example_predictions.jsonl.",
    )
    return parser.parse_args()


def parse_fixed_ks(raw_value: str) -> List[int]:
    fixed_ks: List[int] = []
    for chunk in str(raw_value).split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        fixed_ks.append(parse_raw_label(stripped))

    if not fixed_ks:
        raise ValueError("At least one fixed-k baseline must be provided via --fixed_ks.")
    return fixed_ks


def normalize_question_key(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


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
    raise TypeError(f"{field_name} must be numeric-like, but got {type(value).__name__}.")


def infer_optional_question_field(records: Sequence[Dict[str, Any]]) -> Optional[str]:
    try:
        return infer_field_name(records, QUESTION_FIELD_CANDIDATES, field_type="question")
    except KeyError:
        return None


def infer_optional_label_field(records: Sequence[Dict[str, Any]]) -> Optional[str]:
    try:
        return infer_field_name(records, LABEL_FIELD_CANDIDATES, field_type="label")
    except KeyError:
        return None


def infer_scores_field(records: Sequence[Dict[str, Any]]) -> str:
    available_keys = get_available_keys(records)
    normalized_to_key = {normalize_field_name(key): key for key in available_keys}

    for candidate in SCORES_FIELD_CANDIDATES:
        if candidate in available_keys:
            return candidate

    for candidate in SCORES_FIELD_CANDIDATES:
        normalized_candidate = normalize_field_name(candidate)
        if normalized_candidate in normalized_to_key:
            return normalized_to_key[normalized_candidate]

    for key in available_keys:
        for record in records:
            value = record.get(key)
            if not isinstance(value, dict):
                continue
            if not value:
                continue
            first_value = next(iter(value.values()))
            if isinstance(first_value, dict):
                normalized_child_keys = {normalize_field_name(child_key) for child_key in first_value.keys()}
                if normalize_field_name("em") in normalized_child_keys and normalize_field_name("f1") in normalized_child_keys:
                    return key

    raise KeyError(
        f"Could not infer the replay scores field. Available keys: {available_keys}. "
        f"Tried preferred candidates: {list(SCORES_FIELD_CANDIDATES)}."
    )


def lookup_value_by_candidates(mapping: Dict[str, Any], candidates: Iterable[str]) -> Any:
    normalized_to_key = {normalize_field_name(key): key for key in mapping.keys()}

    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]

    for candidate in candidates:
        normalized_candidate = normalize_field_name(candidate)
        if normalized_candidate in normalized_to_key:
            return mapping[normalized_to_key[normalized_candidate]]

    return None


def normalize_replay_scores(raw_scores: Any, record_index: int) -> Dict[int, Dict[str, float]]:
    normalized_scores: Dict[int, Dict[str, float]] = {}

    if isinstance(raw_scores, dict):
        iterable = raw_scores.items()
    elif isinstance(raw_scores, list):
        iterable = []
        for item_index, item in enumerate(raw_scores):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Replay score list item {item_index} at record {record_index} must be a JSON object."
                )
            raw_k = lookup_value_by_candidates(item, ("k", "topk", "top_k"))
            if raw_k is None:
                raise KeyError(f"Replay score list item {item_index} at record {record_index} is missing a k field.")
            iterable.append((raw_k, item))
    else:
        raise TypeError(
            f"Replay scores at record {record_index} must be a dict or list, but got {type(raw_scores).__name__}."
        )

    for raw_k, raw_value in iterable:
        k_value = parse_raw_label(raw_k)
        if not isinstance(raw_value, dict):
            raise TypeError(
                f"Replay score entry for k={k_value} at record {record_index} must be a JSON object, "
                f"but got {type(raw_value).__name__}."
            )

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


def load_test_examples(
    test_file: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]], Counter]:
    records = load_records_from_file(test_file)
    if not records:
        raise ValueError(f"No records found in {test_file}.")

    question_field = infer_field_name(records, QUESTION_FIELD_CANDIDATES, field_type="question")
    label_field = infer_optional_label_field(records)
    id_field = infer_optional_field_name(records, ID_FIELD_CANDIDATES, field_type="id")
    dataset_field = infer_optional_field_name(records, DATASET_NAME_FIELD_CANDIDATES, field_type="dataset_name")

    raw_label_distribution: Counter = Counter()
    examples: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        question = extract_question_text(record, question_field, index)
        raw_topk_label: Optional[int] = None
        stage1_gt: Optional[int] = None

        if label_field and label_field in record:
            raw_topk_label = parse_raw_label(record[label_field])
            raw_label_distribution[raw_topk_label] += 1
            stage1_gt = stage1_label_from_raw(raw_topk_label)

        sample_id = extract_optional_scalar(record, id_field)
        dataset_name = extract_optional_scalar(record, dataset_field)
        examples.append(
            {
                "example_index": index,
                "id": sample_id if sample_id is not None else make_default_sample_id(index),
                "question": question,
                "dataset_name": dataset_name,
                "raw_topk_label": raw_topk_label,
                "stage1_gt": stage1_gt,
            }
        )

    metadata = {
        "question_field": question_field,
        "label_field": label_field,
        "id_field": id_field,
        "dataset_field": dataset_field,
    }
    return examples, metadata, raw_label_distribution


def load_replay_label_records(
    replay_label_file: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]], Dict[str, int]]:
    records = load_records_from_file(replay_label_file)
    if not records:
        raise ValueError(f"No records found in {replay_label_file}.")

    question_field = infer_optional_question_field(records)
    id_field = infer_optional_field_name(records, ID_FIELD_CANDIDATES, field_type="id")
    dataset_field = infer_optional_field_name(records, DATASET_NAME_FIELD_CANDIDATES, field_type="dataset_name")
    best_k_field = infer_optional_field_name(records, BEST_K_FIELD_CANDIDATES, field_type="best_k")
    scores_field = infer_scores_field(records)

    normalized_records: List[Dict[str, Any]] = []
    stats = {
        "computed_best_k_count": 0,
        "explicit_best_k_count": 0,
        "records_without_id_count": 0,
        "records_without_question_count": 0,
    }

    for index, record in enumerate(records):
        sample_id = extract_optional_scalar(record, id_field)
        question = None
        if question_field and question_field in record:
            raw_question = extract_optional_scalar(record, question_field)
            if raw_question is not None:
                question = str(raw_question)
        dataset_name = extract_optional_scalar(record, dataset_field)

        if sample_id is None:
            stats["records_without_id_count"] += 1
        if question is None:
            stats["records_without_question_count"] += 1
        if sample_id is None and question is None:
            raise ValueError(
                f"Replay label record {index} in {replay_label_file} has neither an id field nor a question field."
            )

        raw_scores = record.get(scores_field)
        if raw_scores is None:
            raise KeyError(f"Replay label record {index} is missing the inferred scores field {scores_field!r}.")
        scores = normalize_replay_scores(raw_scores, record_index=index)

        if best_k_field and best_k_field in record:
            best_k = parse_raw_label(record[best_k_field])
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
                "best_k": best_k,
                "scores": scores,
            }
        )

    metadata = {
        "question_field": question_field,
        "id_field": id_field,
        "dataset_field": dataset_field,
        "best_k_field": best_k_field,
        "scores_field": scores_field,
    }
    return normalized_records, metadata, stats


def build_replay_indices(
    replay_records: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Counter]:
    records_by_id: Dict[str, Dict[str, Any]] = {}
    question_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stats: Counter = Counter()

    for record in replay_records:
        sample_id = record.get("id")
        if sample_id is not None:
            sample_id = str(sample_id)
            if sample_id in records_by_id:
                raise ValueError(f"Duplicate replay label id detected: {sample_id}")
            records_by_id[sample_id] = record
            stats["unique_ids"] += 1

        question = record.get("question")
        if question is not None:
            question_to_records[normalize_question_key(str(question))].append(record)

    records_by_question: Dict[str, Dict[str, Any]] = {}
    ambiguous_question_count = 0
    for question_key, question_records in question_to_records.items():
        if len(question_records) == 1:
            records_by_question[question_key] = question_records[0]
            continue
        ambiguous_question_count += 1

    stats["unique_question_keys"] = len(records_by_question)
    stats["ambiguous_question_keys"] = ambiguous_question_count
    return records_by_id, records_by_question, stats


def align_examples_with_replay(
    examples: Sequence[Dict[str, Any]],
    replay_by_id: Dict[str, Dict[str, Any]],
    replay_by_question: Dict[str, Dict[str, Any]],
) -> Counter:
    alignment_stats: Counter = Counter()

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
            alignment_stats["missing_alignment_count"] += 1
        elif matched_by == "id":
            alignment_stats["matched_by_id_count"] += 1
        elif matched_by == "question":
            alignment_stats["matched_by_question_count"] += 1

        if example.get("stage1_gt") is None and aligned_record is not None:
            oracle_best_k = parse_raw_label(aligned_record["best_k"])
            example["stage1_gt"] = stage1_label_from_raw(oracle_best_k)
            example["raw_topk_label"] = oracle_best_k

    alignment_stats["aligned_count"] = alignment_stats["matched_by_id_count"] + alignment_stats["matched_by_question_count"]
    return alignment_stats


def choose_device(raw_device: Optional[str]) -> torch.device:
    if raw_device:
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_logits_on_device(
    texts: Sequence[str],
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, but got {batch_size}.")

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

    if not logits_batches:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), []

    logits = np.concatenate(logits_batches, axis=0)
    probabilities = softmax(logits)
    predictions = np.argmax(logits, axis=-1).astype(int).tolist()
    return logits, probabilities, predictions


def normalize_bucket_definitions(raw_bucket_definitions: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    normalized: Dict[int, Dict[str, Any]] = {}
    for raw_bucket_id, bucket_info in raw_bucket_definitions.items():
        bucket_id = int(raw_bucket_id)
        if not isinstance(bucket_info, dict):
            raise TypeError(f"Bucket definition for {raw_bucket_id!r} must be a JSON object.")
        topk_range = bucket_info.get("topk_range")
        if not isinstance(topk_range, list) or len(topk_range) != 2:
            raise ValueError(f"Bucket definition for {raw_bucket_id!r} must contain a 2-item topk_range list.")
        lower_bound = parse_raw_label(topk_range[0])
        upper_bound = parse_raw_label(topk_range[1])
        if upper_bound < lower_bound:
            raise ValueError(f"Bucket definition for {raw_bucket_id!r} has an invalid topk_range: {topk_range}")

        normalized[bucket_id] = {
            "label_name": bucket_info.get("label_name", str(bucket_id)),
            "description": bucket_info.get("description"),
            "topk_range": [lower_bound, upper_bound],
        }

    if not normalized:
        raise ValueError("No usable stage-2 bucket definitions were found.")
    return normalized


def bucket_id_from_topk(topk_value: int, bucket_definitions: Dict[int, Dict[str, Any]]) -> Optional[int]:
    if topk_value <= 0:
        return None

    for bucket_id, bucket_info in sorted(bucket_definitions.items()):
        lower_bound, upper_bound = bucket_info["topk_range"]
        if lower_bound <= topk_value <= upper_bound:
            return int(bucket_id)
    return None


def map_bucket_to_replay_k(
    bucket_id: int,
    bucket_definitions: Dict[int, Dict[str, Any]],
    strategy: str,
) -> int:
    if bucket_id not in bucket_definitions:
        raise KeyError(f"Unknown stage-2 bucket id: {bucket_id}")

    lower_bound, upper_bound = bucket_definitions[bucket_id]["topk_range"]
    if strategy == "lower_bound":
        return lower_bound
    if strategy == "upper_bound":
        return upper_bound
    if strategy == "midpoint":
        return int(round((lower_bound + upper_bound) / 2.0))
    raise ValueError(f"Unsupported bucket_to_topk_strategy: {strategy}")


def clip_k_to_supported_range(
    proposed_k: int,
    min_supported_k: int,
    max_supported_k: int,
) -> Tuple[int, bool]:
    clipped_k = min(max(proposed_k, min_supported_k), max_supported_k)
    return clipped_k, clipped_k != proposed_k


def compute_binary_accuracy(predictions: Sequence[int], gold_labels: Sequence[int]) -> Dict[str, Any]:
    if len(predictions) != len(gold_labels):
        raise ValueError("predictions and gold_labels must have the same length.")
    if not gold_labels:
        return {
            "sample_count": 0,
            "accuracy": None,
            "correct_count": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    confusion = [[0, 0], [0, 0]]
    correct_count = 0
    for pred_label, gold_label in zip(predictions, gold_labels):
        confusion[int(gold_label)][int(pred_label)] += 1
        correct_count += int(pred_label == gold_label)

    sample_count = len(gold_labels)
    return {
        "sample_count": sample_count,
        "accuracy": float(correct_count / sample_count),
        "correct_count": correct_count,
        "confusion_matrix": confusion,
    }


def evaluate_replay_system(
    system_name: str,
    examples: Sequence[Dict[str, Any]],
    requested_k_resolver,
    total_input_sample_count: int,
) -> Dict[str, Any]:
    k_sum = 0
    retrieval_count = 0
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
            retrieval_count += int(proposed_k > 0)

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

    avg_em = float(em_sum / evaluated_sample_count) if evaluated_sample_count else None
    avg_f1 = float(f1_sum / evaluated_sample_count) if evaluated_sample_count else None
    avg_topk = float(k_sum / k_determined_sample_count) if k_determined_sample_count else None
    retrieval_rate = float(retrieval_count / k_determined_sample_count) if k_determined_sample_count else None
    retrieval_rate_total_input = float(retrieval_count / total_input_sample_count) if total_input_sample_count else None

    return {
        "system_name": system_name,
        "input_sample_count": total_input_sample_count,
        "k_determined_sample_count": k_determined_sample_count,
        "aligned_sample_count": aligned_sample_count,
        "evaluated_sample_count": evaluated_sample_count,
        "skipped_missing_alignment_count": missing_alignment_count,
        "skipped_missing_score_count": missing_score_count,
        "retrieval_count": retrieval_count,
        "avg_em": avg_em,
        "avg_f1": avg_f1,
        "avg_topk": avg_topk,
        "retrieval_rate": retrieval_rate,
        "retrieval_rate_total_input": retrieval_rate_total_input,
    }


def write_summary_csv(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system_name",
        "avg_em",
        "avg_f1",
        "avg_topk",
        "retrieval_rate",
        "retrieval_rate_total_input",
        "stage1_acc",
        "stage2_acc",
        "stage2_acc_type",
        "input_sample_count",
        "k_determined_sample_count",
        "aligned_sample_count",
        "evaluated_sample_count",
        "skipped_missing_alignment_count",
        "skipped_missing_score_count",
        "retrieval_count",
    ]

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
        ("retrieval_rate", "retrieval_rate"),
        ("evaluated_sample_count", "eval_n"),
        ("skipped_missing_alignment_count", "missing_align"),
        ("skipped_missing_score_count", "missing_score"),
    ]

    widths: List[int] = []
    formatted_rows: List[List[str]] = []
    for row in rows:
        formatted_row = []
        for field_name, _ in columns:
            value = row.get(field_name)
            if field_name.startswith("avg_") or field_name == "retrieval_rate":
                formatted = float_to_display(value)
            else:
                formatted = str(value)
            formatted_row.append(formatted)
        formatted_rows.append(formatted_row)

    for column_index, (_, header) in enumerate(columns):
        content_width = max([len(header)] + [len(row[column_index]) for row in formatted_rows]) if formatted_rows else len(header)
        widths.append(content_width)

    header_line = " | ".join(header.ljust(widths[index]) for index, (_, header) in enumerate(columns))
    separator_line = "-+-".join("-" * widths[index] for index in range(len(columns)))
    print(header_line)
    print(separator_line)
    for formatted_row in formatted_rows:
        print(" | ".join(formatted_row[index].ljust(widths[index]) for index in range(len(columns))))


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)

    fixed_ks = parse_fixed_ks(args.fixed_ks)
    stage1_batch_size = args.stage1_batch_size or args.batch_size
    stage2_batch_size = args.stage2_batch_size or args.batch_size
    device = choose_device(args.device)

    test_file = resolve_input_path(args.test_file)
    replay_label_file = resolve_input_path(args.replay_label_file)
    stage1_model_dir = str(resolve_input_path(args.stage1_model_dir))
    stage2_model_dir = str(resolve_input_path(args.stage2_model_dir))

    test_examples, test_metadata, raw_label_distribution = load_test_examples(test_file)
    replay_records, replay_metadata, replay_load_stats = load_replay_label_records(replay_label_file)
    replay_by_id, replay_by_question, replay_index_stats = build_replay_indices(replay_records)
    alignment_stats = align_examples_with_replay(test_examples, replay_by_id, replay_by_question)

    dataset_name = infer_dataset_name(args.dataset_name, test_examples, test_file)
    split_name = infer_split_name(args.split_name, test_file)
    stage1_model_tag = infer_model_tag(None, stage1_model_dir)
    stage2_model_tag = infer_model_tag(None, stage2_model_dir)

    resolved_output_root = resolve_output_path(args.output_dir)
    safe_dataset_name = sanitize_artifact_component(dataset_name, fallback="dataset")
    default_run_name = f"{split_name}__stage1_{stage1_model_tag}__stage2_{stage2_model_tag}"
    run_name = sanitize_artifact_component(args.run_name, fallback=default_run_name)
    run_output_dir = resolved_output_root / safe_dataset_name / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    from kbqa_classifier.model.stage1_bert_classifier import (
        STAGE1_ID2LABEL,
        STAGE1_LABEL2ID,
        build_stage1_model,
        load_stage1_tokenizer,
    )
    from kbqa_classifier.model.stage2_bert_classifier import (
        build_stage2_model,
        load_stage2_tokenizer,
    )
    from kbqa_classifier.train.train_stage1 import load_config_label_mapping

    stage1_id2label, stage1_label2id = load_config_label_mapping(stage1_model_dir, STAGE1_ID2LABEL, STAGE1_LABEL2ID)
    stage1_positive_label_id = stage1_label2id.get("needs_retrieval", 1)
    stage1_model = build_stage1_model(stage1_model_dir)
    stage1_tokenizer = load_stage1_tokenizer(stage1_model_dir)

    stage1_logits, stage1_probabilities, stage1_pred_labels = predict_logits_on_device(
        texts=[example["question"] for example in test_examples],
        model=stage1_model,
        tokenizer=stage1_tokenizer,
        max_length=args.stage1_max_length,
        batch_size=stage1_batch_size,
        device=device,
    )

    for example, logit_row, probability_row, pred_label in zip(
        test_examples,
        stage1_logits,
        stage1_probabilities,
        stage1_pred_labels,
    ):
        pred_label = int(pred_label)
        example["stage1_pred"] = pred_label
        example["stage1_pred_name"] = stage1_id2label.get(pred_label, str(pred_label))
        example["stage1_needs_retrieval"] = pred_label == stage1_positive_label_id
        example["stage1_logits"] = [float(value) for value in logit_row.tolist()]
        example["stage1_probabilities"] = [float(value) for value in probability_row.tolist()]

    stage2_id2label, _, raw_bucket_definitions = load_stage2_label_metadata(stage2_model_dir)
    bucket_definitions = normalize_bucket_definitions(raw_bucket_definitions)
    bucket_to_topk_mapping = {
        str(bucket_id): map_bucket_to_replay_k(bucket_id, bucket_definitions, args.bucket_to_topk_strategy)
        for bucket_id in sorted(bucket_definitions)
    }

    global_supported_ks = sorted(
        {
            int(k_value)
            for replay_record in replay_records
            for k_value in replay_record["scores"].keys()
        }
    )
    if not global_supported_ks:
        raise ValueError("Replay label file contains no supported k values.")

    min_supported_k = global_supported_ks[0]
    max_supported_k = global_supported_ks[-1]

    stage2_candidate_examples = [example for example in test_examples if example["stage1_needs_retrieval"]]
    if stage2_candidate_examples:
        stage2_model = build_stage2_model(stage2_model_dir)
        stage2_tokenizer = load_stage2_tokenizer(stage2_model_dir)
        stage2_logits, stage2_probabilities, stage2_pred_labels = predict_logits_on_device(
            texts=[example["question"] for example in stage2_candidate_examples],
            model=stage2_model,
            tokenizer=stage2_tokenizer,
            max_length=args.stage2_max_length,
            batch_size=stage2_batch_size,
            device=device,
        )
    else:
        stage2_logits = np.zeros((0, 0), dtype=np.float32)
        stage2_probabilities = np.zeros((0, 0), dtype=np.float32)
        stage2_pred_labels = []

    clipped_final_k_count = 0
    for example in test_examples:
        example["stage2_executed"] = False
        example["stage2_pred_raw"] = None
        example["stage2_pred_name"] = None
        example["stage2_logits"] = None
        example["stage2_probabilities"] = None
        example["stage2_gt"] = None
        example["stage2_gt_name"] = None
        example["stage2_gt_type"] = "bucket"

        if example.get("raw_topk_label") is not None:
            stage2_gt_bucket = bucket_id_from_topk(int(example["raw_topk_label"]), bucket_definitions)
            if stage2_gt_bucket is not None:
                example["stage2_gt"] = int(stage2_gt_bucket)
                example["stage2_gt_name"] = stage2_id2label.get(int(stage2_gt_bucket), str(stage2_gt_bucket))

    for example, logit_row, probability_row, pred_bucket_id in zip(
        stage2_candidate_examples,
        stage2_logits,
        stage2_probabilities,
        stage2_pred_labels,
    ):
        pred_bucket_id = int(pred_bucket_id)
        example["stage2_executed"] = True
        example["stage2_pred_raw"] = pred_bucket_id
        example["stage2_pred_name"] = stage2_id2label.get(pred_bucket_id, str(pred_bucket_id))
        example["stage2_logits"] = [float(value) for value in logit_row.tolist()]
        example["stage2_probabilities"] = [float(value) for value in probability_row.tolist()]

    for example in test_examples:
        if not example["stage1_needs_retrieval"]:
            final_k_before_clip = 0
        else:
            predicted_bucket = example.get("stage2_pred_raw")
            if predicted_bucket is None:
                raise ValueError("Stage-2 prediction is missing for a sample that was routed to stage 2.")
            final_k_before_clip = map_bucket_to_replay_k(
                predicted_bucket,
                bucket_definitions=bucket_definitions,
                strategy=args.bucket_to_topk_strategy,
            )

        final_k_hat, was_clipped = clip_k_to_supported_range(
            proposed_k=int(final_k_before_clip),
            min_supported_k=min_supported_k,
            max_supported_k=max_supported_k,
        )
        clipped_final_k_count += int(was_clipped)
        example["final_k_before_clip"] = int(final_k_before_clip)
        example["final_k_hat"] = int(final_k_hat)
        example["final_k_was_clipped"] = bool(was_clipped)

        replay_record = example.get("aligned_replay")
        example["oracle_best_k"] = int(replay_record["best_k"]) if replay_record is not None else None
        example["available_score_ks"] = sorted(replay_record["scores"].keys()) if replay_record is not None else []

        replay_score = None
        if replay_record is not None:
            replay_score = replay_record["scores"].get(int(final_k_hat))

        example["replay_score_available"] = replay_score is not None
        example["replay_em"] = float(replay_score["em"]) if replay_score is not None else None
        example["replay_f1"] = float(replay_score["f1"]) if replay_score is not None else None

    stage1_gold_labels = [int(example["stage1_gt"]) for example in test_examples if example.get("stage1_gt") is not None]
    stage1_pred_labels_for_acc = [
        int(example["stage1_pred"]) for example in test_examples if example.get("stage1_gt") is not None
    ]
    stage1_metrics = compute_binary_accuracy(stage1_pred_labels_for_acc, stage1_gold_labels)

    stage2_gt_positive_count = 0
    stage2_executed_on_gt_positive_count = 0
    stage2_correct_count = 0
    for example in test_examples:
        if example.get("stage1_gt") != 1:
            continue
        if example.get("stage2_gt") is None:
            continue
        stage2_gt_positive_count += 1
        if not example.get("stage2_executed"):
            continue
        stage2_executed_on_gt_positive_count += 1
        stage2_correct_count += int(example.get("stage2_pred_raw") == example.get("stage2_gt"))

    stage2_acc = (
        float(stage2_correct_count / stage2_executed_on_gt_positive_count)
        if stage2_executed_on_gt_positive_count
        else None
    )
    stage2_coverage_on_gt_positive = (
        float(stage2_executed_on_gt_positive_count / stage2_gt_positive_count) if stage2_gt_positive_count else None
    )

    total_input_sample_count = len(test_examples)
    system_rows: List[Dict[str, Any]] = []

    for fixed_k in fixed_ks:
        system_name = f"fixed_k_{fixed_k}"
        system_rows.append(
            evaluate_replay_system(
                system_name=system_name,
                examples=test_examples,
                requested_k_resolver=lambda _example, chosen_k=fixed_k: chosen_k,
                total_input_sample_count=total_input_sample_count,
            )
        )

    two_stage_row = evaluate_replay_system(
        system_name="two_stage_dynamic_topk",
        examples=test_examples,
        requested_k_resolver=lambda example: example.get("final_k_hat"),
        total_input_sample_count=total_input_sample_count,
    )
    two_stage_row.update(
        {
            "stage1_acc": stage1_metrics["accuracy"],
            "stage2_acc": stage2_acc,
            "stage2_acc_type": "bucket_accuracy",
            "stage2_gt_positive_count": stage2_gt_positive_count,
            "stage2_executed_on_gt_positive_count": stage2_executed_on_gt_positive_count,
            "stage2_coverage_on_gt_positive": stage2_coverage_on_gt_positive,
        }
    )
    system_rows.append(two_stage_row)

    oracle_row = evaluate_replay_system(
        system_name="oracle_best_k",
        examples=test_examples,
        requested_k_resolver=lambda example: example.get("oracle_best_k"),
        total_input_sample_count=total_input_sample_count,
    )
    system_rows.append(oracle_row)

    for row in system_rows:
        row.setdefault("stage1_acc", None)
        row.setdefault("stage2_acc", None)
        row.setdefault("stage2_acc_type", None)

    summary_json_path = run_output_dir / "summary.json"
    summary_csv_path = run_output_dir / "summary.csv"
    per_example_path = run_output_dir / "per_example_predictions.jsonl"
    config_json_path = run_output_dir / "config.json"

    per_example_rows: List[Dict[str, Any]] = []
    for example in test_examples:
        replay_record = example.get("aligned_replay")
        resolved_dataset_name = example["dataset_name"]
        if resolved_dataset_name is None and replay_record is not None:
            resolved_dataset_name = replay_record.get("dataset_name")

        row = {
            "example_index": example["example_index"],
            "id": example["id"],
            "question": example["question"],
            "dataset_name": resolved_dataset_name,
            "aligned": bool(example.get("aligned")),
            "match_status": "matched" if example.get("aligned") else "missing_alignment",
            "matched_by": example.get("matched_by"),
            "replay_record_id": replay_record.get("id") if replay_record is not None else None,
            "replay_record_dataset_name": replay_record.get("dataset_name") if replay_record is not None else None,
            "raw_topk_label": example.get("raw_topk_label"),
            "stage1_gt": example.get("stage1_gt"),
            "stage1_pred": example.get("stage1_pred"),
            "stage1_pred_name": example.get("stage1_pred_name"),
            "stage2_executed": example.get("stage2_executed"),
            "stage2_gt": example.get("stage2_gt"),
            "stage2_gt_name": example.get("stage2_gt_name"),
            "stage2_pred_raw": example.get("stage2_pred_raw"),
            "stage2_pred_name": example.get("stage2_pred_name"),
            "stage2_pred_type": "bucket",
            "final_k_before_clip": example.get("final_k_before_clip"),
            "final_k_hat": example.get("final_k_hat"),
            "final_k_was_clipped": example.get("final_k_was_clipped"),
            "oracle_best_k": example.get("oracle_best_k"),
            "available_score_ks": example.get("available_score_ks"),
            "replay_score_available": example.get("replay_score_available"),
            "replay_em": example.get("replay_em"),
            "replay_f1": example.get("replay_f1"),
        }

        if args.do_save_stage_probabilities:
            row["stage1_logits"] = example.get("stage1_logits")
            row["stage1_probabilities"] = example.get("stage1_probabilities")
            row["stage2_logits"] = example.get("stage2_logits")
            row["stage2_probabilities"] = example.get("stage2_probabilities")

        per_example_rows.append(row)

    summary_payload = {
        "task": "end2end_replay_eval",
        "dataset_name": dataset_name,
        "split_name": split_name,
        "run_name": run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "stage1_model_dir": stage1_model_dir,
            "stage2_model_dir": stage2_model_dir,
            "test_file": str(test_file),
            "replay_label_file": str(replay_label_file),
            "output_dir": str(run_output_dir),
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "per_example_predictions": str(per_example_path),
            "config_json": str(config_json_path),
        },
        "test_file_metadata": {
            "question_field": test_metadata["question_field"],
            "label_field": test_metadata["label_field"],
            "id_field": test_metadata["id_field"],
            "dataset_field": test_metadata["dataset_field"],
            "raw_label_distribution": {str(k): v for k, v in sorted(raw_label_distribution.items())},
        },
        "replay_label_metadata": {
            "question_field": replay_metadata["question_field"],
            "id_field": replay_metadata["id_field"],
            "dataset_field": replay_metadata["dataset_field"],
            "best_k_field": replay_metadata["best_k_field"],
            "scores_field": replay_metadata["scores_field"],
            "computed_best_k_count": replay_load_stats["computed_best_k_count"],
            "explicit_best_k_count": replay_load_stats["explicit_best_k_count"],
            "records_without_id_count": replay_load_stats["records_without_id_count"],
            "records_without_question_count": replay_load_stats["records_without_question_count"],
        },
        "data_stats": {
            "input_sample_count": total_input_sample_count,
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
            "supported_k_values": global_supported_ks,
            "supported_k_min": min_supported_k,
            "supported_k_max": max_supported_k,
            "stage2_bucket_to_topk_strategy": args.bucket_to_topk_strategy,
            "stage2_bucket_to_topk_mapping": bucket_to_topk_mapping,
            "stage2_bucket_definitions": {str(k): v for k, v in sorted(bucket_definitions.items())},
            "clipped_final_k_count": clipped_final_k_count,
        },
        "stage_metrics": {
            "stage1_acc": stage1_metrics["accuracy"],
            "stage1_sample_count": stage1_metrics["sample_count"],
            "stage1_correct_count": stage1_metrics["correct_count"],
            "stage1_confusion_matrix": stage1_metrics["confusion_matrix"],
            "stage1_positive_label_id": stage1_positive_label_id,
            "stage2_acc": stage2_acc,
            "stage2_acc_type": "bucket_accuracy",
            "stage2_gt_positive_count": stage2_gt_positive_count,
            "stage2_executed_on_gt_positive_count": stage2_executed_on_gt_positive_count,
            "stage2_correct_count": stage2_correct_count,
            "stage2_coverage_on_gt_positive": stage2_coverage_on_gt_positive,
            "stage2_denominator_note": (
                "stage2_acc is bucket accuracy on samples where stage1_gt=1 and stage2 actually executed "
                "(that is, stage1_pred also routed the sample to stage 2)."
            ),
        },
        "warning_counts": {
            "missing_alignment_count": alignment_stats["missing_alignment_count"],
            "clipped_final_k_count": clipped_final_k_count,
            "missing_score_count_by_system": {
                row["system_name"]: row["skipped_missing_score_count"] for row in system_rows
            },
        },
        "system_results": system_rows,
    }

    config_payload = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "run_name": run_name,
        "seed": args.seed,
        "stage1_model_dir": stage1_model_dir,
        "stage2_model_dir": stage2_model_dir,
        "test_file": str(test_file),
        "replay_label_file": str(replay_label_file),
        "output_dir": str(run_output_dir),
        "fixed_k_values": fixed_ks,
        "batch_size": args.batch_size,
        "stage1_batch_size": stage1_batch_size,
        "stage2_batch_size": stage2_batch_size,
        "stage1_max_length": args.stage1_max_length,
        "stage2_max_length": args.stage2_max_length,
        "device": str(device),
        "bucket_to_topk_strategy": args.bucket_to_topk_strategy,
        "bucket_to_topk_mapping": bucket_to_topk_mapping,
        "supported_k_values": global_supported_ks,
        "do_save_stage_probabilities": args.do_save_stage_probabilities,
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
    if alignment_stats["missing_alignment_count"]:
        print(f"Warning: {alignment_stats['missing_alignment_count']} samples could not be aligned to replay labels.")
    print(
        f"Two-stage retrieval rate: {float_to_display(two_stage_row['retrieval_rate'])} | "
        f"Avg EM: {float_to_display(two_stage_row['avg_em'])} | "
        f"Avg F1: {float_to_display(two_stage_row['avg_f1'])} | "
        f"Avg Top-k: {float_to_display(two_stage_row['avg_topk'])}"
    )
    print(
        f"Stage-1 Acc: {float_to_display(stage1_metrics['accuracy'])} | "
        f"Stage-2 Acc ({two_stage_row['stage2_acc_type']}): {float_to_display(stage2_acc)}"
    )
    print(f"Clipped final k count: {clipped_final_k_count}")
    for row in system_rows:
        if row["skipped_missing_score_count"]:
            print(
                f"Warning: {row['system_name']} skipped {row['skipped_missing_score_count']} aligned samples "
                f"because the requested k score was missing."
            )
    print(f"Summary JSON saved to: {summary_json_path}")
    print(f"Summary CSV saved to: {summary_csv_path}")
    print(f"Per-example predictions saved to: {per_example_path}")
    print("")
    print_comparison_table(system_rows)


if __name__ == "__main__":
    main()
