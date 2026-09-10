"""Predictor-level and RAG replay metrics shared by train/eval scripts."""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

from .dataset import CurveRecord, DATASETS, NUM_LABELS


def _validate_k_sequence(values: Sequence[int], name: str) -> None:
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= NUM_LABELS
        ):
            raise ValueError(
                f"{name}[{index}] must be an integer in [1, {NUM_LABELS}], got {value!r}"
            )


def _macro_f1(predicted_ks: Sequence[int], oracle_ks: Sequence[int]) -> float:
    class_f1: list[float] = []
    for label in range(1, NUM_LABELS + 1):
        true_positive = sum(
            predicted == label and oracle == label
            for predicted, oracle in zip(predicted_ks, oracle_ks)
        )
        false_positive = sum(
            predicted == label and oracle != label
            for predicted, oracle in zip(predicted_ks, oracle_ks)
        )
        false_negative = sum(
            predicted != label and oracle == label
            for predicted, oracle in zip(predicted_ks, oracle_ks)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return statistics.fmean(class_f1)


def compute_predictor_metrics(
    predicted_ks: Sequence[int],
    oracle_ks: Sequence[int],
) -> dict[str, float]:
    if not predicted_ks or len(predicted_ks) != len(oracle_ks):
        raise ValueError(
            "Predictions and Oracle labels must be non-empty and have equal length"
        )
    _validate_k_sequence(predicted_ks, "predicted_ks")
    _validate_k_sequence(oracle_ks, "oracle_ks")
    total = len(predicted_ks)
    distances = [
        abs(predicted - oracle) for predicted, oracle in zip(predicted_ks, oracle_ks)
    ]
    squared_distances = [distance * distance for distance in distances]
    return {
        "num_examples": total,
        "accuracy": sum(distance == 0 for distance in distances) / total,
        "macro_f1": _macro_f1(predicted_ks, oracle_ks),
        "mae": statistics.fmean(distances),
        "rmse": math.sqrt(statistics.fmean(squared_distances)),
        "within_1_accuracy": sum(distance <= 1 for distance in distances) / total,
        "within_2_accuracy": sum(distance <= 2 for distance in distances) / total,
        "under_retrieval_rate": sum(
            predicted < oracle for predicted, oracle in zip(predicted_ks, oracle_ks)
        )
        / total,
        "over_retrieval_rate": sum(
            predicted > oracle for predicted, oracle in zip(predicted_ks, oracle_ks)
        )
        / total,
        "avg_predicted_k": statistics.fmean(predicted_ks),
        "avg_oracle_k": statistics.fmean(oracle_ks),
    }


def compute_grouped_predictor_metrics(
    predicted_ks: Sequence[int],
    oracle_ks: Sequence[int],
    dataset_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    if len(predicted_ks) != len(dataset_names) or len(oracle_ks) != len(dataset_names):
        raise ValueError(
            "Predictions, Oracle labels, and dataset names must have equal length"
        )
    unexpected = sorted(set(dataset_names) - set(DATASETS))
    if unexpected:
        raise ValueError(f"Unexpected dataset names: {unexpected}")
    grouped = {"overall": compute_predictor_metrics(predicted_ks, oracle_ks)}
    for dataset in DATASETS:
        indices = [
            index for index, value in enumerate(dataset_names) if value == dataset
        ]
        if not indices:
            raise ValueError(f"No examples found for dataset {dataset!r}")
        grouped[dataset] = compute_predictor_metrics(
            [predicted_ks[index] for index in indices],
            [oracle_ks[index] for index in indices],
        )
    return grouped


def compute_always_k1_predictor_baseline(
    oracle_ks: Sequence[int],
    dataset_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    return compute_grouped_predictor_metrics(
        [1] * len(oracle_ks), oracle_ks, dataset_names
    )


def _summarize_rag_rows(
    rows: Sequence[tuple[str, float, int, int, bool]],
) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("Cannot summarize an empty RAG replay group")
    validated_tokens = [tokens for _, _, tokens, _, validated in rows if validated]
    return {
        "num_examples": len(rows),
        "avg_f1": statistics.fmean(f1 for _, f1, _, _, _ in rows),
        "avg_context_tokens": statistics.fmean(tokens for _, _, tokens, _, _ in rows),
        "avg_k": statistics.fmean(k for _, _, _, k, _ in rows),
        "avg_context_tokens_validated_only": (
            statistics.fmean(validated_tokens) if validated_tokens else None
        ),
        "num_queries_with_unvalidated_tokens": sum(
            not validated for _, _, _, _, validated in rows
        ),
    }


def _group_rag_rows(
    rows: Sequence[tuple[str, float, int, int, bool]],
) -> dict[str, dict[str, float | int | None]]:
    grouped = {"overall": _summarize_rag_rows(rows)}
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row[0] == dataset]
        if not dataset_rows:
            raise ValueError(f"No RAG replay rows found for dataset {dataset!r}")
        grouped[dataset] = _summarize_rag_rows(dataset_rows)
    return grouped


def compute_dynamic_rag_metrics(
    predicted_k_by_id: Mapping[str, int],
    curves_by_id: Mapping[str, CurveRecord],
    ordered_ids: Sequence[str],
) -> dict[str, dict[str, float | int | None]]:
    if set(predicted_k_by_id) != set(ordered_ids):
        missing = sorted(set(ordered_ids) - set(predicted_k_by_id))
        extra = sorted(set(predicted_k_by_id) - set(ordered_ids))
        raise ValueError(
            f"Prediction ID mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    rows: list[tuple[str, float, int, int, bool]] = []
    for question_id in ordered_ids:
        if question_id not in curves_by_id:
            raise ValueError(f"Curve missing for prediction ID {question_id!r}")
        predicted_k = predicted_k_by_id[question_id]
        _validate_k_sequence([predicted_k], f"prediction[{question_id}]")
        curve = curves_by_id[question_id]
        result = curve.results[predicted_k]
        rows.append(
            (
                curve.dataset,
                result.f1,
                result.context_tokens,
                predicted_k,
                result.context_tokens_validated,
            )
        )
    return _group_rag_rows(rows)


def compute_fixed_k_rag_metrics(
    curves_by_id: Mapping[str, CurveRecord],
    ordered_ids: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    baselines: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for k in range(1, NUM_LABELS + 1):
        rows: list[tuple[str, float, int, int, bool]] = []
        for question_id in ordered_ids:
            if question_id not in curves_by_id:
                raise ValueError(f"Curve missing for split ID {question_id!r}")
            curve = curves_by_id[question_id]
            result = curve.results[k]
            rows.append(
                (
                    curve.dataset,
                    result.f1,
                    result.context_tokens,
                    k,
                    result.context_tokens_validated,
                )
            )
        baselines[f"fixed_k_{k}"] = _group_rag_rows(rows)
    return baselines


def compute_global_max_context_tokens(
    curves_by_id: Mapping[str, CurveRecord],
) -> int:
    """Return T_max over every question and k in the complete train-pool curves."""
    if not curves_by_id:
        raise ValueError("Cannot compute T_max from empty curves")
    maximum = 0
    for question_id, curve in curves_by_id.items():
        if set(curve.results) != set(range(1, NUM_LABELS + 1)):
            raise ValueError(
                f"Curve {question_id!r} must contain exactly k=1..{NUM_LABELS}"
            )
        maximum = max(
            maximum,
            *(result.context_tokens for result in curve.results.values()),
        )
    if maximum <= 0:
        raise ValueError(f"T_max must be positive, got {maximum}")
    return maximum


def compute_dynamic_rag_utility(
    predicted_k_by_id: Mapping[str, int],
    curves_by_id: Mapping[str, CurveRecord],
    ordered_ids: Sequence[str],
    *,
    lambda_T: float,
    t_max: int,
) -> float:
    """Mean validation utility F1(k_hat) - lambda_T * T(k_hat) / T_max."""
    if (
        isinstance(lambda_T, bool)
        or not isinstance(lambda_T, (int, float))
        or not math.isfinite(float(lambda_T))
        or lambda_T < 0.0
    ):
        raise ValueError(
            f"lambda_T must be a finite non-negative number, got {lambda_T!r}"
        )
    if isinstance(t_max, bool) or not isinstance(t_max, int) or t_max <= 0:
        raise ValueError(f"t_max must be a positive integer, got {t_max!r}")
    if not ordered_ids:
        raise ValueError("ordered_ids must be non-empty")
    if set(predicted_k_by_id) != set(ordered_ids):
        missing = sorted(set(ordered_ids) - set(predicted_k_by_id))
        extra = sorted(set(predicted_k_by_id) - set(ordered_ids))
        raise ValueError(
            f"Prediction ID mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    utilities: list[float] = []
    for question_id in ordered_ids:
        if question_id not in curves_by_id:
            raise ValueError(f"Curve missing for prediction ID {question_id!r}")
        predicted_k = predicted_k_by_id[question_id]
        _validate_k_sequence([predicted_k], f"prediction[{question_id}]")
        result = curves_by_id[question_id].results[predicted_k]
        utilities.append(result.f1 - float(lambda_T) * result.context_tokens / t_max)
    utility = statistics.fmean(utilities)
    if not math.isfinite(utility):
        raise RuntimeError(f"Computed non-finite RAG utility: {utility}")
    return utility
