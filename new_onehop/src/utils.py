"""Shared configuration, reproducibility, and serialization helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
import yaml

from .losses import (
    build_ordinal_targets,
    ordinal_bce_loss,
    ordinal_cumulative_to_class_probs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """Raised when an experiment configuration is missing or invalid."""


def project_path(value: str | Path) -> Path:
    """Resolve project-relative paths without depending on the current directory."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = project_path(path)
    try:
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing config file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError(f"Config must contain a YAML object: {config_path}")
    return config


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Config field {key!r} must be a mapping")
    return value


def set_random_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"seed must be an integer, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # These settings favor repeatability. Some CUDA kernels may still be
    # nondeterministic depending on the installed PyTorch/CUDA combination.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json_atomic(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(output_path)


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(output_path)


def write_yaml_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(output_path)


def torch_save_atomic(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class MulticlassPredictionOutput:
    loss: float
    rows: list[dict[str, Any]]
    question_ids: list[str]
    dataset_names: list[str]
    predicted_ks: list[int]
    oracle_ks: list[int]


@dataclass(frozen=True)
class OrdinalPredictionOutput:
    loss: float
    rows: list[dict[str, Any]]
    question_ids: list[str]
    dataset_names: list[str]
    predicted_ks: list[int]
    oracle_ks: list[int]
    thresholds: list[float]
    numerical_sanity: dict[str, int | float]


@torch.no_grad()
def predict_multiclass(
    model: nn.Module,
    data_loader: Iterable[dict[str, Any]],
    device: torch.device,
    mixed_precision: bool,
) -> MulticlassPredictionOutput:
    """Run labeled inference once and retain probabilities in strict k=1..15 order."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    rows: list[dict[str, Any]] = []
    question_ids: list[str] = []
    dataset_names: list[str] = []
    predicted_ks: list[int] = []
    oracle_ks: list[int] = []
    use_amp = mixed_precision and device.type == "cuda"

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss: {loss.item()}")
        probabilities = torch.softmax(logits.float(), dim=-1).cpu()
        predictions = torch.argmax(probabilities, dim=-1) + 1
        hard_ks = labels.cpu() + 1
        batch_size = labels.shape[0]
        if logits.ndim != 2 or logits.shape != (batch_size, 15):
            raise RuntimeError(
                f"Expected logits shape ({batch_size}, 15), got {tuple(logits.shape)}"
            )
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        for index in range(batch_size):
            question_id = batch["question_ids"][index]
            dataset = batch["datasets"][index]
            predicted_k = int(predictions[index].item())
            hard_oracle_k = int(hard_ks[index].item())
            probability_list = [float(value) for value in probabilities[index].tolist()]
            rows.append(
                {
                    "question_id": question_id,
                    "dataset": dataset,
                    "hard_oracle_k": hard_oracle_k,
                    "predicted_k": predicted_k,
                    "probabilities": probability_list,
                }
            )
            question_ids.append(question_id)
            dataset_names.append(dataset)
            predicted_ks.append(predicted_k)
            oracle_ks.append(hard_oracle_k)

    if total_examples == 0:
        raise RuntimeError("Validation data loader produced no examples")
    return MulticlassPredictionOutput(
        loss=total_loss / total_examples,
        rows=rows,
        question_ids=question_ids,
        dataset_names=dataset_names,
        predicted_ks=predicted_ks,
        oracle_ks=oracle_ks,
    )


@torch.no_grad()
def predict_ordinal(
    model: nn.Module,
    data_loader: Iterable[dict[str, Any]],
    device: torch.device,
    mixed_precision: bool,
    num_thresholds: int = 14,
) -> OrdinalPredictionOutput:
    """Run labeled ordinal inference with strict numerical validation."""
    if (
        isinstance(num_thresholds, bool)
        or not isinstance(num_thresholds, int)
        or num_thresholds <= 0
    ):
        raise ValueError("num_thresholds must be a positive integer")
    num_labels = num_thresholds + 1
    model.eval()
    total_loss = 0.0
    total_examples = 0
    rows: list[dict[str, Any]] = []
    question_ids: list[str] = []
    dataset_names: list[str] = []
    predicted_ks: list[int] = []
    oracle_ks: list[int] = []
    threshold_values: list[float] = []
    threshold_order_violations = 0
    cumulative_monotonicity_violations = 0
    negative_class_probability_rows = 0
    max_probability_sum_error = 0.0
    use_amp = mixed_precision and device.type == "cuda"

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        batch_size = labels.shape[0]
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            ordinal_logits = output["ordinal_logits"]
            ordinal_targets = build_ordinal_targets(labels, num_thresholds)
            if ordinal_logits.shape != (batch_size, num_thresholds):
                raise RuntimeError(
                    f"Expected ordinal_logits shape ({batch_size}, {num_thresholds}), "
                    f"got {tuple(ordinal_logits.shape)}"
                )
            if ordinal_targets.shape != (batch_size, num_thresholds):
                raise RuntimeError(
                    f"Expected ordinal targets shape ({batch_size}, {num_thresholds}), "
                    f"got {tuple(ordinal_targets.shape)}"
                )
            loss = ordinal_bce_loss(ordinal_logits, ordinal_targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss: {loss.item()}")

        thresholds = output["thresholds"].float()
        if thresholds.ndim != 1 or thresholds.shape[0] != num_thresholds:
            raise RuntimeError(
                f"Expected {num_thresholds} thresholds, got {tuple(thresholds.shape)}"
            )
        if not torch.isfinite(thresholds).all():
            raise RuntimeError("Thresholds contain NaN or Inf")
        batch_threshold_violations = int(
            torch.count_nonzero(thresholds[1:] <= thresholds[:-1]).item()
        )
        threshold_order_violations += batch_threshold_violations
        if batch_threshold_violations:
            raise RuntimeError(
                f"Thresholds are not strictly increasing: {thresholds.tolist()}"
            )
        if not threshold_values:
            threshold_values = [float(value) for value in thresholds.cpu().tolist()]

        cumulative_probabilities = torch.sigmoid(ordinal_logits.float())
        batch_cumulative_violations = int(
            torch.count_nonzero(
                torch.any(
                    cumulative_probabilities[:, 1:]
                    > cumulative_probabilities[:, :-1] + 1e-6,
                    dim=1,
                )
            ).item()
        )
        cumulative_monotonicity_violations += batch_cumulative_violations
        if batch_cumulative_violations:
            raise RuntimeError(
                "Cumulative ordinal probabilities are not monotonically non-increasing"
            )
        raw_class_probabilities = torch.cat(
            (
                1.0 - cumulative_probabilities[:, :1],
                cumulative_probabilities[:, :-1] - cumulative_probabilities[:, 1:],
                cumulative_probabilities[:, -1:],
            ),
            dim=1,
        )
        batch_negative_rows = int(
            torch.count_nonzero(
                torch.any(raw_class_probabilities < -1e-6, dim=1)
            ).item()
        )
        negative_class_probability_rows += batch_negative_rows
        if batch_negative_rows:
            raise RuntimeError(
                "Ordinal conversion produced significantly negative class probabilities"
            )
        batch_sum_error = float(
            torch.max(torch.abs(raw_class_probabilities.sum(dim=1) - 1.0)).item()
        )
        max_probability_sum_error = max(max_probability_sum_error, batch_sum_error)
        class_probabilities = ordinal_cumulative_to_class_probs(
            cumulative_probabilities
        )
        if class_probabilities.shape != (batch_size, num_labels):
            raise RuntimeError(
                f"Expected class probabilities shape ({batch_size}, {num_labels}), "
                f"got {tuple(class_probabilities.shape)}"
            )
        predictions = torch.argmax(class_probabilities, dim=-1) + 1
        if torch.any((predictions < 1) | (predictions > num_labels)):
            raise RuntimeError("Decoded predicted_k falls outside the valid range")
        hard_ks = labels.cpu() + 1
        probabilities_cpu = class_probabilities.cpu()
        cumulative_cpu = cumulative_probabilities.cpu()
        predictions_cpu = predictions.cpu()
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        for index in range(batch_size):
            question_id = batch["question_ids"][index]
            dataset = batch["datasets"][index]
            predicted_k = int(predictions_cpu[index].item())
            hard_oracle_k = int(hard_ks[index].item())
            rows.append(
                {
                    "question_id": question_id,
                    "dataset": dataset,
                    "hard_oracle_k": hard_oracle_k,
                    "predicted_k": predicted_k,
                    "probabilities": [
                        float(value) for value in probabilities_cpu[index].tolist()
                    ],
                    "ordinal_cumulative_probabilities": [
                        float(value) for value in cumulative_cpu[index].tolist()
                    ],
                }
            )
            question_ids.append(question_id)
            dataset_names.append(dataset)
            predicted_ks.append(predicted_k)
            oracle_ks.append(hard_oracle_k)

    if total_examples == 0:
        raise RuntimeError("Validation data loader produced no examples")
    return OrdinalPredictionOutput(
        loss=total_loss / total_examples,
        rows=rows,
        question_ids=question_ids,
        dataset_names=dataset_names,
        predicted_ks=predicted_ks,
        oracle_ks=oracle_ks,
        thresholds=threshold_values,
        numerical_sanity={
            "threshold_count": len(threshold_values),
            "num_threshold_order_violations": threshold_order_violations,
            "num_cumulative_monotonicity_violations": (
                cumulative_monotonicity_violations
            ),
            "num_negative_class_probability_rows": (negative_class_probability_rows),
            "max_probability_sum_error": max_probability_sum_error,
        },
    )
