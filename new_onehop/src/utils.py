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
