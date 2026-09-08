#!/usr/bin/env python3
"""Train the Phase 1 question-only 15-way multiclass CE baseline."""

# ruff: noqa: E402 -- project root must be added before local package imports.

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from new_onehop.src.dataset import (
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VAL_COUNT,
    QuestionBatchCollator,
    build_train_val_datasets,
    load_curve_records,
)
from new_onehop.src.metrics import (
    compute_always_k1_predictor_baseline,
    compute_dynamic_rag_metrics,
    compute_fixed_k_rag_metrics,
    compute_grouped_predictor_metrics,
)
from new_onehop.src.models import MulticlassTopKPredictor
from new_onehop.src.utils import (
    ConfigError,
    load_yaml_config,
    predict_multiclass,
    project_path,
    require_mapping,
    select_device,
    set_random_seed,
    torch_save_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_yaml_atomic,
)


DEFAULT_CONFIG = PROJECT_ROOT / "new_onehop/configs/train_multiclass.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--max_train_steps",
        "--max-train-steps",
        dest="max_train_steps",
        type=int,
        default=None,
        help="Stop after this many optimizer updates (for smoke tests).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override output.run_name, useful for keeping smoke artifacts separate.",
    )
    return parser.parse_args()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")
    return value


def _finite_float(
    value: Any,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {number}")
    return number


def validate_config(config: dict[str, Any]) -> None:
    data = require_mapping(config, "data")
    model = require_mapping(config, "model")
    training = require_mapping(config, "training")
    output = require_mapping(config, "output")
    for key in ("train_oracle", "train_curves", "train_ids", "val_ids"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ConfigError(f"data.{key} must be a non-empty path string")
    if not isinstance(model.get("name"), str) or not model["name"]:
        raise ConfigError("model.name must be a non-empty string")
    if model.get("num_labels") != 15:
        raise ConfigError(
            f"model.num_labels must be exactly 15, got {model.get('num_labels')!r}"
        )
    _positive_int(model.get("max_length"), "model.max_length")
    _finite_float(model.get("dropout"), "model.dropout", 0.0, 0.999999)
    _positive_int(training.get("epochs"), "training.epochs")
    _positive_int(training.get("batch_size"), "training.batch_size")
    _positive_int(
        training.get("gradient_accumulation_steps"),
        "training.gradient_accumulation_steps",
    )
    _finite_float(training.get("learning_rate"), "training.learning_rate", 0.0)
    _finite_float(training.get("weight_decay"), "training.weight_decay", 0.0)
    _finite_float(training.get("warmup_ratio"), "training.warmup_ratio", 0.0, 1.0)
    _finite_float(
        training.get("gradient_clip_norm"),
        "training.gradient_clip_norm",
        0.0,
    )
    if not isinstance(training.get("mixed_precision"), bool):
        raise ConfigError("training.mixed_precision must be boolean")
    for key in ("root", "run_name"):
        if not isinstance(output.get(key), str) or not output[key]:
            raise ConfigError(f"output.{key} must be a non-empty string")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"seed must be an integer, got {seed!r}")


def save_best_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    tokenizer: Any,
    effective_config: dict[str, Any],
    epoch: int,
    optimizer_step: int,
    val_loss: float,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch_save_atomic(
        checkpoint_dir / "model.pt",
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "val_loss": val_loss,
        },
    )
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    write_yaml_atomic(checkpoint_dir / "config.yaml", effective_config)
    write_json_atomic(
        checkpoint_dir / "metadata.json",
        {
            "run_name": effective_config["output"]["run_name"],
            "model_name": effective_config["model"]["name"],
            "num_labels": effective_config["model"]["num_labels"],
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "val_loss": val_loss,
        },
    )


def main() -> None:
    args = parse_args()
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ConfigError("--max_train_steps must be a positive integer")
    config = load_yaml_config(args.config)
    validate_config(config)
    effective_config = copy.deepcopy(config)
    if args.run_name is not None:
        if not args.run_name.strip():
            raise ConfigError("--run-name must be non-empty")
        effective_config["output"]["run_name"] = args.run_name.strip()
    effective_config["runtime"] = {"max_train_steps": args.max_train_steps}

    data_config = effective_config["data"]
    model_config = effective_config["model"]
    training_config = effective_config["training"]
    output_config = effective_config["output"]
    seed = effective_config["seed"]
    set_random_seed(seed)
    device = select_device()
    use_amp = training_config["mixed_precision"] and device.type == "cuda"
    if training_config["mixed_precision"] and not use_amp:
        print(
            "WARNING: mixed precision requested but CUDA is unavailable; using float32"
        )

    run_name = output_config["run_name"]
    output_root = project_path(output_config["root"])
    checkpoint_run_dir = output_root / "checkpoints" / run_name
    best_checkpoint_dir = checkpoint_run_dir / "best"
    log_dir = output_root / "logs" / run_name
    prediction_path = output_root / "predictions" / "val" / f"{run_name}.jsonl"
    predictor_metrics_path = (
        output_root / "metrics" / "val" / f"{run_name}_predictor.json"
    )
    rag_metrics_path = output_root / "metrics" / "val" / f"{run_name}_rag.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(checkpoint_run_dir / "config.yaml", effective_config)
    write_yaml_atomic(log_dir / "config.yaml", effective_config)

    print(f"Device: {device}")
    print(f"Run: {run_name}")
    print(f"Loading tokenizer: {model_config['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], use_fast=True)
    train_dataset, val_dataset = build_train_val_datasets(
        data_config["train_oracle"],
        data_config["train_ids"],
        data_config["val_ids"],
        tokenizer,
        max_length=model_config["max_length"],
    )
    if (
        len(train_dataset) != EXPECTED_TRAIN_COUNT
        or len(val_dataset) != EXPECTED_VAL_COUNT
    ):
        raise RuntimeError(
            f"Unexpected dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}"
        )
    print(f"Datasets loaded: train={len(train_dataset)}, val={len(val_dataset)}")

    collator = QuestionBatchCollator(
        tokenizer,
        pad_to_multiple_of=8 if use_amp else None,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_curves = load_curve_records(data_config["train_curves"])
    val_ids = [record.question_id for record in val_dataset.records]
    fixed_k_baselines = compute_fixed_k_rag_metrics(val_curves, val_ids)

    print(f"Loading encoder: {model_config['name']}")
    model = MulticlassTopKPredictor(
        model_name=model_config["name"],
        num_labels=model_config["num_labels"],
        dropout=model_config["dropout"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    accumulation_steps = training_config["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    configured_total_updates = updates_per_epoch * training_config["epochs"]
    total_updates = (
        min(configured_total_updates, args.max_train_steps)
        if args.max_train_steps is not None
        else configured_total_updates
    )
    warmup_steps = int(total_updates * float(training_config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_val_loss = math.inf
    best_epoch = 0
    optimizer_step = 0
    training_started = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, training_config["epochs"] + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0
        epoch_started = time.time()
        reached_limit = False

        for batch_index, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)[
                    "logits"
                ]
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch={epoch}, batch={batch_index + 1}: "
                    f"{loss.item()}"
                )
            batch_size = labels.shape[0]
            epoch_loss_sum += loss.item() * batch_size
            epoch_examples += batch_size
            scaler.scale(loss / accumulation_steps).backward()

            last_batch = batch_index + 1 == len(train_loader)
            should_update = (batch_index + 1) % accumulation_steps == 0 or last_batch
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_config["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if (
                    args.max_train_steps is not None
                    and optimizer_step >= args.max_train_steps
                ):
                    reached_limit = True
                    break

        train_loss = epoch_loss_sum / epoch_examples
        validation = predict_multiclass(model, val_loader, device, use_amp)
        if len(validation.rows) != EXPECTED_VAL_COUNT:
            raise RuntimeError(
                f"Validation must produce {EXPECTED_VAL_COUNT} predictions, "
                f"got {len(validation.rows)}"
            )
        predictor_metrics = compute_grouped_predictor_metrics(
            validation.predicted_ks,
            validation.oracle_ks,
            validation.dataset_names,
        )
        always_k1_metrics = compute_always_k1_predictor_baseline(
            validation.oracle_ks, validation.dataset_names
        )
        dynamic_rag_metrics = compute_dynamic_rag_metrics(
            dict(zip(validation.question_ids, validation.predicted_ks)),
            val_curves,
            validation.question_ids,
        )
        epoch_record = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "train_loss": train_loss,
            "val_loss": validation.loss,
            "val_accuracy": predictor_metrics["overall"]["accuracy"],
            "val_macro_f1": predictor_metrics["overall"]["macro_f1"],
            "val_mae": predictor_metrics["overall"]["mae"],
            "val_avg_predicted_k": predictor_metrics["overall"]["avg_predicted_k"],
            "val_avg_f1": dynamic_rag_metrics["overall"]["avg_f1"],
            "val_avg_context_tokens": dynamic_rag_metrics["overall"][
                "avg_context_tokens"
            ],
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(epoch_record)
        write_jsonl_atomic(log_dir / "training_history.jsonl", history)
        write_json_atomic(log_dir / "training_history.json", history)

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={validation.loss:.6f} "
            f"val_accuracy={predictor_metrics['overall']['accuracy']:.6f} "
            f"val_macro_f1={predictor_metrics['overall']['macro_f1']:.6f} "
            f"val_mae={predictor_metrics['overall']['mae']:.6f} "
            f"val_avg_predicted_k={predictor_metrics['overall']['avg_predicted_k']:.6f} "
            f"val_avg_f1={dynamic_rag_metrics['overall']['avg_f1']:.6f} "
            f"val_avg_context_tokens={dynamic_rag_metrics['overall']['avg_context_tokens']:.3f}"
        )

        if validation.loss < best_val_loss:
            best_val_loss = validation.loss
            best_epoch = epoch
            save_best_checkpoint(
                best_checkpoint_dir,
                model,
                tokenizer,
                effective_config,
                epoch,
                optimizer_step,
                validation.loss,
            )
            write_jsonl_atomic(prediction_path, validation.rows)
            predictor_payload: dict[str, Any] = {
                "split": "val",
                "run_name": run_name,
                "checkpoint_selection_metric": "val_loss",
                "val_loss": validation.loss,
                **predictor_metrics,
                "always_k_1_predictor_baseline": always_k1_metrics,
            }
            rag_payload = {
                "split": "val",
                "run_name": run_name,
                "dynamic_multiclass": dynamic_rag_metrics,
                "fixed_k_baselines": fixed_k_baselines,
                "always_k_1_rag_baseline": fixed_k_baselines["fixed_k_1"],
            }
            write_json_atomic(predictor_metrics_path, predictor_payload)
            write_json_atomic(rag_metrics_path, rag_payload)
            print(f"Saved new best checkpoint (val_loss={best_val_loss:.6f})")

        if reached_limit:
            break

    write_json_atomic(
        log_dir / "summary.json",
        {
            "run_name": run_name,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "optimizer_steps": optimizer_step,
            "training_seconds": time.time() - training_started,
            "checkpoint": str(best_checkpoint_dir),
            "predictions": str(prediction_path),
            "predictor_metrics": str(predictor_metrics_path),
            "rag_metrics": str(rag_metrics_path),
        },
    )
    print(f"Best checkpoint: {best_checkpoint_dir}")
    print(f"Validation predictions: {prediction_path}")
    print(f"Predictor metrics: {predictor_metrics_path}")
    print(f"RAG replay metrics: {rag_metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
