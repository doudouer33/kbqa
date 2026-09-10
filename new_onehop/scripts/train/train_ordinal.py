#!/usr/bin/env python3
"""Train Phase 2: DeBERTa with an ordered head and plain ordinal BCE."""

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
from new_onehop.src.losses import build_ordinal_targets, ordinal_bce_loss
from new_onehop.src.metrics import (
    compute_always_k1_predictor_baseline,
    compute_dynamic_rag_metrics,
    compute_dynamic_rag_utility,
    compute_fixed_k_rag_metrics,
    compute_global_max_context_tokens,
    compute_grouped_predictor_metrics,
)
from new_onehop.src.models import OrdinalTopKPredictor
from new_onehop.src.utils import (
    ConfigError,
    OrdinalPredictionOutput,
    load_yaml_config,
    predict_ordinal,
    project_path,
    require_mapping,
    select_device,
    set_random_seed,
    torch_save_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_yaml_atomic,
)


DEFAULT_CONFIG = PROJECT_ROOT / "new_onehop/configs/train_ordinal.yaml"
SELECTIONS = ("best_val_loss", "best_rag_utility")


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
        help="Override output.run_name so smoke artifacts remain separate.",
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
    selection = require_mapping(config, "selection")
    output = require_mapping(config, "output")
    for key in ("train_oracle", "train_curves", "train_ids", "val_ids"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ConfigError(f"data.{key} must be a non-empty path string")
    if model.get("type") != "ordinal":
        raise ConfigError(f"model.type must be 'ordinal', got {model.get('type')!r}")
    if not isinstance(model.get("name"), str) or not model["name"]:
        raise ConfigError("model.name must be a non-empty string")
    if model.get("num_labels") != 15:
        raise ConfigError(
            f"model.num_labels must be exactly 15, got {model.get('num_labels')!r}"
        )
    if model.get("num_thresholds") != 14:
        raise ConfigError(
            "model.num_thresholds must be exactly 14, "
            f"got {model.get('num_thresholds')!r}"
        )
    _positive_int(model.get("max_length"), "model.max_length")
    _finite_float(model.get("dropout"), "model.dropout", 0.0, 0.999999)
    threshold_min = _finite_float(
        model.get("threshold_init_min"), "model.threshold_init_min"
    )
    threshold_max = _finite_float(
        model.get("threshold_init_max"), "model.threshold_init_max"
    )
    if threshold_min >= threshold_max:
        raise ConfigError(
            "model.threshold_init_min must be less than threshold_init_max"
        )
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
    _finite_float(
        selection.get("rag_utility_lambda_T"),
        "selection.rag_utility_lambda_T",
        0.0,
    )
    for key in ("root", "run_name"):
        if not isinstance(output.get(key), str) or not output[key]:
            raise ConfigError(f"output.{key} must be a non-empty string")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"seed must be an integer, got {seed!r}")


def save_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    tokenizer: Any,
    effective_config: dict[str, Any],
    *,
    epoch: int,
    optimizer_step: int,
    val_loss: float,
    val_rag_utility: float,
    selection_metric: str,
) -> None:
    if selection_metric not in ("val_loss", "val_rag_utility"):
        raise ValueError(f"Unexpected selection metric: {selection_metric!r}")
    selection_value = val_loss if selection_metric == "val_loss" else val_rag_utility
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch_save_atomic(
        checkpoint_dir / "model.pt",
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "val_loss": val_loss,
            "val_rag_utility": val_rag_utility,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
        },
    )
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    write_yaml_atomic(checkpoint_dir / "config.yaml", effective_config)
    write_json_atomic(
        checkpoint_dir / "metadata.json",
        {
            "run_name": effective_config["output"]["run_name"],
            "model_type": "ordinal",
            "model_name": effective_config["model"]["name"],
            "num_labels": effective_config["model"]["num_labels"],
            "num_thresholds": effective_config["model"]["num_thresholds"],
            "decoding": "categorical_argmax_from_ordinal_distribution",
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "val_loss": val_loss,
            "val_rag_utility": val_rag_utility,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
        },
    )


def save_selected_metrics(
    *,
    output_root: Path,
    run_name: str,
    selection_name: str,
    selection_metric: str,
    selection_value: float,
    checkpoint_dir: Path,
    validation: OrdinalPredictionOutput,
    predictor_metrics: dict[str, dict[str, float]],
    always_k1_metrics: dict[str, dict[str, float]],
    dynamic_rag_metrics: dict[str, dict[str, float | int | None]],
    fixed_k_baselines: dict[str, dict[str, dict[str, float | int | None]]],
    val_rag_utility: float,
    lambda_T: float,
    t_max: int,
) -> None:
    if selection_name not in SELECTIONS:
        raise ValueError(f"Unexpected checkpoint selection name: {selection_name!r}")
    artifact_name = f"{run_name}_{selection_name}"
    prediction_path = output_root / "predictions" / "val" / f"{artifact_name}.jsonl"
    predictor_path = output_root / "metrics" / "val" / f"{artifact_name}_predictor.json"
    rag_path = output_root / "metrics" / "val" / f"{artifact_name}_rag.json"
    write_jsonl_atomic(prediction_path, validation.rows)
    write_json_atomic(
        predictor_path,
        {
            "split": "val",
            "run_name": run_name,
            "model_type": "ordinal",
            "checkpoint": str(checkpoint_dir),
            "checkpoint_selection_metric": selection_metric,
            "checkpoint_selection_value": selection_value,
            "loss": validation.loss,
            "decoding": "categorical_argmax_from_ordinal_distribution",
            "thresholds": validation.thresholds,
            "numerical_sanity": validation.numerical_sanity,
            **predictor_metrics,
            "always_k_1_predictor_baseline": always_k1_metrics,
        },
    )
    write_json_atomic(
        rag_path,
        {
            "split": "val",
            "run_name": artifact_name,
            "model_type": "ordinal",
            "num_predictions": len(validation.rows),
            "checkpoint": str(checkpoint_dir),
            "checkpoint_selection_metric": selection_metric,
            "checkpoint_selection_value": selection_value,
            "dynamic_model": dynamic_rag_metrics,
            "rag_utility_lambda_T": lambda_T,
            "rag_utility_t_max": t_max,
            "val_rag_utility": val_rag_utility,
            "fixed_k_baselines": fixed_k_baselines,
            "always_k_1_rag_baseline": fixed_k_baselines["fixed_k_1"],
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
    selection_config = effective_config["selection"]
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
    checkpoint_dirs = {
        "best_val_loss": checkpoint_run_dir / "best_val_loss",
        "best_rag_utility": checkpoint_run_dir / "best_rag_utility",
    }
    log_dir = output_root / "logs" / run_name
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
    train_pool_curves = load_curve_records(data_config["train_curves"])
    val_ids = [record.question_id for record in val_dataset.records]
    fixed_k_baselines = compute_fixed_k_rag_metrics(train_pool_curves, val_ids)
    t_max = compute_global_max_context_tokens(train_pool_curves)
    lambda_T = float(selection_config["rag_utility_lambda_T"])
    print(f"Validation utility: lambda_T={lambda_T}, train-pool T_max={t_max}")

    print(f"Loading encoder: {model_config['name']}")
    model = OrdinalTopKPredictor(
        model_name=model_config["name"],
        num_labels=model_config["num_labels"],
        num_thresholds=model_config["num_thresholds"],
        dropout=model_config["dropout"],
        threshold_init_min=model_config["threshold_init_min"],
        threshold_init_max=model_config["threshold_init_max"],
    ).to(device)
    initial_thresholds = model.ordered_thresholds().detach()
    if initial_thresholds.shape != (model_config["num_thresholds"],) or torch.any(
        initial_thresholds[1:] <= initial_thresholds[:-1]
    ):
        raise RuntimeError("Initial ordinal thresholds are not strictly increasing")
    print(
        "Initial thresholds: "
        + ", ".join(f"{value:.6f}" for value in initial_thresholds.cpu().tolist())
    )

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
    best_rag_utility = -math.inf
    best_val_loss_epoch = 0
    best_rag_utility_epoch = 0
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
            batch_size = labels.shape[0]
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                ordinal_logits = output["ordinal_logits"]
                ordinal_targets = build_ordinal_targets(
                    labels, model_config["num_thresholds"]
                )
                if ordinal_logits.shape != (
                    batch_size,
                    model_config["num_thresholds"],
                ):
                    raise RuntimeError(
                        "Unexpected ordinal logits shape at "
                        f"epoch={epoch}, batch={batch_index + 1}: "
                        f"{tuple(ordinal_logits.shape)}"
                    )
                if ordinal_targets.shape != ordinal_logits.shape:
                    raise RuntimeError(
                        "Ordinal target shape does not match logits: "
                        f"{tuple(ordinal_targets.shape)} vs {tuple(ordinal_logits.shape)}"
                    )
                loss = ordinal_bce_loss(ordinal_logits, ordinal_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch={epoch}, batch={batch_index + 1}: "
                    f"{loss.item()}"
                )
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
                thresholds_after_step = model.ordered_thresholds().detach()
                if not torch.isfinite(thresholds_after_step).all() or torch.any(
                    thresholds_after_step[1:] <= thresholds_after_step[:-1]
                ):
                    raise RuntimeError(
                        f"Invalid threshold order after optimizer step {optimizer_step}"
                    )
                if (
                    args.max_train_steps is not None
                    and optimizer_step >= args.max_train_steps
                ):
                    reached_limit = True
                    break

        if epoch_examples == 0:
            raise RuntimeError("Training data loader produced no examples")
        train_loss = epoch_loss_sum / epoch_examples
        validation = predict_ordinal(
            model,
            val_loader,
            device,
            use_amp,
            num_thresholds=model_config["num_thresholds"],
        )
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
        predicted_k_by_id = dict(zip(validation.question_ids, validation.predicted_ks))
        dynamic_rag_metrics = compute_dynamic_rag_metrics(
            predicted_k_by_id,
            train_pool_curves,
            validation.question_ids,
        )
        val_rag_utility = compute_dynamic_rag_utility(
            predicted_k_by_id,
            train_pool_curves,
            validation.question_ids,
            lambda_T=lambda_T,
            t_max=t_max,
        )
        overall = predictor_metrics["overall"]
        overall_rag = dynamic_rag_metrics["overall"]
        epoch_record = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "train_loss": train_loss,
            "val_loss": validation.loss,
            "val_accuracy": overall["accuracy"],
            "val_macro_f1": overall["macro_f1"],
            "val_mae": overall["mae"],
            "val_rmse": overall["rmse"],
            "val_within_1_accuracy": overall["within_1_accuracy"],
            "val_within_2_accuracy": overall["within_2_accuracy"],
            "val_under_retrieval_rate": overall["under_retrieval_rate"],
            "val_over_retrieval_rate": overall["over_retrieval_rate"],
            "val_avg_predicted_k": overall["avg_predicted_k"],
            "val_avg_oracle_k": overall["avg_oracle_k"],
            "val_avg_f1": overall_rag["avg_f1"],
            "val_avg_context_tokens": overall_rag["avg_context_tokens"],
            "val_avg_k": overall_rag["avg_k"],
            "val_rag_utility": val_rag_utility,
            "threshold_min": min(validation.thresholds),
            "threshold_max": max(validation.thresholds),
            "thresholds": validation.thresholds,
            **validation.numerical_sanity,
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(epoch_record)
        write_jsonl_atomic(log_dir / "training_history.jsonl", history)
        write_json_atomic(log_dir / "training_history.json", history)

        print(
            f"epoch={epoch} optimizer_step={optimizer_step} "
            f"train_loss={train_loss:.6f} val_loss={validation.loss:.6f} "
            f"val_accuracy={overall['accuracy']:.6f} "
            f"val_macro_f1={overall['macro_f1']:.6f} "
            f"val_mae={overall['mae']:.6f} "
            f"val_avg_predicted_k={overall['avg_predicted_k']:.6f} "
            f"val_avg_f1={overall_rag['avg_f1']:.6f} "
            f"val_avg_context_tokens={overall_rag['avg_context_tokens']:.3f} "
            f"val_rag_utility={val_rag_utility:.6f}"
        )

        if validation.loss < best_val_loss:
            best_val_loss = validation.loss
            best_val_loss_epoch = epoch
            checkpoint_dir = checkpoint_dirs["best_val_loss"]
            save_checkpoint(
                checkpoint_dir,
                model,
                tokenizer,
                effective_config,
                epoch=epoch,
                optimizer_step=optimizer_step,
                val_loss=validation.loss,
                val_rag_utility=val_rag_utility,
                selection_metric="val_loss",
            )
            save_selected_metrics(
                output_root=output_root,
                run_name=run_name,
                selection_name="best_val_loss",
                selection_metric="val_loss",
                selection_value=validation.loss,
                checkpoint_dir=checkpoint_dir,
                validation=validation,
                predictor_metrics=predictor_metrics,
                always_k1_metrics=always_k1_metrics,
                dynamic_rag_metrics=dynamic_rag_metrics,
                fixed_k_baselines=fixed_k_baselines,
                val_rag_utility=val_rag_utility,
                lambda_T=lambda_T,
                t_max=t_max,
            )
            print(f"Saved best_val_loss checkpoint ({best_val_loss:.6f})")

        if val_rag_utility > best_rag_utility:
            best_rag_utility = val_rag_utility
            best_rag_utility_epoch = epoch
            checkpoint_dir = checkpoint_dirs["best_rag_utility"]
            save_checkpoint(
                checkpoint_dir,
                model,
                tokenizer,
                effective_config,
                epoch=epoch,
                optimizer_step=optimizer_step,
                val_loss=validation.loss,
                val_rag_utility=val_rag_utility,
                selection_metric="val_rag_utility",
            )
            save_selected_metrics(
                output_root=output_root,
                run_name=run_name,
                selection_name="best_rag_utility",
                selection_metric="val_rag_utility",
                selection_value=val_rag_utility,
                checkpoint_dir=checkpoint_dir,
                validation=validation,
                predictor_metrics=predictor_metrics,
                always_k1_metrics=always_k1_metrics,
                dynamic_rag_metrics=dynamic_rag_metrics,
                fixed_k_baselines=fixed_k_baselines,
                val_rag_utility=val_rag_utility,
                lambda_T=lambda_T,
                t_max=t_max,
            )
            print("Saved best_rag_utility checkpoint " f"({best_rag_utility:.6f})")

        if reached_limit:
            break

    training_seconds = time.time() - training_started
    write_json_atomic(
        log_dir / "summary.json",
        {
            "run_name": run_name,
            "best_val_loss_epoch": best_val_loss_epoch,
            "best_val_loss": best_val_loss,
            "best_rag_utility_epoch": best_rag_utility_epoch,
            "best_rag_utility": best_rag_utility,
            "optimizer_steps": optimizer_step,
            "configured_total_optimizer_steps": configured_total_updates,
            "training_seconds": training_seconds,
            "train_pool_t_max": t_max,
            "rag_utility_lambda_T": lambda_T,
            "checkpoints": {name: str(path) for name, path in checkpoint_dirs.items()},
        },
    )
    print(f"Training seconds: {training_seconds:.3f}")
    print(f"Best val-loss checkpoint: {checkpoint_dirs['best_val_loss']}")
    print(f"Best RAG-utility checkpoint: {checkpoint_dirs['best_rag_utility']}")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
