"""Smoke-test and fine-tune six independent BERT heads, then freeze Validation selections."""

import fcntl
import json
import math
import random
import time
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, default_collate

from prepare_data import provenance, read_json, require, sha256
from training_common import (CLASSES, EXPERIMENTS, HEAD_CLASSES, RULE, check_training_config, checkpoint, class_weights,
                             classifier_metrics, e2e, entrypoint, head_path, label_counts, load_data, read_labels, route,
                             score_key, setup_device, write_csv)
from analysis.hotpot_formal_runner.storage import atomic_json  # noqa: E402

INITIAL_LOSS_SCALE = 1024


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenizer_for(paths):
    from transformers import AutoTokenizer

    cfg = paths.config
    return AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"], local_files_only=True, use_fast=True)


def dataset_for(rows, head, tokenizer, max_length):
    require(rows, "Cannot tokenize an empty cohort")
    enc = tokenizer([row["question"] for row in rows], padding=True, pad_to_multiple_of=8,
                    truncation=True, max_length=max_length, return_tensors="pt")
    target = torch.tensor([row[f"{head}_label"] if row[f"{head}_label"] is not None else -100 for row in rows])
    return TensorDataset(*(enc[key] for key in ("input_ids", "attention_mask", "token_type_ids")), target)


def trim_batch(batch):
    ids, mask, segments, target = default_collate(batch)
    length = (int(mask.sum(1).max()) + 7) // 8 * 8
    return ids[:, :length], mask[:, :length], segments[:, :length], target


def batches(data, size, seed=None):
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(data, batch_size=size, shuffle=seed is not None, generator=generator, pin_memory=True,
                      collate_fn=trim_batch, num_workers=0)


def new_model(paths, head, device, directory=None):
    from transformers import BertForSequenceClassification

    if directory is None:
        cfg = paths.config
        names = {i: str(v) for i, v in enumerate(HEAD_CLASSES[head])}
        model = BertForSequenceClassification.from_pretrained(cfg["model"], revision=cfg["revision"], local_files_only=True,
                                                               num_labels=len(names), id2label=names,
                                                               label2id={name: i for i, name in names.items()})
    else:
        model = BertForSequenceClassification.from_pretrained(str(directory), local_files_only=True)
    require(model.config.num_labels == len(HEAD_CLASSES[head]), f"Wrong {head} output dimension")
    return model.to(device)


def run_epoch(model, loader, device, weights, optimizer=None, scaler=None, scheduler=None, clip=1.0):
    training = optimizer is not None
    model.train(training)
    predicted, truth = [], []
    numerator = denominator = 0.
    with torch.set_grad_enabled(training):
        for ids, mask, segments, target in loader:
            ids, mask, segments, target = (x.to(device, non_blocking=True) for x in (ids, mask, segments, target))
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=mask, token_type_ids=segments).logits
            losses = F.cross_entropy(logits.float(), target, weight=weights, reduction="none", ignore_index=-100)
            active = target != -100
            per_batch_sum = losses.sum()
            per_batch_weight = active.sum() if weights is None else weights[target.clamp_min(0)].masked_fill(~active, 0).sum()
            if training:
                loss = per_batch_sum / per_batch_weight
                require(torch.isfinite(loss).item(), "Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            numerator = numerator + per_batch_sum.detach()
            denominator = denominator + per_batch_weight.detach()
            predicted.append(logits.detach().argmax(-1))
            truth.append(target)
    actual_denom = float(denominator)
    avg_loss = float(numerator) / actual_denom if actual_denom else None
    require(avg_loss is None or math.isfinite(avg_loss), "Nonfinite epoch loss")
    return avg_loss, torch.cat(truth).cpu().numpy(), torch.cat(predicted).cpu().numpy()


def make_optimizer(model, paths, steps):
    from transformers import get_linear_schedule_with_warmup

    cfg = paths.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * cfg["warmup_ratio"]), steps)
    return optimizer, scheduler, torch.cuda.amp.GradScaler(init_scale=INITIAL_LOSS_SCALE)


def loss_weights(rows, head, weighted, device):
    counts = label_counts(rows, head)
    weights = class_weights(counts) if weighted else np.ones(counts.size)
    return counts.tolist(), weights.tolist(), torch.tensor(weights, dtype=torch.float32, device=device) if weighted else None


def smoke_test(paths, labels, tokenizer, device):
    seed = paths.config["seed"]
    results = []
    for experiment, (_, heads, weighted) in EXPERIMENTS.items():
        for head in heads:
            seed_everything(seed)
            classes = HEAD_CLASSES[head]
            sample = [next(row for row in labels["train"] if row[f"{head}_label"] == i) for i in range(len(classes))]
            full_counts, weights, tensor = loss_weights(labels["train"], head, weighted, device)
            data = dataset_for(sample, head, tokenizer, paths.config["max_sequence_length"])
            model = new_model(paths, head, device)
            probe = model.bert.encoder.layer[0].attention.self.query.weight.detach().clone()
            optimizer, scheduler, scaler = make_optimizer(model, paths, 1)
            loss, targets, predictions = run_epoch(model, batches(data, len(sample)), device, tensor,
                                                   optimizer, scaler, scheduler, paths.config["gradient_clip"])
            require(np.array_equal(targets, np.arange(len(classes))) and np.isin(predictions, targets).all()
                    and bool((probe != model.bert.encoder.layer[0].attention.self.query.weight).any()),
                    f"Smoke output, target encoding or encoder update failed for {experiment}/{head}")
            results.append({"experiment": experiment, "head": head, "train_counts": full_counts,
                            "weights": weights, "sample_count": len(sample), "loss": loss, "output_classes": len(classes)})
            del model, optimizer, scheduler, scaler
    require(route(np.array([0, 1, 0, 1]), np.array([3, -1, 0, -1])).tolist() == [3, 10, 0, 10], "Smoke routing failed")
    atomic_json(paths.training / "smoke.json", {"status": "passed", "heads": results,
                                                "initial_loss_scale": INITIAL_LOSS_SCALE, "predicted_routing_verified": True})
    print("SMOKE PASS: six heads, both losses, labels and predicted routing", flush=True)


def train_head(paths, experiment, head, weighted, labels, tokenizer, device):
    cfg = paths.config
    directory = head_path(paths, experiment, head)
    size = cfg["train_batch_size"]
    counts, weights, weight_tensor = loss_weights(labels["train"], head, weighted, device)
    train_rows = [row for row in labels["train"] if row[f"{head}_label"] is not None]
    train = dataset_for(train_rows, head, tokenizer, cfg["max_sequence_length"])
    val = dataset_for(labels["val"], head, tokenizer, cfg["max_sequence_length"])
    val_loader = batches(val, size)
    checkpoint_root = directory / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        saved = checkpoint(directory, epoch)
        if not saved.exists():
            require(not any(checkpoint(directory, later).exists() for later in range(epoch+1, cfg["epochs"]+1)),
                    "Checkpoint sequence has a gap")
            break
        history.append(read_json(saved / "epoch.json"))
    start_epoch = len(history) + 1
    if start_epoch <= cfg["epochs"]:
        seed_everything(cfg["seed"])
        last = checkpoint(directory, start_epoch - 1)
        model = new_model(paths, head, device, last if history else None)
        optimizer, scheduler, scaler = make_optimizer(model, paths, math.ceil(len(train) / size) * cfg["epochs"])
        if history:
            state = torch.load(last / "resume.pt", map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            del state
        probe = model.bert.encoder.layer[0].attention.self.query.weight.detach().clone()
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            start = time.monotonic()
            seed = cfg["seed"] + epoch - 1
            seed_everything(seed)
            train_loss, truth, pred = run_epoch(model, batches(train, size, seed), device, weight_tensor,
                                                optimizer, scaler, scheduler, cfg["gradient_clip"])
            train_stats = classifier_metrics(truth, pred, HEAD_CLASSES[head])
            val_loss, truth, pred = run_epoch(model, val_loader, device, weight_tensor)
            labeled = truth != -100
            validation_stats = classifier_metrics(truth[labeled], pred[labeled], HEAD_CLASSES[head])
            changed = bool((probe != model.bert.encoder.layer[0].attention.self.query.weight).any())
            require(changed, f"BERT encoder did not update in {experiment}/{head}")
            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_online": train_stats,
                      "validation": validation_stats, "learning_rate_end": scheduler.get_last_lr()[0],
                      "seconds": round(time.monotonic()-start, 3), "encoder_updated": changed}
            saved = checkpoint(directory, epoch)
            with TemporaryDirectory(dir=checkpoint_root, suffix=".tmp") as tmp:
                staged = Path(tmp) / "epoch"
                model.save_pretrained(staged)
                atomic_json(staged / "epoch.json", record)
                atomic_json(staged / "validation_classes.json", {"qids": [row["qid"] for row in labels["val"]],
                                                                   "class_indices": pred.tolist()})
                if epoch < cfg["epochs"]:
                    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                "scaler": scaler.state_dict()}, staged / "resume.pt")
                staged.replace(saved)
            (checkpoint(directory, epoch-1) / "resume.pt").unlink(missing_ok=True)
            history.append(record)
            atomic_json(directory / "history.json", {"experiment": experiment, "head": head, "train_counts": counts,
                                                      "weights": weights, "history": history})
            print(f"EPOCH {experiment}/{head} {epoch}/{cfg['epochs']} loss={train_loss:.5f} "
                  f"val_loss={val_loss:.5f} val_macro_f1={validation_stats['macro_f1']:.5f} seconds={record['seconds']:.1f}", flush=True)
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
    atomic_json(directory / "history.json", {"experiment": experiment, "head": head, "train_counts": counts,
                                              "weights": weights, "history": history})


def fit(paths, args):
    check_training_config(paths)
    device = setup_device(paths)
    datasets, sources = load_data(paths, ("train", "val"))
    labels = read_labels(paths, datasets)
    plan = {"configuration": paths.config, "sources": {**sources, "labels": provenance(paths.labels),
             "training_config": provenance(paths.config_path)}, "model_input": ["question"],
            "optimizer": "AdamW", "scheduler": "linear with warmup", "sequence_padding": "batch maximum rounded to 8",
            "gradient_accumulation": 1, "checkpoint_selection": RULE, "train_test_isolation": "Validation only",
            "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__, "cuda": torch.version.cuda}
    paths.training.mkdir(parents=True, exist_ok=True)
    with (paths.training / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_file = paths.training / "plan.json"
        if plan_file.exists():
            require(read_json(plan_file) == plan, "Existing training run has a different configuration or source")
        else:
            atomic_json(plan_file, plan)
        tokenizer = tokenizer_for(paths)
        if args.smoke or not (paths.training / "smoke.json").exists():
            smoke_test(paths, labels, tokenizer, device)
        require(read_json(paths.training / "smoke.json")["status"] == "passed", "Smoke test has not passed")
        if args.smoke:
            return
        for experiment, (_, heads, weighted) in EXPERIMENTS.items():
            for head in heads:
                train_head(paths, experiment, head, weighted, labels, tokenizer, device)
        freeze(paths, datasets["val"][1], labels["val"], plan)


def freeze(paths, cohort, labels, plan):
    epochs = range(1, paths.config["epochs"] + 1)
    qids = [row["qid"] for row in labels]
    choices, runs = {}, []
    for experiment, (_, heads, _) in EXPERIMENTS.items():
        predictions = {}
        for head in heads:
            for epoch in epochs:
                saved = checkpoint(head_path(paths, experiment, head), epoch)
                row = read_json(saved / "validation_classes.json")
                require(row["qids"] == qids, f"Validation predictions do not align in {saved}")
                predictions[head, epoch] = np.asarray(row["class_indices"], dtype=int)
        runs_for_model = []
        for combination in product(epochs, repeat=len(heads)):
            first = predictions[heads[0], combination[0]]
            selected = route(first, predictions[heads[1], combination[1]]) if len(heads) == 2 else np.asarray(CLASSES)[first]
            metrics = e2e(cohort, selected, paths)
            runs_for_model.append((metrics, combination))
            runs.append({"experiment": experiment, "first_epoch": combination[0],
                         "stage2_epoch": combination[1] if len(heads) == 2 else None,
                         **{k: v for k, v in metrics.items() if k != "selected_k_distribution"},
                         "selected_k_distribution": json.dumps(metrics["selected_k_distribution"])})
        best, chosen = max(runs_for_model, key=lambda pair: score_key(*pair))
        selected_heads = {}
        for head, epoch in zip(heads, chosen):
            directory = checkpoint(head_path(paths, experiment, head), epoch)
            selected_heads[head] = {"epoch": epoch, "weights": provenance(directory / "pytorch_model.bin"),
                                    "model_config": provenance(directory / "config.json")}
        best["selected_k_distribution"] = {str(k): v for k, v in best["selected_k_distribution"].items()}
        choices[experiment] = {"heads": selected_heads, "validation": best, "combinations_checked": len(runs_for_model)}
    write_csv(paths.training / "validation_checkpoint_sweep.csv", runs)
    frozen = {"schema_version": 1, "plan": plan, "selection_split": "val", "selection_rule": RULE,
              "score_lambda": paths.config["score_lambda"], "token_scale": paths.selection["token_scale"],
              "random_prior_seeds": paths.config["random_prior_seeds"], "choices": choices,
              "sweep_sha256": sha256(paths.training / "validation_checkpoint_sweep.csv")}
    output = paths.training / "frozen_selection.json"
    if output.exists():
        require(read_json(output) == frozen, "Previously frozen selections differ")
    else:
        atomic_json(output, frozen)
    print("VALIDATION CHECKPOINTS FROZEN:", {exp: {head: spec["epoch"] for head, spec in row["heads"].items()}
                                             for exp, row in choices.items()}, flush=True)


if __name__ == "__main__":
    entrypoint(fit, __doc__, lambda parser: parser.add_argument("--smoke", action="store_true", help="Run only the smoke test"))
