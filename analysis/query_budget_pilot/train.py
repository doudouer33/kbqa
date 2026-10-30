#!/usr/bin/env python3
"""CPU-only Linear / MLP training; select checkpoints using valid validation loss."""

import argparse
import platform
import random
from pathlib import Path

from common import (
    HERE,
    LABELS,
    MODELS,
    allocate_output,
    fingerprint,
    fresh_directory,
    load_config,
    load_prepared,
    read_json,
    require,
    resolve,
    sha256,
    validate_split,
    write_json,
)
from extract_embeddings import load_verified


def paths_from_config(config, prepared=None, embeddings=None, metadata=None):
    paths = (
        Path(prepared or resolve(config["prepared"])),
        Path(embeddings or resolve(config["embeddings"])),
        Path(metadata or resolve(config["embedding_metadata"])),
    )
    require(
        paths[1].is_file() and paths[2].is_file(),
        "Missing real embeddings.npz / embedding_metadata.json. First manually run the A800 extraction job and return both files; no model will be downloaded here.",
    )
    return paths


def load_inputs(config, seed, prepared, embeddings, metadata):
    queries, labels, curves, manifest = load_prepared(prepared, config["expected_count"])
    require(
        manifest["pilot_configuration"]["labels"] == config["labels"],
        "Configured targets differ from prepared labels; prepare a new dataset",
    )
    require(seed in manifest["seeds"], "Seed has no saved split")
    split = read_json(prepared / "splits" / ("seed_{}.json".format(seed)))
    require(split["seed"] == seed, "Split seed mismatch")
    validate_split(split, [q["question_id"] for q in queries])
    features, embedding_metadata = load_verified(
        prepared / "queries.jsonl", embeddings, metadata, len(queries), config["embedding_dim"]
    )
    require(embedding_metadata["normalization"] == "l2", "Pilot expects L2-normalized embeddings")
    return queries, labels, curves, split, features


def make_model(kind, input_dim, hidden_dim=256, dropout=0.2):
    import torch.nn as nn

    require(kind in MODELS, "Unknown model")
    if kind == "linear":
        return nn.Linear(input_dim, 15)
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 15))


def soft_cross_entropy(logits, targets):
    import torch

    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def fit(config, kind, label_name, seed, output, prepared, embeddings, metadata, overrides=None):
    require(kind in MODELS and label_name in LABELS, "Unknown model or target")
    queries, labels, _, split, features = load_inputs(config, seed, prepared, embeddings, metadata)
    import numpy as np
    import torch

    settings = dict(config["training"])
    settings["learning_rate"] = settings["learning_rates"][kind]
    settings.update(overrides or {})
    require(
        settings["epochs"] > 0
        and settings["batch_size"] > 0
        and settings["patience"] > 0
        and settings["num_threads"] > 0,
        "Invalid training counts",
    )
    require(
        settings["learning_rate"] > 0 and settings["weight_decay"] >= 0 and settings["min_delta"] >= 0,
        "Invalid optimizer settings",
    )
    require(0 <= settings["dropout"] < 1 and settings["hidden_dim"] > 0, "Invalid MLP settings")
    torch.set_num_threads(settings["num_threads"])
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    index = {q["question_id"]: i for i, q in enumerate(queries)}
    subsets = {}
    # Invalid samples never enter the loss, including validation loss. Test targets are not read here.
    for part in ("train", "validation"):
        ids = [qid for qid in split["partitions"][part] if labels[qid]["has_valid_label"]]
        require(bool(ids), "No supervised samples in " + part)
        subsets[part] = (
            torch.from_numpy(features[[index[qid] for qid in ids]]),
            torch.tensor([labels[qid]["targets"][label_name]["probabilities"] for qid in ids], dtype=torch.float32),
        )
    model = make_model(kind, features.shape[1], settings["hidden_dim"], settings["dropout"]).to("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    generator = torch.Generator().manual_seed(seed)
    output = fresh_directory(output)
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    data_hashes = fingerprint(prepared, embeddings, metadata, seed)
    x_train, y_train = subsets["train"]
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), settings["batch_size"]):
            batch = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = soft_cross_entropy(model(x_train[batch]), y_train[batch])
            require(bool(torch.isfinite(loss)), "Non-finite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            losses = {part: float(soft_cross_entropy(model(x), y)) for part, (x, y) in subsets.items()}
        require(all(np.isfinite(v) for v in losses.values()), "Non-finite train / validation loss")
        history.append({"epoch": epoch, "train_loss": losses["train"], "validation_loss": losses["validation"]})
        if losses["validation"] < best_loss - settings["min_delta"]:
            best_loss, best_epoch, stale = losses["validation"], epoch, 0
            # Keep only tensors in pickle. Architecture/settings live in training.json;
            # this also works with torch 1.13's restricted weights-only unpickler.
            torch.save({"state_dict": model.state_dict()}, output / "best.pt")
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    summary = {
        "model": kind,
        "label": label_name,
        "seed": seed,
        "device": "cpu",
        "settings": settings,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "selection_metric": "soft cross entropy on valid validation IDs only; earliest improvement beyond min_delta",
        "supervised_counts": {part: len(pair[0]) for part, pair in subsets.items()},
        "full_partition_counts": {part: len(ids) for part, ids in split["partitions"].items()},
        "prepared": str(prepared.resolve()),
        "embeddings": str(embeddings.resolve()),
        "embedding_metadata": str(metadata.resolve()),
        "input_sha256": data_hashes,
        "checkpoint_sha256": sha256(output / "best.pt"),
        "code_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        "configuration": config,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "study_scope": "Exploratory HotpotQA Dev pilot; no independent generalization claim",
    }
    write_json(output / "history.json", history)
    write_json(output / "training.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--label", choices=LABELS, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--embedding-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    for flag, typ in (
        ("epochs", int),
        ("batch-size", int),
        ("patience", int),
        ("num-threads", int),
        ("learning-rate", float),
        ("weight-decay", float),
    ):
        parser.add_argument("--" + flag, type=typ)
    args = parser.parse_args()
    config = load_config(args.config)
    prepared, embeddings, metadata = paths_from_config(config, args.prepared, args.embeddings, args.embedding_metadata)
    overrides = {
        name: getattr(args, name)
        for name in ("epochs", "batch_size", "patience", "num_threads", "learning_rate", "weight_decay")
        if getattr(args, name) is not None
    }
    output = args.output or allocate_output("train")
    result = fit(config, args.model, args.label, args.seed, output, prepared, embeddings, metadata, overrides)
    print(
        "Saved {}: best epoch={}, validation loss={:.6f}".format(
            output, result["best_epoch"], result["best_validation_loss"]
        )
    )


if __name__ == "__main__":
    main()
