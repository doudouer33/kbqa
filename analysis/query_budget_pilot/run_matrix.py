#!/usr/bin/env python3
"""Run 3 seeds x 3 labels x 2 models sequentially on CPU, then report."""

import argparse
from pathlib import Path

from common import HERE, LABELS, MODELS, allocate_output, fresh_directory, load_config, write_json
from evaluate import aggregate, evaluate
from train import fit, paths_from_config


def run(config, output, prepared, embeddings, metadata, overrides=None):
    # Validate real features before allocating output or importing torch.
    from train import load_inputs

    load_inputs(config, config["seeds"][0], prepared, embeddings, metadata)
    output = fresh_directory(output)
    manifest = {
        "status": "running",
        "seeds": config["seeds"],
        "models": list(MODELS),
        "labels": list(LABELS),
        "runs": [],
    }
    write_json(output / "matrix.json", manifest)
    for seed in config["seeds"]:
        for kind in MODELS:
            for label in LABELS:
                relative = "seed_{}/{}_{}".format(seed, kind, label)
                print("Training " + relative, flush=True)
                fit(config, kind, label, seed, output / relative, prepared, embeddings, metadata, overrides)
                evaluate(output / relative)
                manifest["runs"].append(relative)
                write_json(output / "matrix.json", manifest)
    manifest["status"] = "complete"
    write_json(output / "matrix.json", manifest)
    aggregate(output)
    print("Saved report: " + str(output / "report" / "report.md"), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--embedding-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-threads", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = paths_from_config(config, args.prepared, args.embeddings, args.embedding_metadata)
    overrides = {key: getattr(args, key) for key in ("epochs", "num_threads") if getattr(args, key) is not None}
    run(config, args.output or allocate_output("matrix"), *paths, overrides=overrides)


if __name__ == "__main__":
    main()
