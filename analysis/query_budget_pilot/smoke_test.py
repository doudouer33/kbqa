#!/usr/bin/env python3
"""Synthetic-only smoke: 18 tiny CPU runs, offline evaluation and report.

All random features are paired exclusively with synthetic questions/curves.
No HotpotQA training, model downloads or GPU use occurs.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from common import HERE, fresh_directory, read_json, require, write_json

sys.path.insert(0, str(HERE / "tests"))
from synthetic import fixture  # noqa: E402
from run_matrix import run  # noqa: E402


def smoke(output):
    with tempfile.TemporaryDirectory(prefix="query_budget_SYNTHETIC_") as directory:
        config, prepared, embeddings, metadata, _, _ = fixture(Path(directory))
        matrix = Path(directory) / "SYNTHETIC_matrix"
        manifest = run(config, matrix, prepared, embeddings, metadata, {"epochs": 3, "num_threads": 1})
        report = read_json(matrix / "report" / "summary.json")
        require(len(manifest["runs"]) == 18, "Expected full 18-run smoke matrix")
        models = [
            r for r in report["rows"] if r["partition"] == "test" and r["group"] == "combined" and r["cohort"] == "all"
        ]
        require(len(models) == 11, "Expected six learned methods + four fixed baselines + offline Oracle")
        require(all(r["sample_count"]["mean"] == 20 for r in models), "Full synthetic test coverage failed")
        for relative in manifest["runs"]:
            training = read_json(matrix / relative / "training.json")
            require(training["device"] == "cpu", "Smoke must stay on CPU")
        output = fresh_directory(output)
        # Keep only a receipt. Delete synthetic models/embeddings with the temp directory.
        receipt = {
            "passed": True,
            "synthetic_only": True,
            "real_hotpot_training": False,
            "bge_m3_loaded": False,
            "gpu_used": False,
            "runs": 18,
            "seeds": config["seeds"],
            "synthetic_queries": 100,
            "feature_dimension": 1024,
            "archive_order": "reversed to verify ID joins",
            "full_test_n_per_seed": 20,
            "methods_per_seed": 11,
            "report_generated": True,
            "all_checkpoints_cpu": True,
            "temporary_models_and_embeddings_deleted_on_exit": True,
        }
        write_json(output / "smoke_receipt.json", receipt)
        print(receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh receipt directory")
    smoke(parser.parse_args().output)


if __name__ == "__main__":
    main()
