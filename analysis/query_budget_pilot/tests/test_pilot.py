import argparse
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import (  # noqa: E402
    LABELS,
    by_id,
    load_prepared,
    read_json,
    read_rows,
    sha256,
    stratified_split,
    target_labels,
    validate_queries,
    validate_split,
    write_json,
    write_jsonl,
)
from evaluate import evaluate, summarize  # noqa: E402
from extract_embeddings import extract, load_verified, read_queries, validate_arrays  # noqa: E402
from train import fit, load_inputs, paths_from_config, soft_cross_entropy  # noqa: E402
from synthetic import fixture, question, refresh_manifest  # noqa: E402


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="query_budget_test_")
        self.root = Path(self.temp.name)
        self.config, self.prepared, self.embeddings, self.metadata, self.features, self.labels = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def train(self, path="run", kind="linear", label="minimum_feasible", overrides=None):
        return fit(
            self.config, kind, label, 13, self.root / path, self.prepared, self.embeddings, self.metadata, overrides
        )

    def test_query_input_rejects_duplicates_empty_and_extra_fields(self):
        rows = read_rows(self.prepared / "queries.jsonl")
        validate_queries(rows, 100)
        for bad in (
            rows + rows[:1],
            [dict(r, answer="leak") for r in rows],
            [dict(rows[0], question_text=" ")] + rows[1:],
        ):
            with self.assertRaises(ValueError):
                validate_queries(bad, 100)
        path = self.root / "duplicate_keys.jsonl"
        path.write_text('{"question_id":"x","question_id":"y","question_text":"q"}\n')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            read_queries(path, 1)

    def test_labels_match_reused_oracle_and_minimum_cost(self):
        from build_soft_oracle import Parameters, build_label

        q = question(12)
        targets = target_labels(q, self.config)
        p = Parameters.from_dict(
            dict(
                epsilon=0.1,
                tau=0.5,
                scheme="D",
                evidence_metric="sentence_recall_lower",
                cost_metric="context_tokens",
                **self.config["labels"]["original_soft"]
            )
        )
        self.assertEqual(targets["original_soft"]["probabilities"], build_label(q, p)["soft_label"])
        # Nonmonotone costs and a cost tie: choose smaller k among true minimum costs.
        q = replace(question(2), tokens={"context_tokens": (90, 80, 70, 5, 5) + (100,) * 10})
        target = target_labels(q, self.config)["minimum_feasible"]["probabilities"]
        self.assertEqual(target.index(1.0) + 1, 4)

    def test_invalid_targets_stay_null(self):
        for i, status in ((0, "all_zero_f1"), (3, "empty_candidates")):
            targets = target_labels(question(i), self.config)
            for name in LABELS:
                self.assertEqual(targets[name], {"status": status, "probabilities": None})

    def test_split_is_deterministic_disjoint_and_preserves_unlabeled_test(self):
        identities = [(qid, r["type"], r["has_valid_label"]) for qid, r in self.labels.items()]
        for seed in self.config["seeds"]:
            split = stratified_split(identities, seed)
            self.assertEqual(split, stratified_split(list(reversed(identities)), seed))
            validate_split(split, self.labels)
            self.assertEqual([len(split["partitions"][p]) for p in ("train", "validation", "test")], [70, 10, 20])
            self.assertTrue(any(not self.labels[qid]["has_valid_label"] for qid in split["partitions"]["test"]))
        bad = stratified_split(identities, 13)
        bad["partitions"]["test"][0] = bad["partitions"]["train"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split(bad, self.labels)

    def test_embedding_archive_reorder_is_joined_by_id(self):
        aligned, _ = load_verified(self.prepared / "queries.jsonl", self.embeddings, self.metadata, 100)
        np.testing.assert_array_equal(aligned, self.features)
        rows = list(reversed(read_rows(self.prepared / "labels.jsonl")))
        (self.prepared / "labels.jsonl").unlink()
        write_jsonl(self.prepared / "labels.jsonl", rows)
        refresh_manifest(self.prepared, self.config)
        _, labels, _, _, aligned = load_inputs(self.config, 13, self.prepared, self.embeddings, self.metadata)
        self.assertEqual(set(labels), set(self.labels))
        for qid in labels:
            self.assertEqual(labels[qid]["type"], self.labels[qid]["type"])
            for name in LABELS:
                np.testing.assert_equal(
                    labels[qid]["targets"][name]["probabilities"], self.labels[qid]["targets"][name]["probabilities"]
                )
        np.testing.assert_array_equal(aligned, self.features)

    def test_invalid_embedding_values_shape_dtype_ids_and_norm(self):
        ids = np.asarray(["a", "b"])
        array = np.zeros((2, 1024), dtype=np.float32)
        array[:, 0] = 1
        validate_arrays(ids, array, ["a", "b"])
        cases = [
            (ids, array.astype(np.float64)),
            (ids, array[:, :12]),
            (np.asarray(["a", "a"]), array),
            (ids, array * 2),
        ]
        nonfinite = array.copy()
        nonfinite[0, 0] = float("nan")
        cases.append((ids, nonfinite))
        for bad_ids, bad_array in cases:
            with self.assertRaises(ValueError):
                validate_arrays(bad_ids, bad_array, ["a", "b"])
        with self.assertRaises(ValueError):
            validate_arrays(ids, array, ["b", "c"])

    def test_embedding_provenance_and_prepared_hashes_checked(self):
        metadata = read_json(self.metadata)
        metadata["input_sha256"] = "incorrect"
        write_json(self.metadata, metadata)
        with self.assertRaisesRegex(ValueError, "input SHA256"):
            load_verified(self.prepared / "queries.jsonl", self.embeddings, self.metadata, 100)
        with (self.prepared / "offline_curves.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            load_prepared(self.prepared)

    def test_missing_embedding_fails_before_any_download_or_output(self):
        with self.assertRaisesRegex(ValueError, "A800 extraction"):
            paths_from_config(self.config, embeddings=self.root / "missing.npz")

    def test_missing_label_id_and_changed_target_configuration_rejected(self):
        config = dict(self.config, labels=dict(self.config["labels"]))
        config["labels"]["sharper_soft"] = {"lambda": 0.3, "temperature": 0.02}
        with self.assertRaisesRegex(ValueError, "Configured targets differ"):
            load_inputs(config, 13, self.prepared, self.embeddings, self.metadata)
        rows = read_rows(self.prepared / "labels.jsonl")[1:]
        (self.prepared / "labels.jsonl").unlink()
        write_jsonl(self.prepared / "labels.jsonl", rows)
        refresh_manifest(self.prepared, self.config)
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            load_prepared(self.prepared)

    def test_upload_package_is_minimal_hashed_and_refuses_overwrite(self):
        from pack_a800_job import pack

        config = dict(self.config, prepared=str(self.prepared))
        write_json(
            self.prepared / "query_metadata.json",
            {"source_sha256": "synthetic_only", "queries_sha256": sha256(self.prepared / "queries.jsonl")},
        )
        output = self.root / "upload"
        pack(config, output)
        self.assertEqual(
            {p.name for p in output.iterdir()},
            {
                "extract_embeddings.py",
                "queries.jsonl",
                "requirements-embedding.txt",
                "README.md",
                "a800_env.sh",
                "job_manifest.json",
            },
        )
        manifest = read_json(output / "job_manifest.json")
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(sha256(output / name), digest)
        self.assertTrue(all(set(r) == {"question_id", "question_text"} for r in read_rows(output / "queries.jsonl")))
        with self.assertRaises(FileExistsError):
            pack(config, output)

    def test_soft_cross_entropy_matches_known_distribution(self):
        targets = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
        logits = torch.log(torch.tensor([[0.5, 0.5], [0.8, 0.2]]))
        loss = soft_cross_entropy(logits, targets)
        self.assertAlmostEqual(float(loss), float((-np.log(0.5) - np.log(0.8)) / 2), places=6)

    def test_training_reproducible_best_checkpoint_and_cpu(self):
        result = self.train()
        self.train("repeat")
        a = torch.load(self.root / "run" / "best.pt", map_location="cpu", weights_only=True)
        b = torch.load(self.root / "repeat" / "best.pt", map_location="cpu", weights_only=True)
        self.assertTrue(all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"]))
        history = read_json(self.root / "run" / "history.json")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["parameter_count"], 1024 * 15 + 15)
        self.assertEqual(result["best_validation_loss"], history[result["best_epoch"] - 1]["validation_loss"])
        self.assertLess(result["supervised_counts"]["train"], result["full_partition_counts"]["train"])
        self.assertTrue(all(t.device.type == "cpu" for t in a["state_dict"].values()))
        with self.assertRaises(FileExistsError):
            self.train()

    def test_early_stopping_uses_only_validation(self):
        result = self.train(overrides={"epochs": 10, "min_delta": 100, "patience": 2})
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs_run"], 3)

    def test_changing_test_targets_and_scores_cannot_change_training(self):
        self.train()
        split = read_json(self.prepared / "splits" / "seed_13.json")
        test = set(split["partitions"]["test"])
        labels = read_rows(self.prepared / "labels.jsonl")
        for row in labels:
            if row["question_id"] in test and row["has_valid_label"]:
                for target in row["targets"].values():
                    target["probabilities"] = [0.0] * 14 + [1.0]
        curves = read_rows(self.prepared / "offline_curves.jsonl")
        for row in curves:
            if row["question_id"] in test:
                row["f1"] = [0.0] * 15
        for name, rows in (("labels.jsonl", labels), ("offline_curves.jsonl", curves)):
            (self.prepared / name).unlink()
            write_jsonl(self.prepared / name, rows)
        refresh_manifest(self.prepared, self.config)
        self.train("changed_test")
        a = torch.load(self.root / "run" / "best.pt", map_location="cpu", weights_only=True)["state_dict"]
        b = torch.load(self.root / "changed_test" / "best.pt", map_location="cpu", weights_only=True)["state_dict"]
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_evaluation_full_test_and_saved_curve_lookup(self):
        self.train(kind="mlp", label="sharper_soft")
        result = evaluate(self.root / "run")
        rows = [
            r for r in result["metrics"] if (r["partition"], r["group"], r["cohort"]) == ("test", "combined", "all")
        ]
        self.assertEqual(len(rows), 6)  # Model, four fixed k, full-k Oracle reference.
        self.assertTrue(all(r["n"] == 20 and r["valid_label_n"] < 20 for r in rows))
        curves = by_id(read_rows(self.prepared / "offline_curves.jsonl"), "curves")
        expected = np.mean([curves[qid]["f1"][12] for qid in result["test_ids"]])
        self.assertAlmostEqual(next(r["answer_f1"] for r in rows if r["method"] == "fixed_k13"), expected)
        predictions = read_rows(self.root / "run" / "evaluation" / "predictions.jsonl")
        self.assertEqual(len(predictions), 100)
        predicted = {r["question_id"]: r["predicted_k"] for r in predictions}
        manual = summarize(result["test_ids"], predicted, curves, self.labels)
        self.assertEqual(next(r["answer_f1"] for r in rows if r["method"] == "model"), manual["answer_f1"])
        self.assertEqual(next(r["context_tokens"] for r in rows if r["method"] == "model"), manual["context_tokens"])

    def test_standalone_verify_without_repo_or_model_library(self):
        standalone = self.root / "standalone"
        standalone.mkdir()
        script = standalone / "extract_embeddings.py"
        shutil.copyfile(HERE / "extract_embeddings.py", script)
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input",
                str(self.prepared / "queries.jsonl"),
                "--output",
                str(self.embeddings),
                "--metadata",
                str(self.metadata),
                "--expected-count",
                "100",
                "--verify-only",
            ],
            cwd=str(standalone),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"question_id_count": 100', result.stdout)

    def test_fake_encoder_frozen_inference_and_export(self):
        test = self

        class FakeModel(torch.nn.Module):
            def __init__(self, name, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.pooling_mode_cls_token = True
                self.auto_model = types.SimpleNamespace(
                    config=types.SimpleNamespace(_commit_hash="fake", to_dict=lambda: {"synthetic": True})
                )
                test.assertEqual(kwargs["device"], "cpu")

            def __getitem__(self, key):
                return self

            def get_sentence_embedding_dimension(self):
                return 1024

            def tokenizer(self, texts, **kwargs):
                return {"input_ids": [[1, 2, 3] for _ in texts]}

            def encode(self, texts, **kwargs):
                test.assertFalse(self.training)
                test.assertFalse(self.weight.requires_grad)
                test.assertTrue(torch.is_inference_mode_enabled())
                test.assertEqual(kwargs["precision"], "float32")
                test.assertEqual(kwargs["prompt"], "")
                test.assertTrue(kwargs["normalize_embeddings"])
                vectors = np.zeros((len(texts), 1024), dtype=np.float32)
                vectors[:, 0] = 1
                return vectors

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeModel
        args = argparse.Namespace(
            input=self.prepared / "queries.jsonl",
            output=self.root / "fake_export" / "embeddings.npz",
            metadata=None,
            expected_count=100,
            batch_size=16,
            max_length=8192,
            device="cpu",
            model="FAKE_NO_DOWNLOAD",
            cache_folder=None,
            revision=None,
            local_files_only=True,
            normalize=True,
        )
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}), patch(
            "importlib.metadata.version", return_value="synthetic-test"
        ):
            extract(args)
        features, metadata = load_verified(args.input, args.output, expected_count=100)
        self.assertEqual(features.shape, (100, 1024))
        self.assertEqual(metadata["resolved_commit_hash"], "fake")


if __name__ == "__main__":
    unittest.main()
