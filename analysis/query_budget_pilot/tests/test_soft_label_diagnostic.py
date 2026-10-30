"""Offline mathematical/IO tests only: no torch, model fitting, or embedding fixtures."""

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import diagnose_soft_labels as diagnostic  # noqa: E402


def distribution(*values):
    return list(values) + [0.0] * (15 - len(values))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def offline_fixture(root):
    """Handwritten saved-probability fixture; no embedding generation or training."""
    prepared, matrix = root / "prepared", root / "matrix"
    prepared.mkdir()
    matrix.mkdir()
    labels = {}
    for i in range(20):
        valid = i not in (0, 15, 19)
        labels[str(i)] = {
            "question_id": str(i),
            "type": "bridge",
            "has_valid_label": valid,
            "targets": {
                "original_soft": {
                    "status": "valid" if valid else "all_zero_f1",
                    "probabilities": distribution(0.8, 0.2) if valid else None,
                }
            },
        }
    write_jsonl(prepared / "labels.jsonl", labels.values())
    split = {
        "seed": 13,
        "partitions": {
            "train": [str(i) for i in range(14)],
            "validation": ["14", "15"],
            "test": [str(i) for i in range(16, 20)],
        },
    }
    (prepared / "splits").mkdir()
    for seed in diagnostic.SEEDS:
        diagnostic.write_json(prepared / "splits" / "seed_{}.json".format(seed), dict(split, seed=seed))
    artifacts = {"labels.jsonl": diagnostic.sha256(prepared / "labels.jsonl")}
    artifacts.update(
        {
            "splits/seed_{}.json".format(seed): diagnostic.sha256(prepared / "splits" / "seed_{}.json".format(seed))
            for seed in diagnostic.SEEDS
        }
    )
    diagnostic.write_json(
        prepared / "metadata.json",
        {
            "query_count": 20,
            "pilot_configuration": {"labels": {"original_soft": diagnostic.PARAMETERS}},
            "artifacts_sha256": artifacts,
        },
    )
    runs = []
    membership = {qid: part for part, ids in split["partitions"].items() for qid in ids}
    for seed in diagnostic.SEEDS:
        for model in diagnostic.MODELS:
            relative = "seed_{}/{}_original_soft".format(seed, model)
            runs.append(relative)
            run = matrix / relative
            run.mkdir(parents=True)
            diagnostic.write_json(
                run / "training.json",
                {
                    "seed": seed,
                    "model": model,
                    "label": "original_soft",
                    "best_epoch": 1,
                    "configuration": {"labels": {"original_soft": diagnostic.PARAMETERS}},
                    "input_sha256": {
                        Path(p).name: diagnostic.sha256(prepared / p)
                        for p in ("metadata.json", "labels.jsonl", "splits/seed_{}.json".format(seed))
                    },
                },
            )
            write_jsonl(
                run / "evaluation/predictions.jsonl",
                [
                    {
                        "question_id": qid,
                        "partition": membership[qid],
                        "type": "bridge",
                        "has_valid_label": row["has_valid_label"],
                        "probabilities": distribution(0.8, 0.2),
                        "predicted_k": 1,
                    }
                    for qid, row in reversed(list(labels.items()))
                ],
            )
    diagnostic.write_json(matrix / "matrix.json", {"status": "complete", "seeds": diagnostic.SEEDS, "runs": runs})
    return prepared, matrix, labels, split


class DiagnosticMathTests(unittest.TestCase):
    def test_analytic_kl_ce_entropy_and_direction(self):
        p, q = [distribution(0.5, 0.5)], [distribution(0.25, 0.75)]
        kl, ce, entropy = diagnostic.loss_vectors(p, q)
        self.assertAlmostEqual(ce[0], -0.5 * math.log(0.25) - 0.5 * math.log(0.75))
        self.assertAlmostEqual(entropy[0], math.log(2))
        self.assertAlmostEqual(kl[0], 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75))
        self.assertNotAlmostEqual(kl[0], diagnostic.loss_vectors(q, p)[0][0])
        self.assertAlmostEqual(ce[0], kl[0] + entropy[0])

    def test_zero_target_and_zero_prediction_are_finite(self):
        p, q = [distribution(1)], [distribution(0, 1)]
        kl, ce, entropy = diagnostic.loss_vectors(p, q)
        expected = -math.log(diagnostic.EPSILON / (1 + 14 * diagnostic.EPSILON))
        self.assertAlmostEqual(kl[0], expected)
        self.assertEqual(entropy[0], 0)
        self.assertEqual(kl[0], ce[0])
        self.assertLess(abs(diagnostic.loss_vectors(p, p)[0][0]), 1e-10)

    def test_bad_probabilities_rejected_and_float_roundoff_normalized(self):
        for bad in (
            [distribution(0)],
            [distribution(-0.1, 1.1)],
            [[float("nan")] * 15],
            [[float("inf")] * 15],
            [[0.5, 0.5]],
            [distribution(0.4, 0.4)],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                diagnostic.probabilities(bad)
        np.testing.assert_allclose(diagnostic.probabilities([distribution(0.50000001, 0.5)]).sum(axis=1), [1])

    def test_train_mean_excludes_invalid_validation_and_test(self):
        labels = {
            qid: {
                "has_valid_label": qid != "invalid",
                "targets": {"original_soft": {"probabilities": distribution(*v) if qid != "invalid" else None}},
            }
            for qid, v in (("a", (1, 0)), ("b", (0, 1)), ("invalid", ()), ("val", (0, 1)), ("test", (0, 1)))
        }
        split = {"partitions": {"train": ["a", "b", "invalid"], "validation": ["val"], "test": ["test"]}}
        ids, mean = diagnostic.train_mean(labels, split)
        self.assertEqual(ids, ["a", "b"])
        np.testing.assert_array_equal(mean, distribution(0.5, 0.5))
        labels["val"]["targets"]["original_soft"]["probabilities"] = distribution(1)
        labels["test"]["targets"]["original_soft"]["probabilities"] = distribution(1)
        np.testing.assert_array_equal(diagnostic.train_mean(labels, split)[1], mean)
        # The arithmetic mean minimizes total CE among constant distributions.
        target = np.asarray([distribution(1), distribution(0, 1)])
        mean_ce = diagnostic.loss_vectors(target, np.tile(mean, (2, 1)))[1].mean()
        for v in (0.1, 0.25, 0.75, 0.9):
            self.assertLess(mean_ce, diagnostic.loss_vectors(target, [distribution(v, 1 - v)] * 2)[1].mean())

    def test_correlations_ties_and_constant_undefined(self):
        np.testing.assert_array_equal(diagnostic.average_ranks([10, 10, 20, 40]), [1.5, 1.5, 3, 4])
        self.assertAlmostEqual(diagnostic.correlations([1, 2, 3], [9, 4, 1])["spearman"], -1)
        tied = diagnostic.correlations([1, 1, 2, 3], [4, 4, 5, 6])
        self.assertAlmostEqual(tied["spearman"], 1)
        self.assertEqual(diagnostic.correlations([1, 2, 3], [0.5] * 3), {"pearson": None, "spearman": None})

    def test_tail_boundaries_diversity_and_query_specific_gain(self):
        p = np.zeros((2, 15))
        p[0, 4], p[1, 7] = 1, 1  # Exactly k=5 and k=8.
        mean = p.mean(axis=0)
        baseline = np.tile(mean, (2, 1))
        baseline_kl = diagnostic.loss_vectors(p, baseline)[0]
        row, per_query, std = diagnostic.summarize(13, "linear", p, p, mean, baseline_kl)
        self.assertEqual(per_query[0]["target_mass_ge5"], 1)
        self.assertEqual(per_query[0]["target_mass_ge8"], 0)
        self.assertEqual(per_query[1]["target_mass_ge8"], 1)
        self.assertAlmostEqual(row["query_tv_to_own_mean"], 0.5)
        self.assertAlmostEqual(row["query_kl_gain_vs_own_mean"], math.log(2))
        self.assertEqual(row["win_fraction"], 1)
        row, _, std = diagnostic.summarize(13, "mean_distribution", p, baseline, mean, baseline_kl)
        self.assertEqual(row["query_tv_to_own_mean"], 0)
        self.assertEqual(row["kl_gain"], 0)
        self.assertEqual(row["query_kl_gain_vs_own_mean"], 0)
        self.assertEqual(row["tie_fraction"], 1)
        self.assertIsNone(row["mass_ge8_pearson"])
        np.testing.assert_array_equal(std, np.zeros(15))

    def test_full_tail_mass_has_exact_ties_despite_float_roundoff(self):
        p = np.zeros((20, 15))
        p[:, 7:] = np.random.RandomState(42).uniform(size=(20, 8))
        p /= p.sum(axis=1, keepdims=True)
        np.testing.assert_array_equal(diagnostic.tail_mass(p, 8), np.ones(20))
        np.testing.assert_array_equal(diagnostic.average_ranks(diagnostic.tail_mass(p, 8)), np.full(20, 10.5))
        self.assertEqual(diagnostic.correlations([1, 1 + 1e-15, 1], [1, 2, 3]), {"pearson": None, "spearman": None})

    def test_seed_aggregation_uses_sample_sd_and_keeps_null(self):
        rows = [
            {"seed": seed, "method": method, "best_epoch": None, "kl": value, "corr": None}
            for seed, value in zip(diagnostic.SEEDS, (1, 2, 3))
            for method in diagnostic.METHODS
        ]
        for row in diagnostic.aggregate(rows):
            self.assertEqual(row["kl"], {"mean": 2.0, "std": 1.0, "n": 3})
            self.assertEqual(row["corr"], {"mean": None, "std": None, "n": 0})


class DiagnosticIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="offline_soft_label_test_")
        self.root = Path(self.temp.name)
        self.prepared, self.matrix, self.labels, self.split = offline_fixture(self.root)
        self.run = self.matrix / "seed_13/linear_original_soft"
        self.prediction_path = self.run / "evaluation/predictions.jsonl"
        self.test_ids = ["16", "17", "18"]

    def tearDown(self):
        self.temp.cleanup()

    def load_prediction(self):
        return diagnostic.load_prediction(self.run, self.prepared, 13, "linear", self.labels, self.split, self.test_ids)

    def test_join_by_id_with_reordered_saved_predictions(self):
        rows = list(diagnostic.read_index(self.prediction_path).values())
        for row in rows:
            if row["question_id"] == "17":
                row["probabilities"], row["predicted_k"] = distribution(0.2, 0.8), 2
        write_jsonl(self.prediction_path, rows)
        prediction, epoch = self.load_prediction()
        self.assertEqual(epoch, 1)
        np.testing.assert_array_equal(prediction[:, :2], [[0.8, 0.2], [0.2, 0.8], [0.8, 0.2]])

    def test_missing_test_prediction_is_error_without_shrinking_cohort(self):
        rows = [r for qid, r in diagnostic.read_index(self.prediction_path).items() if qid != "17"]
        write_jsonl(self.prediction_path, rows)
        with self.assertRaisesRegex(ValueError, "Missing valid Test predictions"):
            self.load_prediction()

    def test_duplicate_prediction_and_wrong_partition_rejected(self):
        rows = list(diagnostic.read_index(self.prediction_path).values())
        write_jsonl(self.prediction_path, rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load_prediction()
        rows[0]["partition"] = "validation"
        write_jsonl(self.prediction_path, rows)
        with self.assertRaisesRegex(ValueError, "partition mismatch"):
            self.load_prediction()

    def test_changed_labels_and_wrong_training_identity_rejected(self):
        training_path = self.run / "training.json"
        training = diagnostic.read_json(training_path)
        training["label"] = "sharper_soft"
        training_path.write_text(json.dumps(training))
        with self.assertRaisesRegex(ValueError, "Run identity"):
            self.load_prediction()
        with (self.prepared / "labels.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            diagnostic.validate_data(self.prepared, diagnostic.SEEDS)

    def test_missing_prediction_file_report_and_cli_exit_2(self):
        self.prediction_path.unlink()
        output = self.root / "missing_report"
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "diagnose_soft_labels.py"),
                "--prepared",
                str(self.prepared),
                "--matrix",
                str(self.matrix),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(str(self.prediction_path), result.stderr)
        audit = diagnostic.read_json(output / "input_audit.json")
        self.assertEqual(audit["status"], "missing_inputs")
        self.assertIn(str(self.prediction_path), audit["missing_files"])
        self.assertFalse((output / "summary.json").exists())
        self.assertIn("未重新训练", (output / "report.md").read_text())

    def test_pipeline_same_cohort_read_only_and_refuses_overwrite(self):
        files = [p for p in self.root.rglob("*") if p.is_file()]
        before = {str(p): diagnostic.sha256(p) for p in files}
        output = self.root / "diagnostic"
        with patch.object(diagnostic, "make_figures"):
            result = diagnostic.run_diagnostic(self.prepared, self.matrix, output)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["inputs_unchanged"])
        self.assertEqual(before, {str(p): diagnostic.sha256(p) for p in files})
        summary = diagnostic.read_json(output / "summary.json")
        self.assertEqual(len(summary["per_seed"]), 9)
        self.assertTrue(all(row["n_test"] == 3 for row in summary["per_seed"]))
        with (output / "per_query.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for seed in diagnostic.SEEDS:
            for method in diagnostic.METHODS:
                self.assertEqual(
                    [r["question_id"] for r in rows if r["seed"] == str(seed) and r["method"] == method], self.test_ids
                )
        with self.assertRaisesRegex(ValueError, "Output exists"):
            diagnostic.run_diagnostic(self.prepared, self.matrix, output)
        with self.assertRaisesRegex(ValueError, "outside existing"):
            diagnostic.run_diagnostic(self.prepared, self.matrix, self.matrix / "new")


if __name__ == "__main__":
    unittest.main()
