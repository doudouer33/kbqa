import unittest
import json
import tempfile
from pathlib import Path

from analysis.analyze_hotpot_500_500 import distribution_statistics, percentile, score_predictions, load_api_tokens


class AnalyzeHotpotTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertEqual(percentile([0, 10], 0.9), 9)

    def test_distribution_statistics(self):
        summary = distribution_statistics([1, 1, 2, 3])
        self.assertEqual(summary["distribution"]["1"], 2)
        self.assertEqual(summary["distribution"]["15"], 0)
        self.assertEqual(summary["majority_ratio"], 0.5)
        self.assertEqual(summary["mean_best_k"], 1.75)

    def test_official_f1_ties_choose_smallest_k_without_em_priority(self):
        samples = {"q": {"type": "bridge", "answers_objects": [{"spans": ["alpha beta"]}]}}
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for k, answer in [(1, "beta alpha"), (2, "alpha beta")]:
                path = Path(directory) / (str(k) + ".json")
                path.write_text(json.dumps({"q": answer}))
                paths[k] = path
            result = score_predictions(samples, paths)["q"]
        self.assertEqual(result["best_k"], 1)
        self.assertEqual(result["tied_best_ks"], [1, 2])
        self.assertEqual(result["scores"][1]["em"], 0)
        self.assertEqual(result["scores"][2]["em"], 1)
        self.assertEqual(result["scores"][1]["f1"], 1)

    def test_official_yes_no_does_not_award_partial_overlap(self):
        samples = {"q": {"type": "comparison", "answers_objects": [{"spans": ["yes"]}]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.json"
            path.write_text(json.dumps({"q": "yes indeed"}))
            result = score_predictions(samples, {1: path})["q"]
        self.assertEqual(result["scores"][1]["f1"], 0)

    def test_token_join_rejects_duplicates_and_missing_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "token_usage.jsonl"
            row = {"qid": "q", "k": 1, "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "prompt_cropped": False, "finish_reason": "stop"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_api_tokens({"q": {}}, {1: directory / "prediction.json"})
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_api_tokens({"q": {}, "missing": {}}, {1: directory / "prediction.json"})


if __name__ == "__main__":
    unittest.main()
