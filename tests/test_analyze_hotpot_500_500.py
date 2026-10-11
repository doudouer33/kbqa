import unittest

from analysis.analyze_hotpot_500_500 import distribution_statistics, percentile


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


if __name__ == "__main__":
    unittest.main()
