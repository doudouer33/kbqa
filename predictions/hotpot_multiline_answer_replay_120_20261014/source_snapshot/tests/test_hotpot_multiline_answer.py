import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from analysis.hotpot_multiline_answer_diagnostic.replay import (
    CANDIDATE_CONFIG,
    HISTORICAL_CONFIG,
    NEW_REGEX,
    OLD_REGEX,
    configured,
    extractor_settings,
)
from analysis.hotpot_unicode_json_diagnostic.replay import replay_raw


class HotpotMultilineAnswerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = extractor_settings(configured(HISTORICAL_CONFIG, 1))
        cls.new = extractor_settings(configured(CANDIDATE_CONFIG, 1))

    def test_candidate_only_changes_regex_for_all_15_k_values(self):
        for k in range(1, 16):
            with self.subTest(k=k):
                old, new = configured(HISTORICAL_CONFIG, k), configured(CANDIDATE_CONFIG, k)
                self.assertEqual(new["models"]["extract_answer"]["regex"], NEW_REGEX)
                new["models"]["extract_answer"]["regex"] = OLD_REGEX
                self.assertEqual(new, old)
                self.assertEqual(new["models"]["generate_titles"]["retrieval_count"], k)

    def test_single_line_unicode_yes_no_and_literal_escapes_are_unchanged(self):
        for answer in (
            "yes",
            "no",
            "tennis player",
            "Atlético Mineiro",
            "4 × 100 metre relay",
            "Beyoncé 中文 한국어",
            r"literal \u00e9",
            r"C:\new\test",
        ):
            with self.subTest(answer=answer):
                raw = "Reasoning. So the answer is: " + answer + "."
                old, new = replay_raw(raw, self.old), replay_raw(raw, self.new)
                self.assertEqual(new, old)
                self.assertEqual(new["prediction"], answer)

    def test_multiline_reasoning_finds_final_answer(self):
        for newline in ("\n", "\r\n", "\r"):
            with self.subTest(newline=repr(newline)):
                raw = newline.join(("First reasoning line.", "Another line.", "So the answer is: tennis player."))
                result = replay_raw(raw, self.new)
                self.assertTrue(result["primary_match"])
                self.assertFalse(result["fallback"])
                self.assertEqual(result["prediction"], "tennis player")

    def test_next_line_explanation_is_not_captured(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)):
                result = replay_raw(
                    "Reasoning." + newline + "So the answer is: yes." + newline + "Further explanation.", self.new
                )
                self.assertEqual(result["prediction"], "yes")

    def test_greedy_last_marker_is_kept(self):
        for separator in (" ", "\n"):
            with self.subTest(separator=repr(separator)):
                result = replay_raw("So the answer is: first." + separator + "So the answer is: Beyoncé.", self.new)
                self.assertEqual(result["prediction"], "Beyoncé")

    def test_missing_marker_preserves_original_fallback(self):
        for raw in ("No answer marker.", "First reasoning line.\nSecond reasoning line."):
            with self.subTest(raw=raw):
                old, new = replay_raw(raw, self.old), replay_raw(raw, self.new)
                self.assertEqual(new, old)
                self.assertFalse(new["primary_match"])
                self.assertTrue(new["fallback"])


if __name__ == "__main__":
    unittest.main()
