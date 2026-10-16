"""Synthetic edge cases plus independent checks of the full generated tables."""
import csv
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analyze_evidence", HERE / "analyze_evidence.py")
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def gold(title="Entity", sentences=None, indices=None):
    sentences = sentences or ["Entity was born in Paris in the year 1900.", "Entity died in London in 1980."]
    text = " ".join(sentences)
    return {"title": title, "text": text, "norm": d.norm(text), "compact": d.compact(text),
            "facts": [{"sentence_index": i, "text": sentences[i]} for i in (indices if indices is not None else range(len(sentences)))]}


def doc(title, text):
    return {"title": title, "paragraph_text": text}


def scores(fs, es=None):
    es = es if es is not None else [int(f == 1) for f in fs]
    return {k: {"f1": f, "em": e, "prediction": "answer"} for k, (f, e) in enumerate(zip(fs, es), 1)}


class MatchingTest(unittest.TestCase):
    def test_exact_paragraph_and_sentences(self):
        g = gold()
        ev = d.match_evidence([g], [doc(g["title"], g["text"])])
        self.assertEqual((ev["paragraph_status"], ev["sentence_status"]), ("complete", "complete"))
        self.assertEqual(ev["sentence_hit"], 2)

    def test_title_alone_does_not_confirm_evidence(self):
        ev = d.match_evidence([gold()], [doc("Entity", "An entirely different article about an unrelated entity.")])
        self.assertEqual(ev["paragraph_status"], "unknown")
        self.assertEqual(ev["sentence_status"], "unknown")
        self.assertEqual(ev["sentence_hit"], 0)

    def test_title_punctuation_collision_is_unknown(self):
        g = gold("Money, Money, Money")
        ev = d.match_evidence([g], [doc("Money! Money! Money!", "A different silent film.")])
        self.assertEqual(ev["paragraph_status"], "unknown")
        self.assertNotEqual(d.norm("Money, Money, Money"), d.norm("Money! Money! Money!"))

    def test_nonmatching_title_and_text_are_zero(self):
        ev = d.match_evidence([gold()], [doc("Unrelated", "A separate set of unrelated facts.")])
        self.assertEqual(ev["sentence_status"], "zero")
        self.assertEqual(ev["paragraph_status"], "zero")

    def test_spacing_normalization_is_explicit(self):
        g = gold("Film", ['This is a movie called "Good Bye Lenin!" released in 2003.'])
        text = g["text"].replace('Lenin!"', 'Lenin! "')
        ev = d.match_evidence([g], [doc("Film", text)])
        self.assertEqual(ev["sentence_status"], "complete")
        self.assertEqual(ev["facts"][0]["matches"][0]["method"], "whitespace_only")

    def test_negation_and_number_changes_are_not_fuzzy_matched(self):
        g = gold()
        changed = g["text"].replace("was born", "was not born").replace("1980", "1981")
        ev = d.match_evidence([g], [doc(g["title"], changed)])
        self.assertEqual(ev["sentence_hit"], 0)
        self.assertEqual(ev["sentence_status"], "unknown")

    def test_cross_title_sentence_is_separate_from_paragraph(self):
        g = gold(sentences=["Entity was born in the capital city of Paris in the year 1900.", "Entity died in London in 1980."])
        ev = d.match_evidence([g], [doc("Biography of a colleague", "Other text. " + g["facts"][0]["text"])])
        self.assertEqual(ev["sentence_status"], "partial")
        self.assertEqual(ev["paragraph_status"], "zero")
        self.assertTrue(ev["facts"][0]["matches"][0]["cross_title"])

    def test_short_unanchored_sentence_is_unknown(self):
        g = gold("Short", ["It is true."])
        ev = d.match_evidence([g], [doc("Other", "It is true.")])
        self.assertEqual(ev["sentence_status"], "unknown")
        self.assertEqual(ev["paragraph_hit"], 0)

    def test_gold_sentence_in_title_is_not_body_evidence(self):
        g = gold()
        rendered = d.model_doc(doc(g["facts"][0]["text"], "Unrelated body."))
        self.assertEqual(d.match_evidence([g], [rendered])["sentence_hit"], 0)

    def test_model_truncation_can_hit_paragraph_but_miss_sentence(self):
        lead = " ".join(f"word{i}" for i in range(350)) + "."
        tail = "This final sentence contains the actual answer needed here."
        g = gold("Long", [lead, tail], [1])
        source = doc("Long", g["text"])
        raw = d.match_evidence([g], [source])
        model = d.match_evidence([g], [d.model_doc(source)])
        self.assertEqual(raw["sentence_status"], "complete")
        self.assertEqual(model["paragraph_status"], "complete")
        self.assertEqual(model["full_paragraph_text_count"], 0)
        self.assertEqual(model["sentence_status"], "zero")

    def test_tiny_same_title_prefix_is_unknown(self):
        g = gold()
        ev = d.match_evidence([g], [doc(g["title"], "Entity was")])
        self.assertEqual(ev["paragraph_status"], "unknown")

    def test_partial_hit_plus_unknown_is_unknown_with_bounds(self):
        a, b = gold("One"), gold("Two", ["Two has a completely different set of specific facts here."])
        ev = d.match_evidence([a, b], [doc("One", a["text"]), doc("Two", "Different version.")])
        self.assertEqual(ev["paragraph_status"], "unknown")
        self.assertEqual(ev["paragraph_recall_lower"], .5)
        self.assertEqual(ev["paragraph_recall_upper"], 1)
        self.assertEqual(ev["sentence_hit"], 2)
        self.assertEqual(ev["sentence_unknown"], 1)

    def test_presence_overrides_conflicting_same_title_candidate(self):
        g = gold()
        ev = d.match_evidence([g], [doc("Entity", "Other entity."), doc("Entity", g["text"])])
        self.assertEqual(ev["paragraph_status"], "complete")
        self.assertEqual(ev["sentence_status"], "complete")


class ScoresAndAlignmentTest(unittest.TestCase):
    def test_official_f1_tie_is_not_broken_by_em(self):
        api = d.load_official_scores()
        f1 = api["f1_score"]("beta alpha", "alpha beta")[0]
        self.assertEqual(f1, 1)
        self.assertFalse(api["exact_match_score"]("beta alpha", "alpha beta"))
        f = d.curve_features(scores([1, 1], [0, 1]))
        self.assertEqual(f["oracle_k"], 1)
        self.assertEqual(f["tied_best_count"], 2)

    def test_yes_no_scoring_has_no_partial_credit(self):
        api = d.load_official_scores()
        self.assertEqual(api["f1_score"]("yes indeed", "yes")[0], 0)

    def test_all_zero_partition_and_oracle(self):
        f = d.curve_features(scores([0] * 15))
        self.assertTrue(f["all_zero_f1"] and f["all_em_zero"] and f["flat_f1"] and f["k1_tied_best"])
        self.assertFalse(f["flat_positive_f1"] or f["variable_f1"])
        self.assertEqual((f["oracle_k"], f["delta_f1"], f["tied_best_count"]), (1, 0, 15))

    def test_variable_curve_can_still_have_k1_tied_best(self):
        f = d.curve_features(scores([1, .5, 1]))
        self.assertTrue(f["variable_f1"] and f["k1_tied_best"] and f["any_f1_drop"])
        self.assertEqual(f["delta_f1"], 0)

    def test_positive_f1_with_no_em_is_not_all_zero(self):
        f = d.curve_features(scores([.4, .6, .6], [0, 0, 0]))
        self.assertTrue(f["positive_f1_no_em"])
        self.assertFalse(f["all_zero_f1"])
        self.assertEqual(f["oracle_k"], 2)

    def test_diagnostic_delta_boundaries_handle_float_roundoff(self):
        self.assertEqual(d.delta_bin(.8 - .5), "(0.1,0.3]")
        self.assertEqual(d.delta_bin(.4 - .3), "(0,0.1]")
        self.assertEqual(d.delta_bin(.30001), "(0.3,1]")
        self.assertEqual(d.delta_bin(0), "zero")

    def test_duplicate_qid_rejected(self):
        with self.assertRaises(ValueError):
            d.index_rows([{"qid": "q"}, {"qid": "q"}], "qid", "test")

    def test_json_duplicate_key_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.json"
            path.write_text('{"q":1,"q":2}')
            with self.assertRaises(ValueError):
                d.read_json(path)

    def test_join_uses_ids_not_input_position(self):
        a = d.index_rows([{"qid": "a", "value": 1}, {"qid": "b", "value": 2}], "qid", "a")
        b = d.index_rows([{"qid": "b", "value": 20}, {"qid": "a", "value": 10}], "qid", "b")
        self.assertEqual({q: (a[q]["value"], b[q]["value"]) for q in a}, {"a": (1, 10), "b": (2, 20)})

    def test_empty_stratum_has_null_ratio_not_zero(self):
        self.assertIsNone(d.ratio_count(0, 0)["ratio"])
        self.assertIsNone(d.coverage_summary([], 1)["sentence"]["macro_recall_lower"])

    def test_macro_and_micro_recall_are_distinct(self):
        rows = []
        for i, (total, hits) in enumerate(((1, 1), (3, 0))):
            ev = {}
            for unit in ("paragraph", "sentence"):
                ev.update({unit + "_total": total, unit + "_hit": hits, unit + "_unknown": 0,
                           unit + "_recall_lower": hits / total, unit + "_recall_upper": hits / total,
                           unit + "_status": "complete" if hits == total else "zero"})
            rows.append({"oracle_k": 1, "k1_em": i, "per_k": {1: {**ev, "f1": i, "em": i}}})
        s = d.coverage_summary(rows, 1)
        self.assertEqual(s["sentence"]["macro_recall_lower"], .5)
        self.assertEqual(s["sentence"]["micro_recall_lower"], .25)
        self.assertEqual(sum(v["count"] for v in s["sentence"]["states"].values()), 2)


class NestingTest(unittest.TestCase):
    def test_filtered_lists_can_be_nested_while_short(self):
        lists = {k: list(range(min(k, 5))) for k in d.KS}
        self.assertTrue(d.nesting(lists)["nested"])

    def test_relative_order_change_detected(self):
        lists = {k: ["a", "b", "c"][:k] for k in d.KS}
        lists[15] = ["b", "a", "c"]
        result = d.nesting(lists)
        self.assertFalse(result["nested"])
        self.assertIn([2, 15], result["order_change_pairs"])
        self.assertEqual(result["lost_document_pairs"], [])

    def test_replacement_is_distinguished_from_reordering(self):
        lists = {k: ["a", "b"][:k] for k in d.KS}
        lists[15] = ["a", "c"]
        result = d.nesting(lists)
        self.assertIn([2, 15], result["lost_document_pairs"])
        self.assertNotIn([2, 15], result["order_change_pairs"])


@unittest.skipUnless((HERE / "summary.json").exists(), "Run the offline analysis first for artifact checks")
class FullArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((HERE / "summary.json").read_text())
        with (HERE / "per_question.csv").open() as stream:
            cls.q = list(csv.DictReader(stream))
        with (HERE / "per_k_evidence.csv").open() as stream:
            cls.p = list(csv.DictReader(stream))

    def test_counts_unique_ids_and_complete_cross_product(self):
        self.assertEqual(len(self.q), 1000)
        self.assertEqual(Counter(r["type"] for r in self.q), {"bridge": 500, "comparison": 500})
        self.assertEqual(len({r["qid"] for r in self.q}), 1000)
        pairs = {(r["qid"], int(r["k"])) for r in self.p}
        self.assertEqual(len(self.p), 15000)
        self.assertEqual(pairs, {(r["qid"], k) for r in self.q for k in range(1, 16)})

    def test_independently_reproduce_baseline_and_partitions(self):
        self.assertEqual(sum(int(r["oracle_k"]) == 1 for r in self.q), 704)
        self.assertEqual(sum(r["all_zero_f1"] == "True" for r in self.q), 181)
        self.assertEqual(sum(r["flat_f1"] == "True" for r in self.q), 498)
        self.assertEqual(sum(int(r["tied_best_count"]) > 1 for r in self.q), 917)
        for r in self.q:
            self.assertEqual(sum(r[k] == "True" for k in ("all_zero_f1", "flat_positive_f1", "variable_f1")), 1)
            self.assertEqual(sum(r[k] == "True" for k in ("k1_unique_best", "k1_tied_best", "k1_not_best")), 1)
            self.assertEqual(float(r["delta_f1"]) == 0, int(r["oracle_k"]) == 1)

    def test_csv_and_summary_coverage_agree_for_every_group_and_k(self):
        for group in d.GROUPS:
            for k in d.KS:
                rows = [r for r in self.p if int(r["k"]) == k and (group == "combined" or r["type"] == group)]
                c = self.summary["groups"][group]["coverage_by_k"][str(k)]
                for unit in ("paragraph", "sentence"):
                    counts = Counter(r[unit + "_status"] for r in rows)
                    self.assertEqual(sum(counts.values()), c["n"])
                    for state in d.STATES:
                        self.assertEqual(counts[state], c[unit]["states"][state]["count"])
                    for r in rows:
                        hit, unknown, total = [int(r[unit + "_" + a]) for a in ("hit", "unknown", "total")]
                        self.assertLessEqual(hit + unknown, total)
                        self.assertAlmostEqual(float(r[unit + "_recall_lower"]), hit / total)

    def test_representative_cases_are_unique_and_balanced(self):
        cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text().splitlines()]
        self.assertEqual(len(cases), 48)
        self.assertEqual(len({c["question_id"] for c in cases}), 48)
        self.assertEqual(Counter(c["type"] for c in cases), {"bridge": 24, "comparison": 24})
        qids = {r["qid"] for r in self.q}
        for c in cases:
            self.assertIn(c["question_id"], qids)
            self.assertEqual(len(c["f1_curve"]), 15)
            self.assertEqual(len(c["em_curve"]), 15)
            self.assertTrue(c["observations"] and c["unconfirmed_explanations"])


if __name__ == "__main__":
    unittest.main()
