import copy
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_soft_oracle import (  # noqa: E402
    Parameters,
    Question,
    allocate_output,
    build_label,
    load_data,
    normalize_cost,
    parse_json,
    stable_softmax,
    write_csv,
)
from analyze_soft_oracle import feasibility, summarize  # noqa: E402


def params(**changes):
    data = dict(
        epsilon=0.1,
        tau=0.5,
        lambda_=0.1,
        temperature=0.1,
        evidence_metric="sentence_recall_lower",
        cost_metric="context_tokens",
        scheme="D",
    )
    data.update(changes)
    return Parameters(**data)


def evidence_row(lower, upper=None):
    upper = lower if upper is None else upper
    return {
        "sentence_recall_lower": lower,
        "sentence_recall_upper": upper,
        "paragraph_recall_lower": lower,
        "paragraph_recall_upper": upper,
        "sentence_unknown": int(upper > lower),
    }


def question(f1=(1.0, 0.95, 0.8), ev=(0.5, 1.0, 1.0), tokens=(10, 20, 100), upper=None):
    ks = tuple(range(1, len(f1) + 1))
    best = max(f1)
    return Question(
        "q1",
        "bridge",
        "Example?",
        "Answer",
        ks,
        tuple(f1),
        tuple(int(f == 1) for f in f1),
        {
            "context_tokens": tuple(tokens),
            "prompt_tokens": tuple(t + 1000 for t in tokens),
            "completion_tokens": tuple(2 for _ in tokens),
            "total_tokens": tuple(t + 1002 for t in tokens),
        },
        tuple(evidence_row(v, upper[i] if upper else None) for i, v in enumerate(ev)),
        tuple("Answer" for _ in f1),
        ks[list(f1).index(best)],
    )


class LabelTests(unittest.TestCase):
    def test_both_thresholds_inclusive(self):
        q = question(f1=(1.0, 0.9, 0.899), ev=(0.499, 0.5, 1.0))
        r = build_label(q, params())
        self.assertEqual(r["valid_candidates"], (2,))

    def test_roundoff_at_threshold(self):
        r = build_label(question(f1=(0.3, 0.2, 0.0), ev=(0, 0.5, 1)), params(epsilon=0.1))
        self.assertEqual(r["valid_candidates"], (2,))

    def test_smax_not_recomputed_after_evidence(self):
        r = build_label(question(f1=(1, 0.5, 0.4), ev=(0, 1, 1)), params(epsilon=0.1))
        self.assertEqual(r["label_status"], "empty_candidates")

    def test_multi_softmax_matches_utility_difference(self):
        r = build_label(question(), params())
        self.assertEqual(r["valid_candidates"], (1, 2))
        self.assertAlmostEqual(r["soft_label"][0] / r["soft_label"][1], math.exp((0.99 - 0.93) / 0.1))
        self.assertAlmostEqual(sum(r["soft_label"]), 1.0)
        self.assertEqual(r["soft_label"][2], 0.0)

    def test_singleton_is_onehot(self):
        r = build_label(question(ev=(0, 1, 0)), params())
        self.assertEqual(r["soft_label"], (0.0, 1.0, 0.0))
        self.assertEqual(r["entropy"], 0)
        self.assertIsNone(r["entropy_normalized_candidates"])

    def test_empty_is_null_without_fallback(self):
        r = build_label(question(ev=(0, 0, 0)), params())
        self.assertEqual(r["label_status"], "empty_candidates")
        self.assertIsNone(r["soft_label"])
        self.assertIsNone(r["argmax_soft_label_k"])

    def test_all_zero_even_if_candidates_exist(self):
        for scheme in "ABCD":
            r = build_label(question(f1=(0, 0, 0)), params(scheme=scheme))
            self.assertEqual(r["label_status"], "all_zero_f1")
            self.assertIsNone(r["soft_label"])
            self.assertEqual(r["valid_candidate_count"], 3)

    def test_all_zero_has_priority_over_empty(self):
        r = build_label(question(f1=(0, 0, 0), ev=(0, 0, 0)), params())
        self.assertEqual(r["label_status"], "all_zero_f1")
        self.assertEqual(r["valid_candidate_count"], 0)

    def test_unknown_not_treated_as_confirmed_hit(self):
        q = question(f1=(1, 1, 1), ev=(0.5, 0, 0), upper=(1, 0, 0))
        lower = build_label(q, params(tau=1))
        upper = build_label(q, params(tau=1, evidence_metric="sentence_recall_upper"))
        self.assertEqual(lower["label_status"], "empty_candidates")
        self.assertEqual(upper["valid_candidates"], (1,))
        self.assertTrue(lower["has_unknown_sentence_evidence"])

    def test_numeric_stability(self):
        probabilities = stable_softmax((10000, 9999, -10000), (True, True, False), 0.000001)
        self.assertEqual(probabilities, (1, 0, 0))

    def test_tiny_temperature_and_huge_utility(self):
        p = stable_softmax((1e300, 1e300, -1e300), (True, True, True), 1e-300)
        self.assertEqual(p, (0.5, 0.5, 0))

    def test_non_candidates_exact_zero(self):
        p = stable_softmax((100000, -100000, -100000), (False, True, True), 0.1)
        self.assertEqual(p[0], 0)
        self.assertEqual(p[1:], (0.5, 0.5))

    def test_maximum_over_all_k_for_cost(self):
        r = build_label(question(), params())
        self.assertEqual(r["normalized_cost_curve"], (0.1, 0.2, 1.0))

    def test_parameter_runs_do_not_pollute_state(self):
        q = question()
        before = copy.deepcopy(q)
        a = build_label(q, params())
        build_label(q, params(tau=1, epsilon=0, lambda_=100, temperature=0.01))
        b = build_label(q, params())
        self.assertEqual(a, b)
        self.assertEqual(q, before)

    def test_ablation_masks(self):
        q = question(f1=(1, 0.95, 0.1), ev=(0, 1, 1))
        actual = {s: build_label(q, params(scheme=s))["valid_candidates"] for s in "ABCD"}
        self.assertEqual(actual, {"A": (1, 2, 3), "B": (1, 2), "C": (2, 3), "D": (2,)})

    def test_tau_zero_equals_no_evidence_constraint(self):
        q = question()
        self.assertEqual(build_label(q, params(tau=0))["soft_label"], build_label(q, params(scheme="B"))["soft_label"])

    def test_temperature_preserves_argmax(self):
        q = question()
        self.assertEqual(
            {build_label(q, params(temperature=t))["argmax_soft_label_k"] for t in (0.00001, 0.1, 1000)}, {1}
        )

    def test_tied_utility_selects_smallest_k(self):
        r = build_label(question(f1=(1, 1, 1)), params(lambda_=0))
        self.assertEqual(r["argmax_soft_label_k"], 1)
        self.assertTrue(all(abs(v - 1 / 3) < 1e-12 for v in r["soft_label"]))

    def test_bad_parameters(self):
        for changes in (
            {"epsilon": -0.1},
            {"tau": 1.1},
            {"lambda_": -1},
            {"temperature": 0},
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"scheme": "X"},
            {"cost_metric": "missing"},
            {"evidence_metric": "raw_retrieval"},
            {"tau": "0.5"},
            {"lambda_": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                params(**changes)

    def test_bad_costs(self):
        for values in ((0, 0), (-1, 1), (float("nan"), 1), (float("inf"), 1), (), (True, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_cost(values)

    def test_normalize_cost(self):
        self.assertEqual(normalize_cost((0, 10, 20)), (0, 0.5, 1))

    def test_alternative_token_cost(self):
        q = question()
        r = build_label(q, params(cost_metric="prompt_tokens"))
        self.assertAlmostEqual(r["normalized_cost_curve"][0], 1010 / 1100)
        with self.assertRaises(ValueError):
            build_label(
                replace(q, tokens={**q.tokens, "completion_tokens": (0, 0, 0)}), params(cost_metric="completion_tokens")
            )

    def test_summary_same_subset_fixed_baseline(self):
        q1 = question()
        q2 = replace(question(f1=(0.1, 0.1, 1)), question_id="q2")
        qs = [q1, q2]
        labels = [build_label(q, params()) for q in qs]
        s = summarize(qs, labels, [1])
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["fixed_k"][0]["f1"], 0.1)
        self.assertEqual(s["hard_oracle_k_hist"]["3"], 1)

    def test_feasibility_partition_denominators(self):
        qs = [question(), question(ev=(0, 0, 0)), question(f1=(0, 0, 0))]
        labels = [build_label(q, params()) for q in qs]
        f = feasibility(qs, labels, [0, 1, 2], params(), "combined")
        self.assertEqual((f["valid_n"], f["empty_nonzero_n"], f["all_zero_f1_n"]), (1, 1, 1))
        self.assertEqual(sum(f["excluded_partition_all"].values()), 9)
        self.assertEqual(sum(f["excluded_partition_nonzero"].values()), 6)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.paths = {
            name: self.base / (name + ext)
            for name, ext in (("curves", ".jsonl"), ("dataset", ".jsonl"), ("questions", ".csv"), ("evidence", ".csv"))
        }
        qid = "q1"
        self.rows = {
            "curves": [
                {
                    "question_id": qid,
                    "type": "bridge",
                    "scores": {str(k): {"f1": 1, "em": 1, "prediction": "yes"} for k in (1, 2)},
                    "best_k": 1,
                    "tied_best_ks": [1, 2],
                    "context_tokens": {"1": 10, "2": 20},
                    "api_token_usage": {
                        str(k): {
                            "qid": qid,
                            "k": k,
                            "prompt_tokens": 100 + k,
                            "completion_tokens": 2,
                            "total_tokens": 102 + k,
                            "prediction": "yes",
                        }
                        for k in (1, 2)
                    },
                }
            ],
            "dataset": [{"question_id": qid, "type": "bridge", "question_text": "Example?"}],
            "questions": [
                {
                    "qid": qid,
                    "question_id": qid,
                    "type": "bridge",
                    "question": "Example?",
                    "gold_answer": "yes",
                    "oracle_k": 1,
                    "tied_best_ks": "[1,2]",
                    "f1_curve": "[1,1]",
                    "em_curve": "[1,1]",
                    "k1_f1": 1,
                    "k1_em": 1,
                    "max_f1": 1,
                    "max_em": 1,
                    "all_zero_f1": "False",
                }
            ],
            "evidence": [],
        }
        for k in (1, 2):
            row = {
                "qid": qid,
                "question_id": qid,
                "k": k,
                "type": "bridge",
                "oracle_k": 1,
                "f1": 1,
                "em": 1,
                "unknown_gold_facts": "[]",
                "unknown_gold_titles": "[]",
            }
            for unit in ("sentence", "paragraph"):
                row.update(
                    {
                        unit + "_hit": 1,
                        unit + "_total": 2,
                        unit + "_unknown": 0,
                        unit + "_status": "partial",
                        unit + "_recall_lower": 0.5,
                        unit + "_recall_upper": 0.5,
                    }
                )
            self.rows["evidence"].append(row)

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        for name, rows in self.rows.items():
            if self.paths[name].suffix == ".jsonl":
                self.paths[name].write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            else:
                write_csv(self.paths[name], rows)
        return load_data(self.paths, (1, 2), {"bridge": 1})

    def test_valid_data_join(self):
        qs, v = self.load()
        self.assertEqual((len(qs), v["pair_count"]), (1, 2))

    def test_missing_pair(self):
        self.rows["evidence"].pop()
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.load()

    def test_extra_k(self):
        self.rows["evidence"][1]["k"] = 3
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.load()

    def test_duplicate_pair(self):
        self.rows["evidence"].append(copy.deepcopy(self.rows["evidence"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()

    def test_duplicate_query(self):
        self.rows["curves"].append(copy.deepcopy(self.rows["curves"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()

    def test_qid_sets_differ(self):
        self.rows["dataset"][0]["question_id"] = "other"
        with self.assertRaisesRegex(ValueError, "ID sets"):
            self.load()

    def test_qid_alias_mismatch(self):
        self.rows["evidence"][0]["qid"] = "other"
        with self.assertRaisesRegex(ValueError, "aliases"):
            self.load()

    def test_missing_curve_k(self):
        del self.rows["curves"][0]["scores"]["2"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.load()

    def test_f1_em_conflicts(self):
        for metric in ("f1", "em"):
            self.rows["evidence"][0][metric] = 0
            with self.subTest(metric=metric), self.assertRaisesRegex(ValueError, "Mismatch"):
                self.load()
            self.rows["evidence"][0][metric] = 1

    def test_bad_recall_bounds(self):
        self.rows["evidence"][0]["sentence_recall_lower"] = 1.1
        with self.assertRaisesRegex(ValueError, "range"):
            self.load()

    def test_inconsistent_evidence_counts(self):
        self.rows["evidence"][0]["sentence_hit"] = 3
        with self.assertRaisesRegex(ValueError, "counts"):
            self.load()

    def test_inconsistent_evidence_status(self):
        self.rows["evidence"][0]["sentence_status"] = "unknown"
        with self.assertRaisesRegex(ValueError, "status"):
            self.load()

    def test_unknown_diagnostic_preserved(self):
        row = self.rows["evidence"][0]
        row.update(
            sentence_unknown=1, sentence_recall_upper=1, sentence_status="unknown", unknown_gold_facts='[["title",0]]'
        )
        qs, _ = self.load()
        self.assertEqual(qs[0].evidence[0]["sentence_unknown"], 1)
        self.assertEqual(qs[0].evidence[0]["sentence_recall_lower"], 0.5)

    def test_missing_tokens(self):
        del self.rows["curves"][0]["context_tokens"]["1"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.load()

    def test_abnormal_tokens(self):
        self.rows["curves"][0]["context_tokens"]["1"] = -1
        with self.assertRaisesRegex(ValueError, "range"):
            self.load()

    def test_noninteger_tokens(self):
        self.rows["curves"][0]["context_tokens"]["1"] = 1.5
        with self.assertRaisesRegex(ValueError, "integer"):
            self.load()

    def test_api_token_total_conflict(self):
        self.rows["curves"][0]["api_token_usage"]["1"]["total_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "Token total"):
            self.load()

    def test_missing_type(self):
        self.rows["dataset"][0]["type"] = "unknown"
        with self.assertRaisesRegex(ValueError, "type counts"):
            self.load()

    def test_type_mismatch(self):
        self.rows["evidence"][0]["type"] = "comparison"
        with self.assertRaisesRegex(ValueError, "type mismatch"):
            self.load()

    def test_hard_oracle_conflict(self):
        self.rows["curves"][0]["best_k"] = 2
        with self.assertRaisesRegex(ValueError, "Hard Oracle"):
            self.load()

    def test_missing_file(self):
        with self.assertRaisesRegex(ValueError, "Missing offline input"):
            load_data(self.paths, (1, 2), {"bridge": 1})


class OutputSafetyTests(unittest.TestCase):
    def test_refuse_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "keep.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                allocate_output(d)
            self.assertEqual(marker.read_text(), "keep")

    def test_refuse_duplicate_json_object_keys(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            parse_json('{"scores":{"1":1,"1":2}}')


if __name__ == "__main__":
    unittest.main()
