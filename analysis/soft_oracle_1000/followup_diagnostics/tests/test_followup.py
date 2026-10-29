import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import (  # noqa: E402
    allocate_followup_output,
    budget_records,
    classify_empty,
    compare_budgets,
    empty_transitions,
    min_feasible_k,
    sharpness_record,
    summarize_empty,
    summarize_sharpness,
)
from build_soft_oracle import Parameters, Question, build_label  # noqa: E402


def parameters(**changes):
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


def question(f1=(0.9, 1, 1), evidence=(0.5, 1, 1), tokens=(10, 20, 30), unknown=None, qid="q1", type_="bridge"):
    n = len(f1)
    unknown = [0] * n if unknown is None else unknown
    return Question(
        qid,
        type_,
        "Example?",
        "Answer",
        tuple(range(1, n + 1)),
        tuple(f1),
        tuple(int(f == 1) for f in f1),
        {
            "context_tokens": tuple(tokens),
            "prompt_tokens": tuple(t + 100 for t in tokens),
            "total_tokens": tuple(t + 102 for t in tokens),
        },
        tuple(
            {"sentence_recall_lower": e, "sentence_recall_upper": min(1, e + u * 0.5), "sentence_unknown": u}
            for e, u in zip(evidence, unknown)
        ),
        tuple("Answer" for _ in f1),
        list(f1).index(max(f1)) + 1,
    )


class BudgetTests(unittest.TestCase):
    def test_minimum_uses_actual_tokens(self):
        q = question(tokens=(100, 10, 20))
        self.assertEqual(min_feasible_k(q, build_label(q, parameters())), 2)

    def test_equal_token_tie_chooses_smaller_k(self):
        q = question(tokens=(20, 10, 10))
        self.assertEqual(min_feasible_k(q, build_label(q, parameters())), 2)

    def test_reject_non_candidates_when_minimizing(self):
        q = question(evidence=(0, 1, 1), tokens=(1, 10, 20))
        self.assertEqual(min_feasible_k(q, build_label(q, parameters())), 2)

    def test_no_budget_for_empty(self):
        q = question(evidence=(0, 0, 0))
        self.assertIsNone(min_feasible_k(q, build_label(q, parameters())))

    def test_no_budget_for_all_zero(self):
        q = question(f1=(0, 0, 0))
        self.assertIsNone(min_feasible_k(q, build_label(q, parameters())))

    def test_utility_quality_cost_gain(self):
        q = question()
        p = parameters()
        rows = budget_records([q], [build_label(q, p)], p)
        r = rows[0]
        self.assertEqual((r["min_feasible_k"], r["utility_k"]), (1, 2))
        self.assertAlmostEqual(r["utility_minus_minimum_f1"], 0.1)
        self.assertEqual(r["utility_minus_minimum_context_tokens"], 10)

    def test_lambda_changes_choice(self):
        q = question()
        self.assertEqual(build_label(q, parameters(lambda_=0.1))["argmax_soft_label_k"], 2)
        self.assertEqual(build_label(q, parameters(lambda_=1))["argmax_soft_label_k"], 1)

    def test_budget_comparison_evaluates_same_valid_qids(self):
        qs = [question(), question(evidence=(0, 0, 0), qid="q2"), question(qid="q3", type_="comparison")]
        p = parameters()
        labels = [build_label(q, p) for q in qs]
        rows = budget_records(qs, labels, p)
        table, summary = compare_budgets(qs, rows)
        self.assertEqual(summary["combined"]["evaluated_n"], 2)
        self.assertEqual(summary["combined"]["population_n"], 3)
        self.assertEqual({r["evaluated_n"] for r in table if r["group"] == "combined"}, {2})
        self.assertIsNone(rows[1]["utility_f1"])


class SharpnessTests(unittest.TestCase):
    def test_singleton_has_null_top_two_metrics(self):
        q = question(evidence=(0, 1, 0))
        p = parameters()
        r = sharpness_record(q, build_label(q, p), p)
        self.assertEqual((r["max_probability"], r["entropy"]), (1, 0))
        for field in ("second_probability", "top1_top2_probability_gap", "top_utility_gap", "normalized_entropy"):
            self.assertIsNone(r[field])

    def test_probability_gap_temperature_identity(self):
        q = question()
        p = parameters(temperature=0.02)
        r = sharpness_record(q, build_label(q, p), p)
        self.assertAlmostEqual(
            math.log(r["max_probability"] / r["second_probability"]), r["top_utility_gap"] / p.temperature
        )
        self.assertAlmostEqual(r["top1_top2_probability_gap"], r["max_probability"] - r["second_probability"])

    def test_temperature_invariance_and_entropy(self):
        q = question()
        rs = []
        for t in (0.01, 0.02, 0.05, 0.1):
            p = parameters(temperature=t)
            rs.append(sharpness_record(q, build_label(q, p), p))
        self.assertEqual(len({r["argmax_k"] for r in rs}), 1)
        self.assertEqual(len({r["top_utility_gap"] for r in rs}), 1)
        self.assertTrue(all(b["entropy"] >= a["entropy"] for a, b in zip(rs, rs[1:])))

    def test_tied_utilities_do_not_become_onehot(self):
        q = question(f1=(1, 1, 1), tokens=(10, 10, 30))
        p = parameters(temperature=0.000001)
        r = sharpness_record(q, build_label(q, p), p)
        self.assertEqual(r["top_utility_gap"], 0)
        self.assertEqual(r["max_probability"], 0.5)
        self.assertEqual(r["top1_top2_probability_gap"], 0)

    def test_expected_cost_from_probability(self):
        q = question()
        p = parameters()
        r = sharpness_record(q, build_label(q, p), p)
        self.assertAlmostEqual(
            r["expected_context_tokens"], sum(a * b for a, b in zip(r["soft_label"], q.tokens["context_tokens"]))
        )

    def test_cost_only_labels_depend_on_lambda_over_temperature(self):
        q = question(f1=(1, 1, 1))
        a = parameters(lambda_=0.1, temperature=0.01)
        b = parameters(lambda_=0.2, temperature=0.02)
        self.assertEqual(build_label(q, a)["soft_label"], build_label(q, b)["soft_label"])

    def test_no_parameter_state_pollution(self):
        q = question()
        before = copy.deepcopy(q)
        p = parameters()
        r = sharpness_record(q, build_label(q, p), p)
        other = parameters(lambda_=0.8, temperature=0.01)
        sharpness_record(q, build_label(q, other), other)
        self.assertEqual(q, before)
        self.assertEqual(r, sharpness_record(q, build_label(q, p), p))

    def test_summary_excludes_singleton_undefined_metrics(self):
        p = parameters()
        qs = [question(), question(evidence=(0, 1, 0), qid="q2")]
        records = [sharpness_record(q, build_label(q, p), p) for q in qs]
        row, detail = summarize_sharpness(records, p, "combined", 3, 0.95)
        self.assertEqual((row["evaluated_n"], row["multiple_n"], row["singleton_n"]), (2, 1, 1))
        self.assertEqual(detail["normalized_entropy"]["n"], 1)
        self.assertEqual(detail["top_utility_gap"]["n"], 1)
        self.assertEqual(sum(detail["max_probability_hist"]["counts"]), 2)


class EmptyTests(unittest.TestCase):
    def test_A_no_evidence_budget(self):
        r = classify_empty(question(evidence=(0, 0.25, 0.4)), parameters())
        self.assertEqual(r["category"], "evidence_unreachable")
        self.assertIsNone(r["best_f1_with_evidence"])
        self.assertAlmostEqual(r["max_evidence_recall"], 0.4)

    def test_B_evidence_quality_conflict(self):
        q = question(f1=(1, 0.7, 0.6), evidence=(0, 0.5, 1))
        r = classify_empty(q, parameters())
        self.assertEqual(r["category"], "constraint_conflict")
        self.assertAlmostEqual(r["global_to_evidence_f1_gap"], 0.3)
        self.assertAlmostEqual(r["gap_excess_over_epsilon"], 0.2)

    def test_boundary_is_valid_not_conflict(self):
        q = question(f1=(1, 0.9, 0.5), evidence=(0, 0.5, 0.4))
        self.assertEqual(classify_empty(q, parameters())["category"], "valid")

    def test_roundoff_boundary_reuses_original_tolerance(self):
        q = question(f1=(1, 0.9 - 5e-13, 0.5), evidence=(0, 0.5 - 5e-13, 0.4))
        self.assertEqual(classify_empty(q, parameters())["category"], "valid")

    def test_all_zero_separate_from_A(self):
        r = classify_empty(question(f1=(0, 0, 0), evidence=(0, 0, 0)), parameters())
        self.assertEqual(r["category"], "all_zero_f1")

    def test_all_zero_separate_from_B(self):
        r = classify_empty(question(f1=(0, 0, 0), evidence=(0, 1, 1)), parameters())
        self.assertEqual(r["category"], "all_zero_f1")

    def test_unknown_keeps_lower_and_upper(self):
        q = question(evidence=(0, 0.25, 0.25), unknown=(0, 1, 1))
        r = classify_empty(q, parameters())
        self.assertEqual(r["category"], "evidence_unreachable")
        self.assertTrue(r["has_unknown"])
        self.assertEqual(r["max_sentence_recall_upper"], 0.75)

    def test_smax_is_global_for_conflict(self):
        q = question(f1=(1, 0.3, 0.3), evidence=(0, 1, 1))
        self.assertEqual(classify_empty(q, parameters())["category"], "constraint_conflict")

    def test_reason_summary_partition(self):
        qs = [
            question(),
            question(evidence=(0, 0, 0), qid="q2"),
            question(f1=(1, 0.7, 0.6), evidence=(0, 0.5, 1), qid="q3"),
            question(f1=(0, 0, 0), qid="q4"),
        ]
        rows = summarize_empty([classify_empty(q, parameters()) for q in qs])
        r = next(x for x in rows if x["group"] == "combined")
        self.assertEqual(
            (
                r["population_n"],
                r["valid_n"],
                r["all_zero_f1_n"],
                r["evidence_unreachable_n"],
                r["constraint_conflict_n"],
            ),
            (4, 1, 1, 1, 1),
        )

    def test_tau_transitions_distinguish_new_losses_and_old_B_to_A(self):
        qs = [question(evidence=(0.5, 0.5, 0.5)), question(f1=(1, 0.5, 0.5), evidence=(0, 0.5, 0.5), qid="q2")]
        records = [classify_empty(q, parameters(tau=t)) for t in (0.5, 1) for q in qs]
        r = next(r for r in empty_transitions(records) if r["group"] == "combined")
        self.assertEqual((r["new_loss_A_n"], r["new_loss_B_n"], r["previous_conflict_to_unreachable_n"]), (1, 0, 1))

    def test_reject_other_scheme(self):
        with self.assertRaises(ValueError):
            classify_empty(question(), parameters(scheme="B"))


class OutputTests(unittest.TestCase):
    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "keep.txt"
            p.write_text("keep")
            with self.assertRaises(FileExistsError):
                allocate_followup_output(d)
            self.assertEqual(p.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
