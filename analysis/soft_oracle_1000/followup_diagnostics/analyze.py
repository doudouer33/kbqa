#!/usr/bin/env python3
"""Offline follow-up diagnostics. Reuse the existing Soft Oracle without editing it."""

import argparse
import itertools
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from build_soft_oracle import (  # noqa: E402
    BOUNDARY_ATOL,
    ROOT,
    allocate_output,
    assert_inputs_unchanged,
    build_label,
    close,
    context,
    index_rows,
    number,
    read_json,
    read_rows,
    require,
    sha256,
    write_csv,
    write_json,
)
from analyze_soft_oracle import binned, distribution, group_indices, histogram, mean, ratio  # noqa: E402

VERSION = "soft-oracle-followup-1.0.0"
GROUPS = ("combined", "bridge", "comparison")
STATUSES = ("valid", "evidence_unreachable", "constraint_conflict", "all_zero_f1")


def min_feasible_k(question, label, cost_metric="context_tokens"):
    if label["label_status"] != "valid":
        return None
    return min(label["valid_candidates"], key=lambda k: (question.tokens[cost_metric][question.ks.index(k)], k))


def selected_metrics(question, k, params):
    if k is None:
        return {key: None for key in ("k", "f1", "context_tokens", "evidence_recall", "f1_loss", "evidence_pass")}
    i = question.ks.index(k)
    evidence = question.evidence[i][params.evidence_metric]
    return {
        "k": k,
        "f1": question.f1[i],
        "context_tokens": question.tokens["context_tokens"][i],
        "evidence_recall": evidence,
        "f1_loss": max(question.f1) - question.f1[i],
        "evidence_pass": evidence + BOUNDARY_ATOL >= params.tau,
    }


def budget_records(questions, labels, params):
    output = []
    for q, label in zip(questions, labels):
        minimum = min_feasible_k(q, label)
        utility = label["argmax_soft_label_k"]
        record = {
            "question_id": q.question_id,
            "type": q.type,
            "label_status": label["label_status"],
            "candidate_count": label["valid_candidate_count"],
            "valid_candidates": label["valid_candidates"],
            "hard_oracle_k": q.original_hard_oracle_k,
            "min_feasible_k": minimum,
            "utility_k": utility,
            "different_k": minimum != utility if minimum is not None else None,
        }
        for strategy, k in (("hard", q.original_hard_oracle_k), ("minimum", minimum), ("utility", utility)):
            record.update({strategy + "_" + field: value for field, value in selected_metrics(q, k, params).items()})
        record["utility_minus_minimum_f1"] = (
            record["utility_f1"] - record["minimum_f1"] if minimum is not None else None
        )
        record["utility_minus_minimum_context_tokens"] = (
            record["utility_context_tokens"] - record["minimum_context_tokens"] if minimum is not None else None
        )
        if minimum is not None:
            require(record["utility_minus_minimum_context_tokens"] >= 0, "Utility cheaper than the minimum")
            require(
                record["utility_minus_minimum_f1"] >= -BOUNDARY_ATOL,
                "Utility lost F1 versus minimum for nonnegative lambda",
            )
        output.append(record)
    return output


def compare_budgets(questions, records):
    rows, detailed = [], {}
    for group in GROUPS:
        population = [r for r in records if group == "combined" or r["type"] == group]
        valid = [r for r in population if r["label_status"] == "valid"]
        changed = [r for r in valid if r["different_k"]]
        paired = {
            "different_k_n": len(changed),
            "different_k_ratio": ratio(len(changed), len(valid)),
            "utility_minus_minimum_f1": mean(r["utility_minus_minimum_f1"] for r in valid),
            "utility_minus_minimum_context_tokens": mean(r["utility_minus_minimum_context_tokens"] for r in valid),
            "changed_only_f1_gain": distribution(r["utility_minus_minimum_f1"] for r in changed),
            "changed_only_extra_context_tokens": distribution(
                r["utility_minus_minimum_context_tokens"] for r in changed
            ),
        }
        strategies = {}
        for strategy in ("hard", "minimum", "utility"):
            metrics = {
                field: mean(r[strategy + "_" + field] for r in valid)
                for field in ("f1", "context_tokens", "evidence_recall", "f1_loss", "k", "evidence_pass")
            }
            metrics["k_hist"] = histogram([r[strategy + "_k"] for r in valid], questions[0].ks)
            strategies[strategy] = metrics
            rows.append(
                {
                    "group": group,
                    "strategy": strategy,
                    "population_n": len(population),
                    "evaluated_n": len(valid),
                    "coverage": ratio(len(valid), len(population)),
                    **metrics,
                    **paired,
                }
            )
        detailed[group] = {
            "population_n": len(population),
            "evaluated_n": len(valid),
            "strategies": strategies,
            "paired": paired,
        }
    return rows, detailed


def sharpness_record(question, label, params):
    require(label["label_status"] == "valid", "Sharpness needs valid labels")
    candidates = [i for i, m in enumerate(label["candidate_mask"]) if m]
    ordered = sorted(candidates, key=lambda i: (-label["utility_curve"][i], question.ks[i]))
    first, second = ordered[0], ordered[1] if len(ordered) > 1 else None
    probabilities = label["soft_label"]
    utilities = [label["utility_curve"][i] for i in candidates]
    p1, p2 = probabilities[first], probabilities[second] if second is not None else None
    gap = label["utility_curve"][first] - label["utility_curve"][second] if second is not None else None
    result = {
        "question_id": question.question_id,
        "type": question.type,
        "lambda": params.lambda_,
        "temperature": params.temperature,
        "candidate_count": len(candidates),
        "argmax_k": label["argmax_soft_label_k"],
        "min_feasible_k": min_feasible_k(question, label),
        "entropy": label["entropy"],
        "normalized_entropy": label["entropy_normalized_candidates"],
        "max_probability": p1,
        "second_probability": p2,
        "top1_top2_probability_gap": p1 - p2 if p2 is not None else None,
        "top_utility_gap": gap,
        "top_utility_gap_over_temperature": gap / params.temperature if gap is not None else None,
        "utility_span": max(utilities) - min(utilities),
        "utility_span_over_temperature": (max(utilities) - min(utilities)) / params.temperature,
        "candidate_f1_span": max(question.f1[i] for i in candidates) - min(question.f1[i] for i in candidates),
        "top_utility_exact_tie": gap == 0 if gap is not None else None,
        "top_utility_near_tie": gap <= BOUNDARY_ATOL if gap is not None else None,
        "soft_label": probabilities,
        "candidate_mask": label["candidate_mask"],
        "utility_curve": label["utility_curve"],
        "normalized_cost_curve": label["normalized_cost_curve"],
        "expected_f1": math.fsum(p * f for p, f in zip(probabilities, question.f1)),
        "expected_context_tokens": math.fsum(p * t for p, t in zip(probabilities, question.tokens["context_tokens"])),
        "expected_evidence_recall": math.fsum(
            p * e[params.evidence_metric] for p, e in zip(probabilities, question.evidence)
        ),
    }
    result.update(
        {
            "argmax_" + k: v
            for k, v in selected_metrics(question, label["argmax_soft_label_k"], params).items()
            if k != "k"
        }
    )
    minimum = selected_metrics(question, result["min_feasible_k"], params)
    result["differs_from_minimum"] = result["argmax_k"] != result["min_feasible_k"]
    result["argmax_minus_minimum_f1"] = result["argmax_f1"] - minimum["f1"]
    result["argmax_minus_minimum_context_tokens"] = result["argmax_context_tokens"] - minimum["context_tokens"]
    if second is not None and p2 > 0:
        close(math.log(p1 / p2), gap / params.temperature, "log(p1/p2) = utility gap / T")
    return result


def summarize_sharpness(records, params, group, population_n, near_one_hot):
    part = [r for r in records if group == "combined" or r["type"] == group]
    multi = [r for r in part if r["candidate_count"] > 1]
    metrics = {
        "parameters": params.to_dict(),
        "group": group,
        "population_n": population_n,
        "valid_n": len(part),
        "coverage": ratio(len(part), population_n),
        "singleton_n": len(part) - len(multi),
        "multiple_n": len(multi),
    }
    for field in (
        "entropy",
        "normalized_entropy",
        "max_probability",
        "top1_top2_probability_gap",
        "top_utility_gap",
        "top_utility_gap_over_temperature",
        "utility_span",
        "utility_span_over_temperature",
        "candidate_f1_span",
    ):
        metric_part = multi if field in ("utility_span", "utility_span_over_temperature") else part
        metrics[field] = distribution(r[field] for r in metric_part)
    metrics["max_probability_multi"] = distribution(r["max_probability"] for r in multi)
    metrics["near_one_hot_threshold"] = near_one_hot
    metrics["near_one_hot_all_ratio"] = ratio(sum(r["max_probability"] >= near_one_hot for r in part), len(part))
    metrics["near_one_hot_multi_ratio"] = ratio(sum(r["max_probability"] >= near_one_hot for r in multi), len(multi))
    metrics["top_utility_exact_tie_multi_n"] = sum(r["top_utility_exact_tie"] for r in multi)
    metrics["candidate_f1_flat_multi_ratio"] = ratio(sum(r["candidate_f1_span"] == 0 for r in multi), len(multi))
    metrics["argmax_k_hist"] = histogram([r["argmax_k"] for r in part], range(1, 16))
    metrics["max_probability_hist"] = binned(
        [r["max_probability"] for r in part], [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1]
    )
    for field in (
        "expected_f1",
        "argmax_f1",
        "argmax_f1_loss",
        "expected_context_tokens",
        "argmax_context_tokens",
        "argmax_evidence_recall",
        "expected_evidence_recall",
        "differs_from_minimum",
        "argmax_minus_minimum_f1",
        "argmax_minus_minimum_context_tokens",
    ):
        metrics[field] = mean(r[field] for r in part)
    metrics["entropy_by_candidate_count"] = {
        str(c): {
            "n": sum(r["candidate_count"] == c for r in part),
            "entropy": distribution(r["entropy"] for r in part if r["candidate_count"] == c),
            "normalized_entropy": distribution(r["normalized_entropy"] for r in part if r["candidate_count"] == c),
        }
        for c in range(1, 16)
    }
    row = {
        **params.to_dict(),
        "group": group,
        "population_n": population_n,
        "evaluated_n": len(part),
        "coverage": metrics["coverage"],
        "singleton_n": metrics["singleton_n"],
        "multiple_n": metrics["multiple_n"],
        "mean_entropy": metrics["entropy"]["mean"],
        "mean_entropy_normalized_candidates": metrics["normalized_entropy"]["mean"],
        "mean_max_probability": metrics["max_probability"]["mean"],
        "mean_max_probability_multi": metrics["max_probability_multi"]["mean"],
        "mean_top1_top2_probability_gap": metrics["top1_top2_probability_gap"]["mean"],
        "mean_top_utility_gap": metrics["top_utility_gap"]["mean"],
        "mean_top_utility_gap_over_temperature": metrics["top_utility_gap_over_temperature"]["mean"],
        "mean_utility_span_over_temperature": metrics["utility_span_over_temperature"]["mean"],
        **{
            key: metrics[key]
            for key in (
                "near_one_hot_threshold",
                "near_one_hot_all_ratio",
                "near_one_hot_multi_ratio",
                "top_utility_exact_tie_multi_n",
                "candidate_f1_flat_multi_ratio",
                "expected_f1",
                "argmax_f1",
                "argmax_f1_loss",
                "expected_context_tokens",
                "argmax_context_tokens",
                "differs_from_minimum",
                "argmax_minus_minimum_f1",
                "argmax_minus_minimum_context_tokens",
                "argmax_k_hist",
            )
        },
        "max_probability_distribution": metrics["max_probability"],
        "max_probability_hist": metrics["max_probability_hist"],
    }
    return row, metrics


def classify_empty(question, params, label=None):
    require(params.scheme == "D", "Empty-candidate diagnoses require scheme D")
    label = build_label(question, params) if label is None else label
    evidence_indices = [i for i, v in enumerate(label["evidence_pass"]) if v]
    evidence = [r[params.evidence_metric] for r in question.evidence]
    f1_best_at_evidence = max(question.f1[i] for i in evidence_indices) if evidence_indices else None
    if label["label_status"] == "all_zero_f1":
        category = "all_zero_f1"
    elif label["label_status"] == "valid":
        category = "valid"
    elif not evidence_indices:
        category = "evidence_unreachable"
    else:
        category = "constraint_conflict"
        require(
            f1_best_at_evidence + BOUNDARY_ATOL < max(question.f1) - params.epsilon,
            "Conflict classification must miss the F1 boundary",
        )
    gap = max(question.f1) - f1_best_at_evidence if f1_best_at_evidence is not None else None
    upper = [r["sentence_recall_upper"] for r in question.evidence]
    return {
        "question_id": question.question_id,
        "type": question.type,
        "tau": params.tau,
        "epsilon": params.epsilon,
        "label_status": label["label_status"],
        "category": category,
        "max_f1": max(question.f1),
        "max_evidence_recall": max(evidence),
        "max_sentence_recall_upper": max(upper),
        "evidence_deficit_to_tau": max(0, params.tau - max(evidence)),
        "best_f1_with_evidence": f1_best_at_evidence,
        "global_to_evidence_f1_gap": gap,
        "gap_excess_over_epsilon": gap - params.epsilon if gap is not None else None,
        "max_evidence_in_f1_near_optimal_set": max(e for e, yes in zip(evidence, label["f1_pass"]) if yes),
        "evidence_eligible_k": [question.ks[i] for i in evidence_indices],
        "f1_eligible_k": [k for k, yes in zip(question.ks, label["f1_pass"]) if yes],
        "valid_candidates": label["valid_candidates"],
        "sentence_unknown_count_curve": [r["sentence_unknown"] for r in question.evidence],
        "has_unknown": any(r["sentence_unknown"] for r in question.evidence),
        "question_text": question.question_text,
        "gold_answer": question.gold_answer,
        "k_values": question.ks,
        "f1_curve": question.f1,
        "evidence_recall_curve": evidence,
        "sentence_recall_upper_curve": upper,
        "context_tokens_curve": question.tokens["context_tokens"],
        "predictions": question.predictions,
    }


def summarize_empty(records):
    rows = []
    for tau in sorted({r["tau"] for r in records}):
        for group in GROUPS:
            part = [r for r in records if r["tau"] == tau and (group == "combined" or r["type"] == group)]
            counts = Counter(r["category"] for r in part)
            a = [r for r in part if r["category"] == "evidence_unreachable"]
            b = [r for r in part if r["category"] == "constraint_conflict"]
            empty_n = len(a) + len(b)
            row = {
                "tau": tau,
                "group": group,
                "population_n": len(part),
                "valid_n": counts["valid"],
                "valid_ratio": ratio(counts["valid"], len(part)),
                "all_zero_f1_n": counts["all_zero_f1"],
                "empty_nonzero_n": empty_n,
                "evidence_unreachable_n": len(a),
                "constraint_conflict_n": len(b),
                "evidence_unreachable_ratio_empty": ratio(len(a), empty_n),
                "constraint_conflict_ratio_empty": ratio(len(b), empty_n),
                "A_max_evidence_distribution": distribution(r["max_evidence_recall"] for r in a),
                "A_max_evidence_hist": dict(sorted(Counter(str(r["max_evidence_recall"]) for r in a).items())),
                "B_f1_gap_distribution": distribution(r["global_to_evidence_f1_gap"] for r in b),
                "B_gap_excess_over_epsilon_distribution": distribution(r["gap_excess_over_epsilon"] for r in b),
                "empty_with_unknown_n": sum(r["has_unknown"] for r in a + b),
            }
            require(
                sum(counts.values()) == len(part) == counts["valid"] + counts["all_zero_f1"] + empty_n,
                "Reason partition mismatch",
            )
            rows.append(row)
    return rows


def empty_transitions(records):
    lookup = {(r["question_id"], r["tau"]): r for r in records}
    taus = sorted({r["tau"] for r in records})
    output = []
    for before, after in zip(taus, taus[1:]):
        for group in GROUPS:
            part = [r for r in records if r["tau"] == before and (group == "combined" or r["type"] == group)]
            matrix = Counter((r["category"], lookup[(r["question_id"], after)]["category"]) for r in part)
            require(
                not any(matrix[s, "valid"] for s in STATUSES if s != "valid"),
                "Raising tau unexpectedly recovered a label",
            )
            output.append(
                {
                    "tau_before": before,
                    "tau_after": after,
                    "group": group,
                    "population_n": len(part),
                    "valid_before_n": sum(r["category"] == "valid" for r in part),
                    "new_loss_A_n": matrix["valid", "evidence_unreachable"],
                    "new_loss_B_n": matrix["valid", "constraint_conflict"],
                    "new_loss_total_n": matrix["valid", "evidence_unreachable"]
                    + matrix["valid", "constraint_conflict"],
                    "previous_conflict_to_unreachable_n": matrix["constraint_conflict", "evidence_unreachable"],
                    "transition_counts": {"{}->{}".format(a, b): n for (a, b), n in sorted(matrix.items())},
                }
            )
    return output


def verify_source_labels(questions, labels, source):
    saved = index_rows(read_rows(source / "labels.jsonl"), ("question_id",), "source labels")
    require(set(saved) == {(q.question_id,) for q in questions}, "Original label qids differ")
    for q, label in zip(questions, labels):
        original = saved[(q.question_id,)]
        for key in ("label_status", "argmax_soft_label_k", "valid_candidate_count", "parameters"):
            require(label[key] == original[key], "Original label changed: " + key)
        require(list(label["candidate_mask"]) == original["candidate_mask"], "Original mask changed")
        if label["label_status"] == "valid":
            for a, b in zip(label["soft_label"], original["soft_label"]):
                close(a, b, "Original probability")
        else:
            require(original["soft_label"] is None, "Original invalid label is not null")
    old_sweep = read_rows(source / "parameter_sweep.csv")
    require(len(old_sweep) == 2592, "Expected complete existing sweep")
    return old_sweep


def verify_overlap(sweep, old_sweep):
    checked = 0
    for row in sweep:
        matching = [
            r
            for r in old_sweep
            if r["group"] == row["group"]
            and r["scope"] == "own_valid"
            and r["scheme"] == "D"
            and all(float(r[k]) == row[k] for k in ("epsilon", "tau", "lambda", "temperature"))
        ]
        require(len(matching) <= 1, "Duplicate existing sweep key")
        if not matching:
            continue
        old = matching[0]
        require(int(old["evaluated_n"]) == row["evaluated_n"], "Overlap count mismatch")
        for key in (
            "mean_entropy",
            "mean_entropy_normalized_candidates",
            "mean_max_probability",
            "argmax_f1",
            "expected_f1",
            "argmax_context_tokens",
            "expected_context_tokens",
        ):
            close(float(old[key]), row[key], "Existing sweep overlap: " + key)
        checked += 1
    return checked


def snapshot_existing():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(ROOT)).decode().split("\0")
    untracked = (
        subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=str(ROOT))
        .decode()
        .split("\0")
    )
    prefix = str(HERE.relative_to(ROOT)) + "/"
    return {
        name: sha256(ROOT / name)
        for name in sorted(set(tracked + untracked))
        if name and not name.startswith(prefix) and (ROOT / name).is_file()
    }


def allocate_followup_output(path=None):
    if path is not None:
        return allocate_output(path)
    parent = HERE / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = parent / "experiment_{:03d}".format(index)
        try:
            return allocate_output(candidate)
        except FileExistsError:
            index += 1


def choose_examples(questions, budget_rows, empty_rows, reference):
    by_qid = {q.question_id: q for q in questions}
    examples = []
    for group in ("bridge", "comparison"):
        pools = {
            "utility_vs_minimum_different": [r for r in budget_rows if r["type"] == group and r["different_k"]],
            "evidence_unreachable": [
                r
                for r in empty_rows
                if r["type"] == group and r["tau"] == reference.tau and r["category"] == "evidence_unreachable"
            ],
            "constraint_conflict": [
                r
                for r in empty_rows
                if r["type"] == group and r["tau"] == reference.tau and r["category"] == "constraint_conflict"
            ],
        }
        for category, pool in pools.items():
            chosen = sorted(pool, key=lambda r: r["question_id"])
            if category != "utility_vs_minimum_different":
                chosen = chosen[:2]
            for r in chosen:
                q = by_qid[r["question_id"]]
                label = build_label(q, reference)
                examples.append(
                    {
                        "example_category": category,
                        "category_n_in_type": len(pool),
                        "selection_rule": "all different-budget cases; first two qids within type for A/B",
                        "parameters": reference.to_dict(),
                        **classify_empty(q, reference, label),
                        "min_feasible_k": min_feasible_k(q, label),
                        "utility_k": label["argmax_soft_label_k"],
                        "hard_oracle_k": q.original_hard_oracle_k,
                        "soft_label": label["soft_label"],
                        "utility_curve": label["utility_curve"],
                    }
                )
    return examples


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def pct(value):
    return "—" if value is None else "{:.2%}".format(value)


def fmt(value, digits=4):
    return "—" if value is None else ("{:,." + str(digits) + "f}").format(value)


def md_table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in row) + " |" for row in rows]
    )


def sweep_row(rows, reference, group="combined", lambda_=None, temperature=None):
    found = [
        r
        for r in rows
        if r["group"] == group
        and r["lambda"] == (reference.lambda_ if lambda_ is None else lambda_)
        and r["temperature"] == (reference.temperature if temperature is None else temperature)
    ]
    require(len(found) == 1, "Expected one sharpness summary row")
    return found[0]


def make_figures(output, comparison, sweep, details, reasons, examples, reference):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir()
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    colors = {"hard": "#8995aa", "minimum": "#4b9276", "utility": "#dc9c3d"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field, title in zip(
        axes,
        ("f1", "context_tokens", "evidence_recall"),
        ("Answer F1", "Context tokens", "Gold sentence recall (lower)"),
    ):
        for i, strategy in enumerate(colors):
            values = [comparison[g]["strategies"][strategy][field] for g in GROUPS]
            ax.bar([j + (i - 1) * 0.24 for j in range(3)], values, width=0.24, label=strategy, color=colors[strategy])
        ax.set_xticks(range(3), GROUPS)
        ax.set_title(title)
        if field != "context_tokens":
            ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Offline Oracle: identical valid queries for all three strategies")
    fig.tight_layout()
    fig.savefig(directory / "min_feasible_comparison.png")
    plt.close(fig)

    lambdas = sorted({r["lambda"] for r in sweep})
    temps = sorted({r["temperature"] for r in sweep})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, field, title in zip(
        axes,
        ("mean_entropy_normalized_candidates", "mean_max_probability", "expected_context_tokens"),
        ("H / log(candidate count), multi only", "Mean maximum probability, all valid", "Expected context tokens"),
    ):
        data = [
            [sweep_row(sweep, reference, lambda_=lambda_value, temperature=t)[field] for t in temps]
            for lambda_value in lambdas
        ]
        im = ax.imshow(data, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(temps)), [str(t) for t in temps])
        ax.set_yticks(range(len(lambdas)), [str(lambda_value) for lambda_value in lambdas])
        for y, row in enumerate(data):
            for x, v in enumerate(row):
                ax.text(
                    x,
                    y,
                    "{:.2f}".format(v) if field != "expected_context_tokens" else "{:.0f}".format(v),
                    ha="center",
                    va="center",
                    color="white" if v > (min(map(min, data)) + max(map(max, data))) / 2 else "black",
                )
        ax.set(xlabel="Temperature", ylabel="Lambda", title=title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Combined: fixed epsilon/tau and fixed valid sample set")
    fig.tight_layout()
    fig.savefig(directory / "sharpness_heatmap.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lambda_value in lambdas:
        axes[0].plot(
            temps,
            [
                sweep_row(sweep, reference, lambda_=lambda_value, temperature=t)["mean_top1_top2_probability_gap"]
                for t in temps
            ],
            "o-",
            label="lambda={}".format(lambda_value),
        )
    axes[0].set(xlabel="Temperature", ylabel="Mean p1 - p2 (multi only)", title="Probability separation")
    axes[0].legend()
    selected = sorted({(reference.lambda_, reference.temperature), (lambdas[0], temps[0]), (lambdas[-1], temps[0])})
    for lambda_value, t in selected:
        d = next(
            r
            for r in details
            if r["group"] == "combined"
            and r["parameters"]["lambda"] == lambda_value
            and r["parameters"]["temperature"] == t
        )
        h = d["max_probability_hist"]
        axes[1].stairs(h["counts"], h["edges"], label="lambda={}, T={}".format(lambda_value, t), linewidth=2)
    axes[1].set(
        xlabel="Maximum label probability", ylabel="Queries (all valid)", title="Maximum probability distribution"
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "probability_sharpness.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lambda_value in lambdas:
        part = [sweep_row(sweep, reference, lambda_=lambda_value, temperature=t) for t in temps]
        axes[0].plot(
            [r["expected_context_tokens"] for r in part],
            [r["expected_f1"] for r in part],
            "o-",
            label="lambda={}".format(lambda_value),
        )
        a = part[0]
        axes[0].scatter(a["argmax_context_tokens"], a["argmax_f1"], marker="x", s=70)
    minimum = comparison["combined"]["strategies"]["minimum"]
    axes[0].scatter(
        minimum["context_tokens"], minimum["f1"], marker="*", s=130, color="black", label="Minimum feasible"
    )
    axes[0].set(xlabel="Context tokens", ylabel="Answer F1", title="Offline expected points; x = argmax")
    axes[0].legend(fontsize=8)
    for group in GROUPS:
        rows = [
            sweep_row(sweep, reference, group, lambda_=lambda_value, temperature=reference.temperature)
            for lambda_value in lambdas
        ]
        axes[1].plot(lambdas, [r["differs_from_minimum"] for r in rows], "o-", label=group)
    axes[1].set(xlabel="Lambda", ylabel="Fraction choosing different k", title="Utility argmax versus minimum feasible")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(directory / "quality_cost_sensitivity.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    status_colors = {
        "valid_n": "#4b9276",
        "evidence_unreachable_n": "#c77c60",
        "constraint_conflict_n": "#d3b95b",
        "all_zero_f1_n": "#9b9b9b",
    }
    for ax, group in zip(axes, GROUPS):
        rows = [r for r in reasons if r["group"] == group]
        bottom = [0] * len(rows)
        for status, color in status_colors.items():
            values = [r[status] for r in rows]
            ax.bar(range(len(rows)), values, bottom=bottom, label=status.replace("_n", ""), color=color)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set_xticks(range(len(rows)), [str(r["tau"]) for r in rows])
        ax.set(xlabel="Evidence threshold tau", ylabel="Queries", title=group)
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Empty causes: A = evidence unreachable; B = constraint conflict; all-zero separate")
    fig.tight_layout()
    fig.savefig(directory / "empty_candidate_causes.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    chosen = []
    for category in ("evidence_unreachable", "constraint_conflict", "utility_vs_minimum_different"):
        pool = [r for r in examples if r["example_category"] == category]
        if pool:
            chosen.append(pool[0])
    conflicts = [r for r in examples if r["example_category"] == "constraint_conflict" and r["type"] == "comparison"]
    if conflicts:
        chosen.append(conflicts[0])
    for ax, example in zip(axes.flat, chosen):
        ax.plot(example["k_values"], example["f1_curve"], "o-", label="F1")
        ax.plot(example["k_values"], example["evidence_recall_curve"], "s-", label="Evidence lower")
        ax.axhline(reference.tau, color="gray", linestyle="--", label="tau")
        ax.axhline(example["max_f1"] - reference.epsilon, color="gray", linestyle=":", label="max F1 - epsilon")
        ax.set(xlabel="k", ylim=(-0.03, 1.04), title=example["example_category"] + "\n" + example["question_id"])
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(chosen) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(directory / "example_curves.png")
    plt.close(fig)
    return {
        "matplotlib_version": matplotlib.__version__,
        "figure_count": 6,
        "language": "English plot text, Chinese report captions",
    }


def make_report(output, questions, reference, config, comparisons, sweep, details, reasons, transitions, examples):
    combined = comparisons["combined"]
    n = combined["evaluated_n"]
    paired = combined["paired"]
    ref = sweep_row(sweep, reference)
    ref_detail = next(r for r in details if r["group"] == "combined" and r["parameters"] == reference.to_dict())
    rreason = next(r for r in reasons if r["group"] == "combined" and r["tau"] == reference.tau)
    a, b = rreason["evidence_unreachable_n"], rreason["constraint_conflict_n"]
    dimension_n = len(config["lambda_values"]) * len(config["temperature_values"])
    parts = [
        "# Soft Oracle 后续诊断与标签参数分析",
        "分析日期：{}（Asia/Shanghai）。输入仅为2026-10-14正式实验的已有F1、Token和模型可见上下文Evidence诊断，以及原Soft Oracle experiment_002。所有结果均为 **Offline Oracle / Label Analysis**，未调用检索器或LLM，未训练或扩大数据集。".format(
            datetime.now(timezone(timedelta(hours=8))).date()
        ),
        "## 1. 直接结论与适用边界",
        "参考参数 ε={}、τ={}、λ={}、T={}，有效题 {}/1000（{}），Bridge/Comparison分别 {}/500、{}/500；181道全零F1继续单列。旧标签与旧参数扫描交叉核对通过，本次未改变原Soft Oracle实现或旧结果。".format(
            reference.epsilon,
            reference.tau,
            reference.lambda_,
            reference.temperature,
            n,
            pct(n / len(questions)),
            comparisons["bridge"]["evaluated_n"],
            comparisons["comparison"]["evaluated_n"],
        ),
        "**Utility相对最小可行预算的边际收益很小。** 仅 {}/{}（{}）题选出不同k，平均F1增加 {} 个百分点，平均Context Tokens增加 {}。这说明当前参数下多数决策已被近优F1和Evidence门槛决定，Utility主要处理少量边界权衡，不能为了保留Utility而夸大收益。".format(
            paired["different_k_n"],
            n,
            pct(paired["different_k_ratio"]),
            fmt(paired["utility_minus_minimum_f1"] * 100),
            fmt(paired["utility_minus_minimum_context_tokens"], 3),
        ),
        "**平滑可由参数调节，但尖锐不等于更好的监督。** λ/T网格涵盖{}组参数、每组相同{}个有效题。参考多候选归一化熵={}；最大概率均值={}。降低T能在不改变最优预算的情况下显著改变概率形状，增大λ则可能改变预算。不能仅按低熵或离线F1自动选训练参数。".format(
            dimension_n, n, fmt(ref["mean_entropy_normalized_candidates"]), fmt(ref["mean_max_probability"])
        ),
        "**参考空候选主要是未观察到足够Gold句覆盖。** 非零空候选{}题中，A类{}题（{}）在全部k下Evidence均不过门槛，B类{}题（{}）是F1/Evidence集合冲突。该分类限于k≤15、当前检索及渲染上下文的Gold字面匹配代理，不能严格证明检索器失败或模型回答的因果机制。".format(
            a + b, a, pct(a / (a + b)), b, pct(b / (a + b))
        ),
        "当前证据支持把Minimum Feasible保留为必要的简单预算基线，并在训练目标中明确概率分布的用途。进入模型训练前仍需处理类型选择偏差、无标签样本、人工证据复核以及独立数据划分；本轮不能证明仅输入Query可预测预算。本报告不宣布最佳参数，不执行任何训练。",
        "## 2. 方法、复用范围与分母",
        "直接复用上级 `build_soft_oracle.py` 的 `context/load_data`、`build_label`、成本归一化、稳定Softmax、边界容差和写表函数；复用 `analyze_soft_oracle.py` 的均值、分位数和直方图函数。未重写F1/Evidence筛选或Utility。每个有效标签与原experiment_002逐题核对，重叠网格的汇总指标也与原parameter_sweep.csv核对。所有候选集合均取全15点Smax、成本最大值，温度不参与预算排序。",
        "Minimum Feasible定义为候选中实际Context Tokens最少，成本并列取较小k；代码不假设Token随k单调。Hard Oracle仍从全部15个k选最大F1并列最小k，可能不满足Evidence约束；三策略仅在同一D有效题集上评估。全体1000题逐题表保留无标签状态；无标签不回退，未给出伪装成全体性能的均值。",
        "尖锐度指标中：H使用自然对数；H/ln|C|、top1-top2概率差和top1-top2 Utility差仅在|C|>1时定义。单候选H=0、pmax=1，其余指标null，单候选数量单列。接近One-hot仅作描述，阈值pmax≥{}，同时给出全有效题和多候选题的比例。分位数为线性插值，无显著性或因果检验。".format(
            config["near_one_hot_threshold"]
        ),
        "## 3. 最小可行预算与Utility最优预算",
        md_table(
            ["组", "策略", "有效/原组", "平均F1", "平均Context", "平均E lower", "相对逐题max F1损失", "E达标比例"],
            [
                [
                    group,
                    strategy,
                    "{}/{}".format(c["evaluated_n"], c["population_n"]),
                    pct(m["f1"]),
                    fmt(m["context_tokens"], 3),
                    pct(m["evidence_recall"]),
                    fmt(m["f1_loss"] * 100) + "个百分点",
                    pct(m["evidence_pass"]),
                ]
                for group, c in comparisons.items()
                for strategy, m in c["strategies"].items()
            ],
        ),
        "Utility−Minimum Feasible的逐题配对差（所有有效题，而非仅差异题作分母）：",
        md_table(
            [
                "组",
                "选不同k/有效",
                "ΔF1(百分点)",
                "ΔContext Tokens",
                "仅差异题平均F1增益(百分点)",
                "仅差异题平均额外Token",
            ],
            [
                [
                    group,
                    "{}/{} ({})".format(
                        c["paired"]["different_k_n"], c["evaluated_n"], pct(c["paired"]["different_k_ratio"])
                    ),
                    fmt(c["paired"]["utility_minus_minimum_f1"] * 100),
                    fmt(c["paired"]["utility_minus_minimum_context_tokens"], 3),
                    (
                        fmt(c["paired"]["changed_only_f1_gain"]["mean"] * 100)
                        if c["paired"]["changed_only_f1_gain"]["n"]
                        else "—"
                    ),
                    fmt(c["paired"]["changed_only_extra_context_tokens"]["mean"], 3),
                ]
                for group, c in comparisons.items()
            ],
        ),
        "在相同候选集、非负λ下，最小成本是下界；Utility选择更昂贵的候选，必须由F1收益抵偿λ乘归一化成本增量。它不会额外加入被门槛排除的k。Hard Oracle具有最大F1保证，但Gold证据不足的比例需另看；不能用其F1优势否认证据目标，也不能把门槛保证的Evidence达标率当成学到的效果。",
        "![同题集的三种预算策略](figures/min_feasible_comparison.png)",
        "所有参考配置下的预算差异题均列于下表，不从中选择收益最大的个例：",
        md_table(
            ["qid", "类型", "最小可行k", "Utility k", "F1增益(百分点)", "额外Context Tokens"],
            [
                [
                    e["question_id"],
                    e["type"],
                    e["min_feasible_k"],
                    e["utility_k"],
                    fmt((e["f1_curve"][e["utility_k"] - 1] - e["f1_curve"][e["min_feasible_k"] - 1]) * 100),
                    e["context_tokens_curve"][e["utility_k"] - 1] - e["context_tokens_curve"][e["min_feasible_k"] - 1],
                ]
                for e in examples
                if e["example_category"] == "utility_vs_minimum_different"
            ],
        ),
        "## 4. λ、Temperature与Soft Label尖锐度",
        "以下只扫描用户指定网格，ε/τ固定。不同λ/T均采用相同有效题集，归一化熵和top1/top2差的分母为各组多候选题，pmax均值包括单候选。Answer F1同时列argmax和概率加权期望，避免混用。",
        md_table(
            [
                "组",
                "λ",
                "T",
                "n/多候选n",
                "H",
                "H/lnC",
                "pmax",
                "p1−p2(多候选)",
                "期望Context",
                "argmax Context",
                "期望F1",
                "argmax F1",
            ],
            [
                [
                    r["group"],
                    r["lambda"],
                    r["temperature"],
                    "{}/{}".format(r["evaluated_n"], r["multiple_n"]),
                    fmt(r["mean_entropy"], 3),
                    fmt(r["mean_entropy_normalized_candidates"], 3),
                    fmt(r["mean_max_probability"], 3),
                    fmt(r["mean_top1_top2_probability_gap"], 3),
                    fmt(r["expected_context_tokens"], 2),
                    fmt(r["argmax_context_tokens"], 2),
                    pct(r["expected_f1"]),
                    pct(r["argmax_f1"]),
                ]
                for r in sweep
            ],
        ),
        "![尖锐度与期望成本热图](figures/sharpness_heatmap.png)",
        "pmax分布与接近One-hot的比例（Combined）：",
        md_table(
            ["λ", "T", "pmax P25", "中位数", "P75", "P90", "pmax≥阈值/全有效", "pmax≥阈值/多候选", "与Minimum不同k"],
            [
                [
                    r["lambda"],
                    r["temperature"],
                    *[fmt(r["max_probability_distribution"][k], 3) for k in ("p25", "median", "p75", "p90")],
                    pct(r["near_one_hot_all_ratio"]),
                    pct(r["near_one_hot_multi_ratio"]),
                    pct(r["differs_from_minimum"]),
                ]
                for r in sweep
                if r["group"] == "combined"
            ],
        ),
        "argmax k=1…15分布（每格为数量，n相同；T不影响排序，因此每λ只列一次）：",
        md_table(
            ["组", "λ", "n", "k=1…15"],
            [
                [
                    r["group"],
                    r["lambda"],
                    r["evaluated_n"],
                    ", ".join(str(r["argmax_k_hist"][str(k)]) for k in range(1, 16)),
                ]
                for r in sweep
                if r["temperature"] == reference.temperature
            ],
        ),
        "![概率差和最大概率分布](figures/probability_sharpness.png)",
        "![质量成本与最小可行预算差异](figures/quality_cost_sensitivity.png)",
        "同λ下T的变化不改变argmax，已逐题验证；增大λ的预算Token不增加，也逐题验证。较低T保留同一Utility偏好但强化概率集中；较高λ改变质量与成本的偏好，不能解释成单纯温度缩放。对于候选F1完全相同的题，P只依赖λ/T及成本差，相同λ/T会产生相同分布；对F1不同的题则不成立。",
        "## 5. 原标签较均匀的原因",
        "对于至少两个候选，ln(p1/p2)=(U1−U2)/T。这给出可检查的尺度关系；top1/top2差只说明最佳两项接近，整个候选集是否接近均匀还需看全Utility跨度/T以及候选数。不能单凭小top1/top2差就说全部概率均匀。",
        md_table(
            [
                "λ",
                "T",
                "多候选n",
                "ΔU12均值",
                "ΔU12中位数",
                "ΔU12/T均值",
                "Utility跨度/T均值",
                "候选F1完全相同/多候选",
                "最佳Utility精确并列题数",
            ],
            [
                [
                    d["parameters"]["lambda"],
                    d["parameters"]["temperature"],
                    d["multiple_n"],
                    fmt(d["top_utility_gap"]["mean"]),
                    fmt(d["top_utility_gap"]["median"]),
                    fmt(d["top_utility_gap_over_temperature"]["mean"]),
                    fmt(d["utility_span_over_temperature"]["mean"]),
                    pct(d["candidate_f1_flat_multi_ratio"]),
                    d["top_utility_exact_tie_multi_n"],
                ]
                for d in details
                if d["group"] == "combined"
            ],
        ),
        "参考配置中，多候选题的ΔU12中位数={}，ΔU12/T均值={}，候选F1完全相同占{}；归一化熵={}。近优F1筛选保留了大量并列或相近F1，λ={}又将全题相对成本惩罚幅度限制在较小范围，和T={}一起产生平滑概率。对相同λ降低T，Utility差不变而相对温度尺度增大，故标签可以变尖锐；对真正Utility并列的多个预算，降低T仍不能区分它们。".format(
            fmt(ref_detail["top_utility_gap"]["median"]),
            fmt(ref_detail["top_utility_gap_over_temperature"]["mean"]),
            pct(ref_detail["candidate_f1_flat_multi_ratio"]),
            fmt(ref["mean_entropy_normalized_candidates"]),
            reference.lambda_,
            reference.temperature,
        ),
        "降低熵不增加新的回答观测，也不会消除F1平坦曲线的不可辨识性。分布的多候选信息与实际预算决策应分开：按Soft Label随机采样的期望成本和直接argmax成本不同；未来若采用KL/交叉熵训练，再以argmax推理，也不能假定训练分布的期望就是部署结果。当前只作离线标签分析。",
        "## 6. 无有效候选：A类与B类",
        "分类采用原实现的1e-12阈值容差。先单列全零F1；非零空候选中，A类为没有任何Evidence达标预算，B类为有Evidence达标预算但无同时F1近优预算。两类互斥且完备，不能把181道全零题计入其中。B类的全局F1−证据达标集合最大F1就是恢复交集所需的最小ε（仅作诊断，不自动放宽阈值）。",
        md_table(
            ["组", "τ", "有效/原组", "全零", "非零空集", "A数", "B数", "A/空集", "B/空集"],
            [
                [
                    r["group"],
                    r["tau"],
                    "{}/{}".format(r["valid_n"], r["population_n"]),
                    r["all_zero_f1_n"],
                    r["empty_nonzero_n"],
                    r["evidence_unreachable_n"],
                    r["constraint_conflict_n"],
                    pct(r["evidence_unreachable_ratio_empty"]),
                    pct(r["constraint_conflict_ratio_empty"]),
                ]
                for r in reasons
            ],
        ),
        "![空候选原因与阈值变化](figures/empty_candidate_causes.png)",
        "A类最大Evidence Recall分布（每题取全部15个k的最大值；列出全部离散值与数量）：",
        md_table(
            ["组", "τ", "A数", "max E均值", "中位数", "最大值", "离散频数"],
            [
                [
                    r["group"],
                    r["tau"],
                    r["evidence_unreachable_n"],
                    fmt(r["A_max_evidence_distribution"]["mean"]),
                    fmt(r["A_max_evidence_distribution"]["median"]),
                    fmt(r["A_max_evidence_distribution"]["max"]),
                    json.dumps(r["A_max_evidence_hist"], ensure_ascii=False),
                ]
                for r in reasons
            ],
        ),
        "B类质量冲突程度：",
        md_table(
            ["组", "τ", "B数", "全局max F1−证据达标最佳F1：均值", "最小", "中位数", "P90", "最大", "超出ε部分均值"],
            [
                [
                    r["group"],
                    r["tau"],
                    r["constraint_conflict_n"],
                    *[fmt(r["B_f1_gap_distribution"][k]) for k in ("mean", "min", "median", "p90", "max")],
                    fmt(r["B_gap_excess_over_epsilon_distribution"]["mean"]),
                ]
                for r in reasons
            ],
        ),
        "提高τ带来的新增样本损失，应只在前一个τ仍有效的题中统计，不能简单相减A/B总数（旧B可能转为A）：",
        md_table(
            ["组", "τ变化", "原有效n", "新增失效n", "新增A", "新增B", "旧B→A"],
            [
                [
                    r["group"],
                    "{}→{}".format(r["tau_before"], r["tau_after"]),
                    r["valid_before_n"],
                    r["new_loss_total_n"],
                    r["new_loss_A_n"],
                    r["new_loss_B_n"],
                    r["previous_conflict_to_unreachable_n"],
                ]
                for r in transitions
            ],
        ),
        "A只表示当前保存的模型可见上下文没有达到Gold匹配门槛；可能涉及k范围、语料/排名、上下文过滤/截断、等价证据未被Gold覆盖或匹配不确定。B说明在这些已观测预算上存在质量与证据约束冲突，不能证明额外证据导致回答变差。unknown保留上下界及计数，未当作确定缺失，未自动使用upper或回退。所有τ、所有1000题的完整曲线都在empty_candidate_per_question.jsonl。",
        "## 7. 具体案例与完整曲线",
        "A/B按类型及qid字典序各选前两题；Utility与Minimum不同的3题全部展示。案例是目的性选例，不代表总体频率。所有案例保留问题、Gold、15点F1/Evidence/Token和预测；未选取收益最大者代替总体。",
        "![部分案例的F1和Evidence曲线](figures/example_curves.png)",
    ]
    names = {
        "evidence_unreachable": "A：全部预算证据门槛未达",
        "constraint_conflict": "B：F1与Evidence约束冲突",
        "utility_vs_minimum_different": "Utility与Minimum预算不同",
    }
    for e in examples:
        parts.extend(
            [
                "### {} / {} / {}".format(names[e["example_category"]], e["type"], e["question_id"]),
                "问题：{}；Gold：{}。本类型该类别共{}题。".format(
                    e["question_text"].strip(), e["gold_answer"], e["category_n_in_type"]
                ),
                "原Hard k={}，Minimum k={}，Utility k={}；max E={}，证据达标集合最佳F1={}，相对全局最大F1差={}。".format(
                    e["hard_oracle_k"],
                    e["min_feasible_k"],
                    e["utility_k"],
                    fmt(e["max_evidence_recall"]),
                    fmt(e["best_f1_with_evidence"]),
                    fmt(e["global_to_evidence_f1_gap"]),
                ),
                md_table(
                    ["k", "F1", "E lower", "E upper", "unknown数", "Context Tokens", "Utility", "概率"],
                    [
                        [
                            k,
                            fmt(e["f1_curve"][i]),
                            fmt(e["evidence_recall_curve"][i]),
                            fmt(e["sentence_recall_upper_curve"][i]),
                            e["sentence_unknown_count_curve"][i],
                            e["context_tokens_curve"][i],
                            fmt(e["utility_curve"][i]),
                            fmt(e["soft_label"][i]) if e["soft_label"] is not None else "null",
                        ]
                        for i, k in enumerate(e["k_values"])
                    ],
                ),
            ]
        )
    parts.extend(
        [
            "## 8. 训练可行性与仍待解决的问题",
            "1. **预算决策基线。** Minimum Feasible与当前Utility几乎相同，未来任何学习方案都需要与该离线基线和可部署固定k基线比较；它本身仍使用所有k的Gold结果，不能在部署时直接获得。小幅离线优势是否值得增加目标复杂度，尚无训练证据。",
            "2. **先确定分布用途。** 若目标是近优候选之间保留替代预算，平滑分布有用途；若目标是最小满足条件的预算，强成本偏好可能贴近Minimum。应预先定义质量容忍、绝对成本、证据目标、概率到预算的规则，不能只追求低熵。",
            "3. **类型偏差与缺失监督。** Bridge全零与A类更多，筛选后训练题的类型比例偏离原500+500；无标签并不表示容易或不需要检索。训练前必须定义这类问题如何参与、评估覆盖与回退，不能只汇报有效子集性能。",
            "4. **证据语义。** τ=0.5只是部分Gold句匹配，τ=1也不是语义充分性的证明；应复核A类、B类、正确但Gold不足和unknown样本。当前数据不能区分语料/排名问题、过滤截断、Gold替代路径和匹配误差的贡献。",
            "5. **泛化与数据划分。** 这些离线标签使用已知答案和全预算结果；尚未验证Query-only能识别预算。未来训练/选参/测试应分开，现开发集不能兼作调参和泛化证据。也没有量化单次生成的随机性和曲线稳定性。",
            "结论：本次诊断足以说明主要样本流失机制，并给出可控的尖锐度范围；仍不足以据此选出最佳训练参数或宣称可学习。建议研究者先明确目标与无标签政策、人工复核代理偏差，再决定是否开展独立的小规模训练验证。本轮仅完成离线诊断。",
            "## 9. 输出、验证与复现",
            "新增文件及完整命令见本子目录README。输出含3策略公平比较、全部1000题逐题预算、{}组参数的{}条逐题概率、{}个τ的全部{}条状态与曲线、分位数/直方图、案例及6张PNG。metadata.json记录输入与旧结果SHA256、复用代码与新代码SHA256、运行配置、命令、时区、全体已有文件快照；validation.json记录落盘重读和参数不变量核验。输出目录存在即拒绝，默认自动分配新编号。".format(
                dimension_n, dimension_n * n, len(config["tau_values"]), len(config["tau_values"]) * len(questions)
            ),
        ]
    )
    (output / "report.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def validate_results(output, questions, labels, sharp_records, sweep, reasons, transitions):
    saved = read_rows(output / "sharpness_per_question.jsonl")
    require(len(saved) == len(sharp_records), "Saved sharpness row count mismatch")
    qs = {q.question_id: q for q in questions}
    valid_ids = {q.question_id for q, r in zip(questions, labels) if r["label_status"] == "valid"}
    lambdas = sorted({r["lambda"] for r in saved})
    temps = sorted({r["temperature"] for r in saved})
    lookup = {(r["question_id"], r["lambda"], r["temperature"]): r for r in saved}
    require(
        len(lookup) == len(saved) and set(lookup) == set(itertools.product(valid_ids, lambdas, temps)),
        "Incomplete/duplicate sharpness keys",
    )
    for r in saved:
        q = qs[r["question_id"]]
        ps = r["soft_label"]
        mask = r["candidate_mask"]
        require(len(ps) == len(q.ks) and all(math.isfinite(p) and p >= 0 for p in ps), "Invalid probability")
        close(sum(ps), 1, "Probability sum")
        require(all(p == 0 for p, yes in zip(ps, mask) if not yes), "Invalid candidate has mass")
        close(
            math.fsum(p * t for p, t in zip(ps, q.tokens["context_tokens"])),
            r["expected_context_tokens"],
            "Expected cost",
        )
        close(math.fsum(p * f for p, f in zip(ps, q.f1)), r["expected_f1"], "Expected F1")
        require(r["argmax_k"] in [k for k, yes in zip(q.ks, mask) if yes], "Argmax not feasible")
        close(ps[q.ks.index(r["argmax_k"])], max(ps), "Utility argmax vs probabilities")
        if r["candidate_count"] == 1:
            require(
                r["normalized_entropy"] is None
                and r["top1_top2_probability_gap"] is None
                and r["top_utility_gap"] is None,
                "Singleton undefined metrics",
            )
    for qid in valid_ids:
        for lambda_value in lambdas:
            ordered = [lookup[qid, lambda_value, t] for t in temps]
            require(len({r["argmax_k"] for r in ordered}) == 1, "Temperature changed argmax")
            require(
                all(r["utility_curve"] == ordered[0]["utility_curve"] for r in ordered), "Temperature changed utilities"
            )
            require(
                all(b["entropy"] + BOUNDARY_ATOL >= a["entropy"] for a, b in zip(ordered, ordered[1:])),
                "Entropy did not increase with temperature",
            )
        for t in temps:
            ordered = [lookup[qid, lambda_value, t] for lambda_value in lambdas]
            require(
                all(b["argmax_context_tokens"] <= a["argmax_context_tokens"] for a, b in zip(ordered, ordered[1:])),
                "Cost increased with lambda",
            )
    comparison_saved = read_rows(output / "min_feasible_per_question.csv")
    require(
        len(comparison_saved) == len(questions) and len({r["question_id"] for r in comparison_saved}) == len(questions),
        "Budget qid count mismatch",
    )
    for q, label, r in zip(questions, labels, comparison_saved):
        require(q.question_id == r["question_id"], "Budget qid alignment mismatch")
        if label["label_status"] == "valid":
            k = min(label["valid_candidates"], key=lambda k: (q.tokens["context_tokens"][q.ks.index(k)], k))
            require(int(r["min_feasible_k"]) == k, "Wrong minimum budget")
        else:
            require(r["min_feasible_k"] == r["utility_k"] == "", "Invalid budget must be empty")
    empty_saved = read_rows(output / "empty_candidate_per_question.jsonl")
    empty_keys = {(r["question_id"], r["tau"]) for r in empty_saved}
    require(
        len(empty_keys) == len(empty_saved) == len(questions) * len({r["tau"] for r in empty_saved}),
        "Empty-reason keys incomplete",
    )
    for r in empty_saved:
        fs = r["f1_curve"]
        es = r["evidence_recall_curve"]
        passing = [i for i, e in enumerate(es) if e + BOUNDARY_ATOL >= r["tau"]]
        joint = [i for i in passing if fs[i] + BOUNDARY_ATOL >= max(fs) - r["epsilon"]]
        expected = (
            "all_zero_f1"
            if max(fs) == 0
            else "valid" if joint else "evidence_unreachable" if not passing else "constraint_conflict"
        )
        require(r["category"] == expected, "Reason classification mismatch")
    for r in reasons:
        part = [
            e for e in empty_saved if e["tau"] == r["tau"] and (r["group"] == "combined" or e["type"] == r["group"])
        ]
        counts = Counter(e["category"] for e in part)
        require(
            counts["valid"] == r["valid_n"]
            and counts["evidence_unreachable"] == r["evidence_unreachable_n"]
            and counts["constraint_conflict"] == r["constraint_conflict_n"]
            and counts["all_zero_f1"] == r["all_zero_f1_n"],
            "Reason summary mismatch",
        )
    for r in sweep:
        part = [
            s
            for s in saved
            if s["lambda"] == r["lambda"]
            and s["temperature"] == r["temperature"]
            and (r["group"] == "combined" or s["type"] == r["group"])
        ]
        require(len(part) == r["evaluated_n"] == sum(r["argmax_k_hist"].values()), "Sharpness denominator mismatch")
        close(mean(s["entropy"] for s in part), r["mean_entropy"], "Saved mean entropy")
        close(
            mean(s["expected_context_tokens"] for s in part), r["expected_context_tokens"], "Saved expected cost mean"
        )
    require(len(read_rows(output / "sharpness_sweep.csv")) == len(sweep), "Saved sweep count mismatch")
    require(
        len(read_rows(output / "empty_candidate_transitions.csv")) == len(transitions),
        "Saved transitions count mismatch",
    )
    return {
        "passed": True,
        "query_count": len(questions),
        "pair_count": sum(len(q.ks) for q in questions),
        "valid_n": len(valid_ids),
        "sharpness_per_question_n": len(saved),
        "sharpness_summary_n": len(sweep),
        "empty_per_question_n": len(empty_saved),
        "reason_summary_n": len(reasons),
        "transition_summary_n": len(transitions),
        "saved_probabilities_expected_values_and_nulls_checked": True,
        "temperature_argmax_and_utility_invariance": True,
        "entropy_monotonic_in_temperature": True,
        "argmax_cost_nonincreasing_in_lambda": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--output", help="New output directory; default allocates a followup experiment_NNN")
    args = parser.parse_args()
    import matplotlib  # noqa: F401

    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    base_config, reference, questions, input_validation, metadata = context(ROOT / config["base_config"])
    require(
        reference.scheme == "D"
        and reference.cost_metric == "context_tokens"
        and reference.evidence_metric == "sentence_recall_lower",
        "Followup requires D/context_tokens/sentence_recall_lower",
    )
    for field in ("lambda_values", "temperature_values", "tau_values"):
        values = config[field]
        require(values and len(values) == len(set(values)), "Empty/duplicate grid: " + field)
        for value in values:
            replace(
                reference,
                **{
                    {"lambda_values": "lambda_", "temperature_values": "temperature", "tau_values": "tau"}[field]: value
                },
            )
    number(config["near_one_hot_threshold"], "near_one_hot_threshold", 0, 1)
    require(
        reference.lambda_ in config["lambda_values"]
        and reference.temperature in config["temperature_values"]
        and reference.tau in config["tau_values"],
        "Reference must be in grids",
    )
    legacy = snapshot_existing()
    source = (ROOT / config["source_experiment"]).resolve()
    source_inputs = {
        str(p): sha256(p)
        for p in (
            source / "labels.jsonl",
            source / "parameter_sweep.csv",
            source / "report.md",
            source / "metadata.json",
            source / "validation.json",
        )
    }
    labels = [build_label(q, reference) for q in questions]
    old_sweep = verify_source_labels(questions, labels, source)
    budget_rows = budget_records(questions, labels, reference)
    comparison, comparison_details = compare_budgets(questions, budget_rows)
    output = allocate_followup_output(args.output)
    print("Output: " + str(output), flush=True)
    write_csv(output / "min_feasible_comparison.csv", comparison)
    write_csv(output / "min_feasible_per_question.csv", budget_rows)
    sharp_rows, sharp_details, per_query = [], [], []
    for lambda_, temperature in itertools.product(config["lambda_values"], config["temperature_values"]):
        params = replace(reference, lambda_=lambda_, temperature=temperature)
        records = [
            sharpness_record(q, build_label(q, params), params)
            for q, label in zip(questions, labels)
            if label["label_status"] == "valid"
        ]
        per_query.extend(records)
        for group in GROUPS:
            row, detail = summarize_sharpness(
                records, params, group, len(group_indices(questions, group)), config["near_one_hot_threshold"]
            )
            sharp_rows.append(row)
            sharp_details.append(detail)
        print("Sharpness lambda={}, T={}".format(lambda_, temperature), flush=True)
    write_csv(output / "sharpness_sweep.csv", sharp_rows)
    write_jsonl(output / "sharpness_per_question.jsonl", per_query)
    empty_rows = [classify_empty(q, replace(reference, tau=tau)) for tau in config["tau_values"] for q in questions]
    reasons = summarize_empty(empty_rows)
    transitions = empty_transitions(empty_rows)
    write_csv(output / "empty_candidate_analysis.csv", reasons)
    write_jsonl(output / "empty_candidate_per_question.jsonl", empty_rows)
    write_csv(output / "empty_candidate_transitions.csv", transitions)
    examples = choose_examples(questions, budget_rows, empty_rows, reference)
    write_jsonl(output / "examples.jsonl", examples)
    write_json(
        output / "distribution_summary.json", {"budget_comparison": comparison_details, "sharpness": sharp_details}
    )
    figure_metadata = make_figures(output, comparison_details, sharp_rows, sharp_details, reasons, examples, reference)
    make_report(
        output,
        questions,
        reference,
        config,
        comparison_details,
        sharp_rows,
        sharp_details,
        reasons,
        transitions,
        examples,
    )
    validation = validate_results(output, questions, labels, per_query, sharp_rows, reasons, transitions)
    validation["existing_sweep_overlap_rows_checked"] = verify_overlap(sharp_rows, old_sweep)
    validation["original_input_validation"] = input_validation
    assert_inputs_unchanged(metadata)
    for name, expected in source_inputs.items():
        require(sha256(name) == expected, "Source result changed: " + name)
    changed = [
        name for name, expected in legacy.items() if not (ROOT / name).is_file() or sha256(ROOT / name) != expected
    ]
    require(not changed, "Preexisting files changed: " + str(changed[:5]))
    validation["preexisting_files_checked"] = len(legacy)
    validation["changed_preexisting_files"] = changed
    write_json(output / "validation.json", validation)
    metadata.update(
        {
            "followup_version": VERSION,
            "configuration": config,
            "followup_config_path": str(config_path),
            "followup_config_sha256": sha256(config_path),
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "python_version": platform.python_version(),
            "runtime_parameters": reference.to_dict(),
            "base_configuration": base_config,
            "source_results_sha256": source_inputs,
            "plotting": figure_metadata,
            "top_two_policy": "top two within feasible set; singleton gap/normalized entropy=null; distributions exclude nulls",
            "near_one_hot_policy": "pmax>=configured threshold; singletons and multi-candidate samples reported separately",
            "empty_classification_policy": "all_zero_f1 separate; A=no evidence-pass k; B=evidence-pass k exist but no joint candidate; use original 1e-12 boundary tolerance",
            "snapshot_sha256": legacy,
            "code_sha256": {
                str(p.relative_to(ROOT)): sha256(p)
                for p in (BASE / "build_soft_oracle.py", BASE / "analyze_soft_oracle.py", *sorted(HERE.rglob("*.py")))
            },
            "output_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        }
    )
    write_json(output / "metadata.json", metadata)
    print(json.dumps(validation, ensure_ascii=False), flush=True)
    print("Completed: " + str(output), flush=True)


if __name__ == "__main__":
    main()
