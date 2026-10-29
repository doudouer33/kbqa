#!/usr/bin/env python3
"""Full offline sensitivity/ablation analysis; matplotlib is used only for figures."""

import argparse
import itertools
import json
import math
import statistics
from collections import Counter
from dataclasses import replace

from build_soft_oracle import (
    HERE,
    SCHEMES,
    allocate_output,
    assert_inputs_unchanged,
    build_label,
    context,
    label_record,
    require,
    save_labels,
    sha256,
    write_csv,
    write_json,
)

GROUPS = ("combined", "bridge", "comparison")


def mean(values):
    values = list(values)
    return (
        statistics.fmean(values)
        if values and hasattr(statistics, "fmean")
        else sum(values) / len(values) if values else None
    )


def ratio(a, b):
    return a / b if b else None


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0, "mean": None, "min": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None}

    def quantile(p):
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)

    return {
        "n": len(values),
        "mean": mean(values),
        "min": values[0],
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "p90": quantile(0.9),
        "max": values[-1],
    }


def histogram(values, bins):
    counts = Counter(values)
    return {str(v): counts[v] for v in bins}


def binned(values, edges):
    counts = [0] * (len(edges) - 1)
    for value in values:
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            if lo <= value < hi or i == len(counts) - 1 and lo <= value <= hi + 1e-12:
                counts[i] += 1
                break
        else:
            raise ValueError("Histogram value out of range: " + str(value))
    return {"edges": edges, "counts": counts, "last_bin_right_inclusive": True}


def group_indices(questions, group):
    return [i for i, q in enumerate(questions) if group == "combined" or q.type == group]


def feasibility(questions, labels, indices, params, group):
    part = [labels[i] for i in indices]
    nonzero = [r for r in part if r["label_status"] != "all_zero_f1"]
    valid = [r for r in part if r["label_status"] == "valid"]
    n, nv = len(part), len(valid)
    empty = sum(r["label_status"] == "empty_candidates" for r in part)
    zero = sum(r["label_status"] == "all_zero_f1" for r in part)
    singleton = sum(r["valid_candidate_count"] == 1 for r in valid)
    use_f1, use_ev = SCHEMES[params.scheme]
    excluded = Counter()
    excluded_nonzero = Counter()
    for r in part:
        for a, b in zip(r["f1_pass"], r["evidence_pass"]):
            af, be = use_f1 and not a, use_ev and not b
            category = "both" if af and be else "f1_only" if af else "evidence_only" if be else "neither"
            excluded[category] += 1
            if r["label_status"] != "all_zero_f1":
                excluded_nonzero[category] += 1
    require(nv + empty + zero == n, "Feasibility partition mismatch")
    return {
        "scheme": params.scheme,
        "epsilon": params.epsilon,
        "tau": params.tau,
        "evidence_metric": params.evidence_metric,
        "group": group,
        "population_n": n,
        "valid_n": nv,
        "valid_ratio": ratio(nv, n),
        "empty_nonzero_n": empty,
        "empty_nonzero_ratio": ratio(empty, n),
        "all_zero_f1_n": zero,
        "all_zero_f1_ratio": ratio(zero, n),
        "constraint_empty_all_n": sum(r["valid_candidate_count"] == 0 for r in part),
        "singleton_n": singleton,
        "singleton_ratio_population": ratio(singleton, n),
        "singleton_ratio_valid": ratio(singleton, nv),
        "multiple_n": nv - singleton,
        "multiple_ratio_population": ratio(nv - singleton, n),
        "multiple_ratio_valid": ratio(nv - singleton, nv),
        "mean_candidate_count_valid": mean(r["valid_candidate_count"] for r in valid),
        "mean_candidate_count_all": mean(r["valid_candidate_count"] for r in part),
        "mean_candidate_count_nonzero": mean(r["valid_candidate_count"] for r in nonzero),
        "candidate_count_hist_all": histogram([r["valid_candidate_count"] for r in part], range(16)),
        "candidate_count_hist_nonzero": histogram([r["valid_candidate_count"] for r in nonzero], range(16)),
        "candidate_count_hist_valid": histogram([r["valid_candidate_count"] for r in valid], range(16)),
        "excluded_partition_all": dict(excluded),
        "excluded_partition_nonzero": dict(excluded_nonzero),
        "f1_excluded_all": excluded["f1_only"] + excluded["both"],
        "evidence_excluded_all": excluded["evidence_only"] + excluded["both"],
        "f1_excluded_nonzero": excluded_nonzero["f1_only"] + excluded_nonzero["both"],
        "evidence_excluded_nonzero": excluded_nonzero["evidence_only"] + excluded_nonzero["both"],
        "k1_candidate_nonzero_n": sum(r["candidate_mask"][0] for r in nonzero),
        "unknown_sentence_query_n": sum(r["has_unknown_sentence_evidence"] for r in part),
    }


def fixed_metrics(questions, indices, k):
    subset = [questions[i] for i in indices]
    j = k - 1
    return {
        "k": k,
        "n": len(subset),
        "f1": mean(q.f1[j] for q in subset),
        "em": mean(q.em[j] for q in subset),
        "context_tokens": mean(q.tokens["context_tokens"][j] for q in subset),
        "prompt_tokens": mean(q.tokens["prompt_tokens"][j] for q in subset),
        "total_tokens": mean(q.tokens["total_tokens"][j] for q in subset),
        "sentence_recall_lower": mean(q.evidence[j]["sentence_recall_lower"] for q in subset),
    }


def summarize(questions, labels, indices):
    """All metrics below use exactly these valid qids, including hard/fixed baselines."""
    require(all(labels[i]["label_status"] == "valid" for i in indices), "Invalid label in evaluation subset")
    pairs = [(questions[i], labels[i]) for i in indices]
    n = len(pairs)
    ks = questions[0].ks
    maxp = [max(r["soft_label"]) for _, r in pairs]
    entropies = [r["entropy"] for _, r in pairs]
    candidate_entropies = [
        r["entropy_normalized_candidates"] for _, r in pairs if r["entropy_normalized_candidates"] is not None
    ]
    probability_mass = [mean(r["soft_label"][j] for _, r in pairs) for j in range(len(ks))]
    argmax_hist = histogram([r["argmax_soft_label_k"] for _, r in pairs], ks)
    hard_hist = histogram([q.original_hard_oracle_k for q, _ in pairs], ks)
    transition = Counter((q.original_hard_oracle_k, r["argmax_soft_label_k"]) for q, r in pairs)

    def selected(field):
        return mean(getattr(q, field)[r["argmax_soft_label_k"] - 1] for q, r in pairs)

    def expected(field):
        return mean(math.fsum(p * v for p, v in zip(r["soft_label"], getattr(q, field))) for q, r in pairs)

    quality = {
        "max_f1": mean(max(q.f1) for q, _ in pairs),
        "argmax_f1": selected("f1"),
        "expected_f1": expected("f1"),
        "argmax_em": selected("em"),
        "expected_em": expected("em"),
        "argmax_f1_loss": mean(max(q.f1) - q.f1[r["argmax_soft_label_k"] - 1] for q, r in pairs),
        "expected_f1_loss": mean(max(q.f1) - math.fsum(p * v for p, v in zip(r["soft_label"], q.f1)) for q, r in pairs),
        "argmax_evidence": mean(
            q.evidence[r["argmax_soft_label_k"] - 1][r["parameters"]["evidence_metric"]] for q, r in pairs
        ),
        "expected_evidence": mean(
            math.fsum(p * e[r["parameters"]["evidence_metric"]] for p, e in zip(r["soft_label"], q.evidence))
            for q, r in pairs
        ),
        "argmax_k": mean(r["argmax_soft_label_k"] for _, r in pairs),
        "expected_k": mean(math.fsum(p * k for p, k in zip(r["soft_label"], ks)) for _, r in pairs),
        "hard_oracle_f1": mean(q.f1[q.original_hard_oracle_k - 1] for q, _ in pairs),
        "hard_oracle_evidence": mean(
            q.evidence[q.original_hard_oracle_k - 1][r["parameters"]["evidence_metric"]] for q, r in pairs
        ),
        "hard_oracle_context_tokens": mean(q.tokens["context_tokens"][q.original_hard_oracle_k - 1] for q, _ in pairs),
    }
    for metric in ("context_tokens", "prompt_tokens", "total_tokens"):
        quality["argmax_" + metric] = mean(q.tokens[metric][r["argmax_soft_label_k"] - 1] for q, r in pairs)
        quality["expected_" + metric] = mean(
            math.fsum(p * v for p, v in zip(r["soft_label"], q.tokens[metric])) for q, r in pairs
        )
    fixed = [fixed_metrics(questions, indices, k) for k in ks]
    differences = {}
    for k in (1, 13):
        if k not in ks:
            continue
        baseline = fixed[k - 1]
        for strategy in ("argmax", "expected"):
            for metric in ("f1", "context_tokens"):
                differences[strategy + "_minus_fixed{}_".format(k) + metric] = (
                    quality[strategy + "_" + metric] - baseline[metric] if n else None
                )
    return {
        "n": n,
        "mean_probability_mass_by_k": probability_mass,
        "argmax_k_hist": argmax_hist,
        "hard_oracle_k_hist": hard_hist,
        "argmax_differs_from_hard_n": sum(q.original_hard_oracle_k != r["argmax_soft_label_k"] for q, r in pairs),
        "argmax_differs_from_hard_ratio": ratio(
            sum(q.original_hard_oracle_k != r["argmax_soft_label_k"] for q, r in pairs), n
        ),
        "mean_absolute_k_shift": mean(abs(q.original_hard_oracle_k - r["argmax_soft_label_k"]) for q, r in pairs),
        "hard_to_soft_argmax_transitions": {
            "{}->{}".format(a, b): count for (a, b), count in sorted(transition.items())
        },
        "entropy": distribution(entropies),
        "entropy_normalized_all_k": distribution(r["entropy_normalized_all_k"] for _, r in pairs),
        "entropy_normalized_candidates": distribution(candidate_entropies),
        "max_probability": distribution(maxp),
        "entropy_hist": binned(entropies, [0, 0.5, 1, 1.5, 2, 2.5, math.log(15)]),
        "max_probability_hist": binned(maxp, [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]),
        "candidate_count_hist": histogram([r["valid_candidate_count"] for _, r in pairs], range(1, 16)),
        "entropy_by_candidate_count": {
            str(c): {
                "entropy": distribution(r["entropy"] for _, r in pairs if r["valid_candidate_count"] == c),
                "normalized": distribution(
                    r["entropy_normalized_candidates"] for _, r in pairs if r["valid_candidate_count"] == c
                ),
            }
            for c in range(1, 16)
        },
        "quality_cost": quality,
        "fixed_k": fixed,
        "fixed_differences": differences,
    }


def flat_summary(params, group, scope, population_n, own_n, metrics):
    row = {
        **params.to_dict(),
        "group": group,
        "scope": scope,
        "population_n": population_n,
        "own_valid_n": own_n,
        "evaluated_n": metrics["n"],
        "coverage_of_population": ratio(metrics["n"], population_n),
        "mean_entropy": metrics["entropy"]["mean"],
        "mean_entropy_normalized_all_k": metrics["entropy_normalized_all_k"]["mean"],
        "mean_entropy_normalized_candidates": metrics["entropy_normalized_candidates"]["mean"],
        "candidate_normalized_entropy_n": metrics["entropy_normalized_candidates"]["n"],
        "mean_max_probability": metrics["max_probability"]["mean"],
        "argmax_differs_from_hard_ratio": metrics["argmax_differs_from_hard_ratio"],
        "mean_probability_mass_by_k": metrics["mean_probability_mass_by_k"],
        "argmax_k_hist": metrics["argmax_k_hist"],
        "hard_oracle_k_hist": metrics["hard_oracle_k_hist"],
        **metrics["quality_cost"],
        **metrics["fixed_differences"],
    }
    row["soft_mass_k1"] = metrics["mean_probability_mass_by_k"][0]
    row["argmax_k1_ratio"] = ratio(metrics["argmax_k_hist"]["1"], metrics["n"])
    return row


def scan(questions, config, reference):
    feasibility_rows, sweep_rows, detailed = [], [], []
    groups = {group: group_indices(questions, group) for group in GROUPS}
    for epsilon, tau in itertools.product(config["sweep"]["epsilon"], config["sweep"]["tau"]):
        base = replace(reference, epsilon=epsilon, tau=tau)
        first = {scheme: [build_label(q, replace(base, scheme=scheme)) for q in questions] for scheme in SCHEMES}
        common = {i for i in range(len(questions)) if all(first[s][i]["label_status"] == "valid" for s in SCHEMES)}
        require(
            common == {i for i, r in enumerate(first["D"]) if r["label_status"] == "valid"},
            "Ablation intersection must equal D",
        )
        for scheme in SCHEMES:
            for group, indices in groups.items():
                feasibility_rows.append(
                    feasibility(questions, first[scheme], indices, replace(base, scheme=scheme), group)
                )
        for lambda_, temperature in itertools.product(config["sweep"]["lambda"], config["sweep"]["temperature"]):
            for scheme in SCHEMES:
                params = replace(base, lambda_=lambda_, temperature=temperature, scheme=scheme)
                labels = [build_label(q, params) for q in questions]
                for group, population in groups.items():
                    own = [i for i in population if labels[i]["label_status"] == "valid"]
                    for scope, indices in (
                        ("own_valid", own),
                        ("common_valid", [i for i in population if i in common]),
                    ):
                        metrics = summarize(questions, labels, indices)
                        sweep_rows.append(flat_summary(params, group, scope, len(population), len(own), metrics))
                        detailed.append(
                            {
                                "parameters": params.to_dict(),
                                "group": group,
                                "scope": scope,
                                "population_n": len(population),
                                "metrics": metrics,
                            }
                        )
        print("Finished epsilon={}, tau={}".format(epsilon, tau), flush=True)
    return feasibility_rows, sweep_rows, detailed


def select_examples(questions, labels, params):
    categories = {
        "k1_correct_evidence_incomplete": lambda q, r: q.em[0] == 1 and q.evidence[0][params.evidence_metric] < 1,
        "multiple_candidates": lambda q, r: r["label_status"] == "valid" and r["valid_candidate_count"] > 1,
        "single_candidate": lambda q, r: r["label_status"] == "valid" and r["valid_candidate_count"] == 1,
        "empty_candidates": lambda q, r: r["label_status"] == "empty_candidates",
        "all_zero_f1": lambda q, r: r["label_status"] == "all_zero_f1",
        "f1_decreases_with_k": lambda q, r: any(b < a for a, b in zip(q.f1, q.f1[1:])),
        "unknown_evidence": lambda q, r: r["has_unknown_sentence_evidence"],
    }
    output = []
    for category, predicate in categories.items():
        for group in ("bridge", "comparison"):
            pool = [(q, r) for q, r in zip(questions, labels) if q.type == group and predicate(q, r)]
            chosen = pool[:1]  # Questions are sorted by qid, no outcome-based ranking.
            for q, r in chosen:
                output.append(
                    {
                        "category": category,
                        "type": group,
                        "category_population_n": len(pool),
                        "selection_rule": "lexicographically first question_id within category and type",
                        **label_record(q, r),
                        "predictions": q.predictions,
                        "adjacent_f1_drop_pairs": [
                            [a + 1, a + 2] for a in range(len(q.ks) - 1) if q.f1[a + 1] < q.f1[a]
                        ],
                    }
                )
            if not chosen:
                output.append({"category": category, "type": group, "category_population_n": 0, "question_id": None})
    return output


def get_row(rows, params, group="combined", scope="own_valid", scheme=None):
    p = params.to_dict()
    if scheme:
        p["scheme"] = scheme
    found = [r for r in rows if r["group"] == group and r["scope"] == scope and all(r[k] == v for k, v in p.items())]
    require(len(found) == 1, "Expected one sweep row: " + str(p))
    return found[0]


def make_figures(output, questions, labels, feasibility_rows, sweep_rows, reference):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir()
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    colors = dict(zip(SCHEMES, ("#8f9bb3", "#4c8cbc", "#e4a34c", "#40886c")))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    eps = sorted({r["epsilon"] for r in feasibility_rows})
    taus = sorted({r["tau"] for r in feasibility_rows})
    for ax, group in zip(axes, GROUPS):
        table = [
            [
                next(
                    r["valid_ratio"]
                    for r in feasibility_rows
                    if r["scheme"] == "D" and r["group"] == group and r["epsilon"] == e and r["tau"] == t
                )
                for t in taus
            ]
            for e in eps
        ]
        im = ax.imshow(table, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(taus)), [str(t) for t in taus])
        ax.set_yticks(range(len(eps)), [str(e) for e in eps])
        for y, line in enumerate(table):
            for x, value in enumerate(line):
                ax.text(
                    x, y, "{:.1%}".format(value), ha="center", va="center", color="white" if value > 0.6 else "black"
                )
        ax.set(xlabel="Evidence threshold tau", ylabel="F1 tolerance epsilon", title=group)
    fig.colorbar(im, ax=axes, label="Valid / original population", shrink=0.85)
    fig.suptitle("D: coverage (all-zero F1 excluded from labels, retained in denominators)")
    fig.savefig(directory / "feasible_ratio.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, group in zip(axes, GROUPS):
        indices = group_indices(questions, group)
        nonzero = [labels[i]["valid_candidate_count"] for i in indices if labels[i]["label_status"] != "all_zero_f1"]
        ax.bar(range(16), [nonzero.count(k) for k in range(16)], color=colors["D"])
        nz = sum(labels[i]["label_status"] == "all_zero_f1" for i in indices)
        ax.set(
            title="{} (all-zero excluded: {})".format(group, nz),
            xlabel="Candidate count; zero = empty",
            ylabel="Questions",
        )
        ax.set_xticks(range(0, 16, 3))
    fig.suptitle("Reference D: nonzero-F1 queries, including empty candidate sets")
    fig.tight_layout()
    fig.savefig(directory / "candidate_count.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for scheme in SCHEMES:
        r = get_row(sweep_rows, reference, scope="common_valid", scheme=scheme)
        axes[0].plot(range(1, 16), r["mean_probability_mass_by_k"], "o-", label=scheme, color=colors[scheme])
        axes[1].plot(
            range(1, 16),
            [r["argmax_k_hist"][str(k)] / r["evaluated_n"] for k in range(1, 16)],
            "o-",
            label=scheme,
            color=colors[scheme],
        )
    r = get_row(sweep_rows, reference, scope="common_valid")
    hard = [r["hard_oracle_k_hist"][str(k)] / r["evaluated_n"] for k in range(1, 16)]
    for ax in axes:
        ax.plot(range(1, 16), hard, "k--", label="Original hard oracle (same subset)")
        ax.set(xlabel="k", ylabel="Probability / proportion")
        ax.legend(fontsize=7)
    axes[0].set_title("Mean soft probability mass")
    axes[1].set_title("Argmax distribution")
    fig.suptitle("Reference A/B/C/D on the same {} queries".format(r["evaluated_n"]))
    fig.tight_layout()
    fig.savefig(directory / "label_distribution.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    common = [i for i, r in enumerate(labels) if r["label_status"] == "valid"]
    for scheme in SCHEMES:
        part = [build_label(questions[i], replace(reference, scheme=scheme)) for i in common]
        axes[0].hist(
            [r["entropy"] for r in part],
            bins=[0, 0.5, 1, 1.5, 2, 2.5, math.log(15)],
            histtype="step",
            linewidth=2,
            label=scheme,
            color=colors[scheme],
        )
        counts = sorted({r["valid_candidate_count"] for r in part if r["valid_candidate_count"] > 1})
        axes[1].plot(
            counts,
            [mean(r["entropy_normalized_candidates"] for r in part if r["valid_candidate_count"] == c) for c in counts],
            "o-",
            label=scheme,
            color=colors[scheme],
        )
    axes[0].set(xlabel="Entropy (nats)", ylabel="Queries", title="Entropy on common valid subset")
    axes[1].set(
        xlabel="Candidate count (>1)", ylabel="Mean H / log(candidate count)", title="Condition on candidate count"
    )
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "label_entropy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    fixed = [fixed_metrics(questions, common, k) for k in range(1, 16)]
    ax.plot(
        [r["context_tokens"] for r in fixed],
        [r["f1"] for r in fixed],
        ".--",
        color="#aaa",
        label="Fixed k=1..15 (same subset)",
    )
    for k in (1, 13, 15):
        f = fixed[k - 1]
        ax.annotate("k={}".format(k), (f["context_tokens"], f["f1"]), xytext=(4, -12), textcoords="offset points")
    for scheme in SCHEMES:
        r = get_row(sweep_rows, reference, scope="common_valid", scheme=scheme)
        for strategy, marker in (("argmax", "o"), ("expected", "X")):
            ax.scatter(
                r[strategy + "_context_tokens"],
                r[strategy + "_f1"],
                color=colors[scheme],
                marker=marker,
                s=65,
                label=scheme + " " + strategy,
            )
    ax.set(
        xlabel="Mean context tokens",
        ylabel="Mean Answer F1",
        title="Offline Oracle / Label Analysis; common valid n={}".format(len(common)),
    )
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "quality_cost_tradeoff.png")
    plt.close(fig)
    return {
        "matplotlib_version": matplotlib.__version__,
        "figure_count": 5,
        "figure_language": "English; Chinese captions in report",
    }


def percent(value):
    return "—" if value is None else "{:.2%}".format(value)


def dec(value, digits=3):
    return "—" if value is None else ("{:,." + str(digits) + "f}").format(value)


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |" for row in rows]
    )


def make_report(output, questions, labels, validation, feasibility_rows, sweep_rows, reference, examples, baselines):
    p = reference
    own = get_row(sweep_rows, p)
    common = own["evaluated_n"]
    ref_f = next(
        r
        for r in feasibility_rows
        if r["scheme"] == p.scheme and r["group"] == "combined" and r["epsilon"] == p.epsilon and r["tau"] == p.tau
    )
    bcommon = get_row(sweep_rows, p, scope="common_valid", scheme="B")
    dcommon = get_row(sweep_rows, p, scope="common_valid", scheme="D")
    costs_delta = dcommon["argmax_context_tokens"] - bcommon["argmax_context_tokens"]
    f1_delta = dcommon["argmax_f1"] - bcommon["argmax_f1"]
    zero = validation["all_zero_f1_count"]
    nonzero_count = len(questions) - zero
    numeric_count = len(sweep_rows) // (len(SCHEMES) * len(GROUPS) * 2)
    dimensions = [len({r[field] for r in sweep_rows}) for field in ("epsilon", "tau", "lambda", "temperature")]
    default_tau_rows = [
        r for r in feasibility_rows if r["scheme"] == "D" and r["group"] == "combined" and r["epsilon"] == p.epsilon
    ]
    epsilon_rows = [
        r for r in feasibility_rows if r["scheme"] == "D" and r["group"] == "combined" and r["tau"] == p.tau
    ]
    default_utility_rows = [
        r
        for r in sweep_rows
        if r["scheme"] == "D"
        and r["group"] == "combined"
        and r["scope"] == "own_valid"
        and r["epsilon"] == p.epsilon
        and r["tau"] == p.tau
    ]
    positive_flat_valid = sum(r["label_status"] == "valid" and len(set(q.f1)) == 1 for q, r in zip(questions, labels))
    parts = [
        "# HotpotQA 1000 题 Soft Oracle 构造与可行性分析",
        "范围：2026-10-14 正式实验，500 Bridge + 500 Comparison，k=1～15。本文所有质量/成本均为 **Offline Oracle / Label Analysis**，使用每题全部 k 的已知回答结果，不能视为仅输入 Query 的预测模型能力。未运行检索、LLM、训练或扩大数据。",
        "## 1. 主要判断",
        "参考配置 ε={}、τ={}、λ={}、T={}（预设示例，未择优），方案 D 有效 {}/1000（{}），非零但空候选 {} 题，全零 F1 {} 题；有效题中单候选 {} 题，多候选 {} 题（{}），平均候选 {} 个。".format(
            p.epsilon,
            p.tau,
            p.lambda_,
            p.temperature,
            common,
            percent(common / 1000),
            ref_f["empty_nonzero_n"],
            zero,
            ref_f["singleton_n"],
            ref_f["multiple_n"],
            percent(ref_f["multiple_ratio_valid"]),
            dec(ref_f["mean_candidate_count_valid"]),
        ),
        "Soft Label 在有效题上保留多个预算的概率与效用信息；但 argmax k=1 仍占 {}，k=1 平均概率质量为 {}。概率分散与 argmax 决策集中可以同时发生，不能据此宣布不均衡已经解决。".format(
            percent(own["argmax_k1_ratio"]), percent(own["soft_mass_k1"])
        ),
        "参考标签中 {} / {} 题的正值 F1 曲线完全平坦；这些题的预算概率差异来自成本偏好，而非已观察到的答案质量差异。按概率采样时的期望 Context Tokens 为 {}，明显高于取 argmax 的 {}（{} 倍）。后续必须明确如何将分布转成预算决策，否则平滑标签可能掩盖成本差异。".format(
            positive_flat_valid,
            common,
            dec(own["expected_context_tokens"], 1),
            dec(own["argmax_context_tokens"], 1),
            dec(own["expected_context_tokens"] / own["argmax_context_tokens"], 2),
        ),
        "在同一 {} 题上，B→D（加入 Evidence 约束）使 argmax 平均 Context Tokens 改变 {}，F1 改变 {} 个百分点，平均证据 Recall 改变 {} 个百分点。这是约束导致的离线选择差异，不能识别证据增加的因果收益；是否值得成本取决于应用是否要求 Gold 证据覆盖。".format(
            common,
            dec(costs_delta, 1),
            dec(f1_delta * 100),
            dec((dcommon["argmax_evidence"] - bcommon["argmax_evidence"]) * 100),
        ),
        "建议保留本轮方法作为可审计的候选标签方案，先检查证据阈值的语义、类型选择偏差与 Query 对预算需求的可预测性，再由研究者决定是否进入训练和扩大自然采样。当前结果不足以承诺泛化收益，也没有自动选取最佳参数。",
        "## 2. 数据完整性与字段依据",
        table(
            ["校验项", "结果"],
            [
                ["唯一 Query / (qid,k)", "{} / {}".format(validation["query_count"], validation["pair_count"])],
                ["Bridge / Comparison", "500 / 500"],
                ["重复 / 缺失 / F1、EM 不一致 / Token 异常", "0 / 0 / 0 / 0"],
                ["全零 F1", zero],
                [
                    "sentence unknown 的题 / 对数",
                    "{} / {}".format(validation["unknown_sentence_queries"], validation["unknown_sentence_pairs"]),
                ],
                ["最小 Context Tokens", validation["context_tokens_min"]],
                ["Context Tokens 相邻下降对数", validation["context_token_decrease_pairs"]],
            ],
        ),
        "实际曲线字段为 `scores[str(k)].f1/em`、`best_k`、`tied_best_ks`、`context_tokens[str(k)]`、`api_token_usage[str(k)]`；问题文本来自 processed dataset 的 `question_text`。Evidence CSV 同时有 `qid` 与 `question_id`，两列必须相同。四源按主键严格连接；分类型数量、每题完整 k、官方分数与原 Oracle 均核对，不静默填补或删除。",
        "复用 `evidence_diagnostic/analyze_evidence.py` 与原 `summary.md` 的已有定义：实际传入 LLM 的过滤后 contexts 经 `para_to_text(...,350)` 截断、添加标题；原诊断已核对 context SHA256。`sentence_recall_lower = sentence_hit / sentence_total`，upper 将 unknown 加入分子；本次只加载和核对计数/边界，不重做文本匹配或重跑旧脚本。unknown 表示不确定，不是确定未命中；Gold 字面覆盖也不等于所有等价证据路径。",
        "## 3. 构造规则、状态与统计分母",
        "Smax 从每题全部 15 个 k 计算；D 候选满足 F1≥Smax−ε 且 E≥τ。A 无约束，B 仅 F1，C 仅 Evidence，D 双约束。各方案均先按同一规则标记全零 F1，不生成训练标签。浮点阈值比较允许 1e-12 绝对舍入误差；原 Hard Oracle 并列判断沿用原始分数的精确相等。",
        "成本 C(k)=Tokens(k)/max_j Tokens(j)，分母始终取全部 k。U=F1−λC，仅在候选中做减最大值的稳定 Softmax。无效候选概率严格为 0；单候选是 One-hot。空候选不回退。`all_zero_f1` 优先于空候选状态：两类无标签记录的 `soft_label=null`、argmax/entropy=null，保留原始候选 mask 和数量；每题都保存在 labels.jsonl。",
        "所有有效率、非零空候选率、全零率的分母是原始 Combined=1000 / Bridge=500 / Comparison=500，三种状态互斥且合计为原组样本数。`constraint_empty_all_n` 另计原始约束空集，可能与全零重叠。候选数量同时保存全体、非零、有效三个直方图；下面平均候选数与单/多候选比例的分母均为有效题。候选排除统计按 (qid,k)，分别给出全体与非零分母，F1 排除和 Evidence 排除可能重叠。",
        "## 4. ε、τ 对候选集合的影响",
        "τ=0 是不施加实际证据限制的消融。下表每行属于方案 D，包含原组分母。",
        "固定参考 ε，τ 对有效数的影响为 {}；固定参考 τ，ε 扫描的有效数范围仅 {}～{}。因此本批数据的覆盖对证据阈值更敏感，放宽 F1 容忍度并未补回多数证据无解题。".format(
            "、".join("τ={}：{}题".format(r["tau"], r["valid_n"]) for r in default_tau_rows),
            min(r["valid_n"] for r in epsilon_rows),
            max(r["valid_n"] for r in epsilon_rows),
        ),
        table(
            ["组", "ε", "τ", "有效/原组", "非零空集", "全零", "单候选/有效", "多候选/有效", "平均候选"],
            [
                [
                    r["group"],
                    r["epsilon"],
                    r["tau"],
                    "{}/{} ({})".format(r["valid_n"], r["population_n"], percent(r["valid_ratio"])),
                    r["empty_nonzero_n"],
                    r["all_zero_f1_n"],
                    "{}/{}".format(r["singleton_n"], r["valid_n"]),
                    "{}/{}".format(r["multiple_n"], r["valid_n"]),
                    dec(r["mean_candidate_count_valid"]),
                ]
                for r in feasibility_rows
                if r["scheme"] == "D"
            ],
        ),
        "![不同 ε、τ 的有效率](figures/feasible_ratio.png)",
        "![参考配置非零题的候选数量，包含空候选](figures/candidate_count.png)",
        "参考配置候选排除的互斥分解（非零 F1 题；每题 15 个候选）：",
        table(
            ["组", "候选分母", "仅 F1 不过", "仅 Evidence 不过", "两者不过", "两者通过"],
            [
                [
                    r["group"],
                    (r["population_n"] - r["all_zero_f1_n"]) * 15,
                    *[
                        r["excluded_partition_nonzero"].get(key, 0)
                        for key in ("f1_only", "evidence_only", "both", "neither")
                    ],
                ]
                for r in feasibility_rows
                if r["scheme"] == "D" and r["epsilon"] == p.epsilon and r["tau"] == p.tau
            ],
        ),
        "## 5. 四种消融：覆盖率与各自/共同有效集",
        "首先分别看各方案自身有效集，性能行不能跨不同分母直接排序。随后固定 A∩B∩C∩D 的共同有效题（对同一 ε、τ 等于 D 有效集），比较约束自身的作用。全零题在四方案中处理完全一致。",
        table(
            [
                "组",
                "方案",
                "有效/原组",
                "平均熵",
                "平均最大概率",
                "k1概率质量",
                "argmax k1",
                "argmax F1",
                "argmax上下文",
            ],
            [
                [
                    g,
                    s,
                    "{}/{}".format(r["evaluated_n"], r["population_n"]),
                    dec(r["mean_entropy"]),
                    percent(r["mean_max_probability"]),
                    percent(r["soft_mass_k1"]),
                    percent(r["argmax_k1_ratio"]),
                    percent(r["argmax_f1"]),
                    dec(r["argmax_context_tokens"], 1),
                ]
                for g in GROUPS
                for s in SCHEMES
                for r in [get_row(sweep_rows, p, g, "own_valid", s)]
            ],
        ),
        "共同有效集公平对比：",
        table(
            [
                "组",
                "方案",
                "共同n",
                "argmax F1",
                "期望F1",
                "argmax Context",
                "期望Context",
                "argmax E",
                "期望 E",
                "与原Hard不同",
            ],
            [
                [
                    g,
                    s,
                    r["evaluated_n"],
                    percent(r["argmax_f1"]),
                    percent(r["expected_f1"]),
                    dec(r["argmax_context_tokens"], 1),
                    dec(r["expected_context_tokens"], 1),
                    percent(r["argmax_evidence"]),
                    percent(r["expected_evidence"]),
                    percent(r["argmax_differs_from_hard_ratio"]),
                ]
                for g in GROUPS
                for s in SCHEMES
                for r in [get_row(sweep_rows, p, g, "common_valid", s)]
            ],
        ),
        "约束的成对边际作用（共同有效集 Combined）：",
        table(
            ["对照", "Δargmax F1(百分点)", "Δ期望F1(百分点)", "Δargmax Context", "Δ期望Context", "Δargmax E(百分点)"],
            [
                [
                    a + "→" + b,
                    dec((rb["argmax_f1"] - ra["argmax_f1"]) * 100),
                    dec((rb["expected_f1"] - ra["expected_f1"]) * 100),
                    dec(rb["argmax_context_tokens"] - ra["argmax_context_tokens"], 1),
                    dec(rb["expected_context_tokens"] - ra["expected_context_tokens"], 1),
                    dec((rb["argmax_evidence"] - ra["argmax_evidence"]) * 100),
                ]
                for a, b in (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"))
                for ra, rb in [
                    (
                        get_row(sweep_rows, p, scope="common_valid", scheme=a),
                        get_row(sweep_rows, p, scope="common_valid", scheme=b),
                    )
                ]
            ],
        ),
        "F1 约束主要剔除超过容忍损失的预算；Evidence 约束可能机械排除较小 k，同时选择易于命中 Gold 的题。D 覆盖率下降既包括原有全零排除，也包括证据门槛与高 F1 区间无交集，必须分别看待。τ=0.5 只要求一半标注句覆盖，不能称作完整证据充分；τ=1 要求完整匹配，但会带来更强选择偏差。",
        "本批数据 k=1 的完整 Gold 支持句覆盖题数为 0，因此 τ=1 时 k=1 标签必然消失。这是候选规则的机械结果，不能作为标签更合理或回答更准确的证据。参考 D 相对非零{}题额外排除{}题（{}），而不是把全部{}个无标签题都归因于双重约束。".format(
            nonzero_count,
            ref_f["empty_nonzero_n"],
            percent(ref_f["empty_nonzero_n"] / nonzero_count),
            len(questions) - common,
        ),
        "## 6. Soft Label 分布、熵与原 Hard Oracle",
        "原 Hard Oracle 全体分布与参考 D 有效集上的原 Hard、Soft argmax、平均概率质量如下；同一行的三种分布均使用同一有效集。原全体 Hard 包含全零并列取 k=1，不能直接与已排除全零的标签对比推断改善。",
        table(
            ["组/集合", "n", "k=1…15 的数量或概率质量"],
            [
                [
                    g + " 原全体Hard",
                    baselines[g]["population_n"],
                    ", ".join(str(baselines[g]["hard_oracle_hist"][str(k)]) for k in range(1, 16)),
                ]
                for g in GROUPS
            ]
            + [
                [
                    g + " D有效集 " + name,
                    r["evaluated_n"],
                    ", ".join(dec(v, 4) if name == "平均Soft概率" else str(v) for v in values),
                ]
                for g in GROUPS
                for r in [get_row(sweep_rows, p, g)]
                for name, values in (
                    ("原Hard", [r["hard_oracle_k_hist"][str(k)] for k in range(1, 16)]),
                    ("Soft argmax", [r["argmax_k_hist"][str(k)] for k in range(1, 16)]),
                    ("平均Soft概率", r["mean_probability_mass_by_k"]),
                )
            ],
        ),
        "![共同有效集的标签分布](figures/label_distribution.png)",
        "H=−Σp ln p（nats），同时输出 H/ln15 与 H/ln|C|。单候选时 H=0，H/ln|C| 无定义，保存 null 并在均值中排除；对应有效计数单独输出。不能直接把较大候选集造成的高熵解释成更优监督。`distribution_summary.json` 对每组参数、方案、类型、评估集合保存熵/最大概率的分位数、直方图，并按候选数分层保存熵。",
        table(
            ["组", "方案", "平均H", "H/ln15", "多候选 H/lnC", "对应n", "最大概率均值"],
            [
                [
                    g,
                    s,
                    dec(r["mean_entropy"]),
                    dec(r["mean_entropy_normalized_all_k"]),
                    dec(r["mean_entropy_normalized_candidates"]),
                    r["candidate_normalized_entropy_n"],
                    dec(r["mean_max_probability"]),
                ]
                for g in GROUPS
                for s in SCHEMES
                for r in [get_row(sweep_rows, p, g, "common_valid", s)]
            ],
        ),
        "![熵分布与按候选数控制的归一化熵](figures/label_entropy.png)",
        "## 7. 离线质量与成本：严格保持相同评估题集",
        "全体 1000 题的固定 k 与原 Hard Oracle（无覆盖率筛选）：",
        table(
            ["组", "策略", "n/原组", "F1", "Context Tokens", "Gold句 Recall Lower"],
            [
                [
                    g,
                    "固定k=" + str(k),
                    "{}/{}".format(baselines[g]["population_n"], baselines[g]["population_n"]),
                    percent(r["f1"]),
                    dec(r["context_tokens"], 1),
                    percent(r["sentence_recall_lower"]),
                ]
                for g in GROUPS
                for k in (1, 13, 15)
                for r in [baselines[g]["fixed_k"][k - 1]]
            ]
            + [
                [
                    g,
                    "原Hard",
                    str(baselines[g]["population_n"]),
                    percent(baselines[g]["hard_oracle_f1"]),
                    dec(baselines[g]["hard_oracle_context_tokens"], 1),
                    percent(baselines[g]["hard_oracle_evidence"]),
                ]
                for g in GROUPS
            ],
        ),
        "参考 D 的有效子集（也等于参考 A/B/C/D 共同有效集）：",
        table(
            ["组", "策略", "覆盖n/原组", "F1", "Context Tokens", "相对逐题max F1损失"],
            [
                [
                    g,
                    strategy,
                    "{}/{}".format(r["evaluated_n"], r["population_n"]),
                    percent(r[strategy + "_f1"]),
                    dec(r[strategy + "_context_tokens"], 1),
                    percent(r[strategy + "_f1_loss"]),
                ]
                for g in GROUPS
                for r in [get_row(sweep_rows, p, g)]
                for strategy in ("argmax", "expected")
            ]
            + [
                [
                    g,
                    "固定k=" + str(k),
                    "{}/{}".format(len(indices), baselines[g]["population_n"]),
                    percent(r["f1"]),
                    dec(r["context_tokens"], 1),
                    percent(mean(max(questions[i].f1) - questions[i].f1[k - 1] for i in indices)),
                ]
                for g in GROUPS
                for indices in [[i for i in group_indices(questions, g) if labels[i]["label_status"] == "valid"]]
                for k in (1, 13)
                for r in [fixed_metrics(questions, indices, k)]
            ],
        ),
        "期望策略是按 Soft Label 对已有分数和成本加权，不是模型对答案的期望置信度。所有 D 候选的逐题 F1 损失≤ε，因此 argmax 与期望 F1 损失均受此界限约束；这是构造保证，不是模型学到的性能。无标签题保留在覆盖分母，本次未定义回退策略，因此不给出伪装为全体 1000 的 D 平均性能。原全体基线与子集结果不能直接相减。",
        "![共同题集上的质量成本关系](figures/quality_cost_tradeoff.png)",
        "## 8. 参数敏感性与类型差异",
        "完整扫描 {}={} 组数值参数，每组 4 方案、3 类型、2 集合（自身/共同），共 {} 行 parameter_sweep.csv；候选集合扫描 {} 行。A/B/C 中不启用的约束参数仍完整记录，可能出现相同统计，这是控制消融的预期，不是缺失实验。未自动优化任何指标。".format(
            "×".join(map(str, dimensions)), numeric_count, len(sweep_rows), len(feasibility_rows)
        ),
        "固定参考 ε、τ，改变 λ、T（D，各自有效集）：",
        "在该 λ/T 网格中，Combined 的期望 Context Tokens 范围为 {}～{}，平均熵为 {}～{}；argmax F1 仅在 {}～{} 间变化。该统计进一步说明，概率形状与实际选取最大概率预算的质量/成本行为需要分别评估。".format(
            dec(min(r["expected_context_tokens"] for r in default_utility_rows), 1),
            dec(max(r["expected_context_tokens"] for r in default_utility_rows), 1),
            dec(min(r["mean_entropy"] for r in default_utility_rows)),
            dec(max(r["mean_entropy"] for r in default_utility_rows)),
            percent(min(r["argmax_f1"] for r in default_utility_rows)),
            percent(max(r["argmax_f1"] for r in default_utility_rows)),
        ),
        table(
            ["组", "λ", "T", "n", "平均H", "最大概率均值", "argmax F1", "期望F1", "argmax Context", "期望Context"],
            [
                [
                    g,
                    r["lambda"],
                    r["temperature"],
                    r["evaluated_n"],
                    dec(r["mean_entropy"]),
                    percent(r["mean_max_probability"]),
                    percent(r["argmax_f1"]),
                    percent(r["expected_f1"]),
                    dec(r["argmax_context_tokens"], 1),
                    dec(r["expected_context_tokens"], 1),
                ]
                for g in GROUPS
                for r in sweep_rows
                if r["group"] == g
                and r["scope"] == "own_valid"
                and r["scheme"] == "D"
                and r["epsilon"] == p.epsilon
                and r["tau"] == p.tau
            ],
        ),
        "T 只改变概率平滑程度，正温度下不改变 Utility 的排序，因此同 λ 的 argmax 应恒定；λ 改变质量与相对成本的权衡。若候选 F1 并列，分布由 λ/T 和成本差决定，高熵未必意味着强回答质量信号。固定 ε、τ 的候选集不受 λ、T 影响。每题最大 Token 归一化后，同一个相对成本在不同题的绝对 Token 含义不同。",
        "τ 敏感性若各用自身有效集会混入题集变化。下面将 D 在全部扫描 τ 下共同有效的题固定（按每题 mask 重算），再比较阈值，λ、T、ε 保持参考：",
    ]
    tau_values = sorted({r["tau"] for r in feasibility_rows})
    tau_labels = {t: [build_label(q, replace(p, tau=t)) for q in questions] for t in tau_values}
    tau_rows = []
    for group in GROUPS:
        indices = [
            i
            for i in group_indices(questions, group)
            if all(tau_labels[t][i]["label_status"] == "valid" for t in tau_values)
        ]
        for t in tau_values:
            m = summarize(questions, tau_labels[t], indices)["quality_cost"]
            tau_rows.append(
                [
                    group,
                    t,
                    len(indices),
                    percent(m["argmax_f1"]),
                    dec(m["argmax_context_tokens"], 1),
                    percent(m["argmax_evidence"]),
                    percent(m["expected_f1"]),
                    dec(m["expected_context_tokens"], 1),
                ]
            )
    parts.append(
        table(["组", "τ", "跨τ共同n", "argmax F1", "argmax Context", "argmax E", "期望F1", "期望Context"], tau_rows)
    )
    bridge, comparison = get_row(sweep_rows, p, "bridge"), get_row(sweep_rows, p, "comparison")
    parts.extend(
        [
            "Bridge/Comparison 参考 D 的覆盖分别为 {}/500、{}/500；平均候选、标签熵和质量成本均应分类型阅读。全零 F1 的{}/{}差异与 Gold 检索覆盖差异共同影响可训练题构成，不能将有效子集当成原始自然分布。".format(
                bridge["evaluated_n"],
                comparison["evaluated_n"],
                sum(q.type == "bridge" and max(q.f1) == 0 for q in questions),
                sum(q.type == "comparison" and max(q.f1) == 0 for q in questions),
            ),
            "## 9. 典型问题（目的性分类、固定 qid 顺序选例）",
            "每一类别在 Bridge/Comparison 内选择 qid 字典序第一题，不按结果收益排序；显示类别总体数量，类别可重叠，不能由案例频率外推。分类条件、预测、Gold 答案、全部曲线与概率保存在 examples.jsonl。不存在的类别如实列为空，不编造。",
        ]
    )
    category_names = {
        "k1_correct_evidence_incomplete": "k=1 正确但 Gold Evidence 不完整",
        "multiple_candidates": "多个 k 满足双重约束",
        "single_candidate": "只有一个 k 满足约束",
        "empty_candidates": "非零 F1 但无 k 满足约束",
        "all_zero_f1": "所有 k 的 F1 均为 0",
        "f1_decreases_with_k": "增加检索文档后 F1 下降",
        "unknown_evidence": "Evidence 匹配存在 unknown",
    }
    for example in examples:
        parts.append(
            "### {} / {}（本组共 {} 题）".format(
                category_names[example["category"]], example["type"], example["category_population_n"]
            )
        )
        if example["question_id"] is None:
            parts.append("该组该类别为空。")
            continue
        e = example
        parts.extend(
            [
                "qid=`{}`；问题：{}；Gold：{}。".format(e["question_id"], e["question_text"].strip(), e["gold_answer"]),
                "状态={}，原 Hard k={}，候选={}，Soft argmax={}；k1 预测={}。".format(
                    e["label_status"],
                    e["original_hard_oracle_k"],
                    list(e["valid_candidates"]),
                    e["argmax_soft_label_k"],
                    e["predictions"][0],
                ),
                table(
                    ["k", "F1", "E lower", "E upper", "unknown数", "Context Tokens", "候选", "概率"],
                    [
                        [
                            k,
                            dec(e["f1_curve"][j]),
                            dec(e["evidence_diagnostic"][j]["sentence_recall_lower"]),
                            dec(e["evidence_diagnostic"][j]["sentence_recall_upper"]),
                            e["evidence_diagnostic"][j]["sentence_unknown"],
                            e["token_curves"]["context_tokens"][j],
                            "是" if e["candidate_mask"][j] else "否",
                            dec(e["soft_label"][j], 4) if e["soft_label"] is not None else "null",
                        ]
                        for j, k in enumerate(e["k_values"])
                    ],
                ),
            ]
        )
        if e["category"] == "k1_correct_evidence_incomplete":
            parts.append(
                "观察：答案正确与 Gold 字面覆盖不同；小 k 被保留或剔除取决于阈值，不能断言模型已完成 Gold 推理链，也不能排除等价证据。"
            )
        elif e["category"] == "f1_decreases_with_k":
            parts.append(
                "观察：相邻下降点={}；增加 k 未保证答案更好，即使已有 Gold 证据保留也可能退化。无法仅凭这些曲线确认干扰或模型随机性的因果解释。".format(
                    e["adjacent_f1_drop_pairs"]
                )
            )
        elif e["category"] == "unknown_evidence":
            parts.append(
                "观察：上下界与 unknown 同时保存；Lower 门槛是保守筛选。未将 unknown 当作确定缺失，也未自动用 Upper 或其他 k 回退。"
            )
        elif e["category"] == "empty_candidates":
            parts.append("观察：F1 近优集合与证据门槛的交集为空，不能强制指定 k=15；当前无法产生正常训练标签。")
        elif e["category"] == "all_zero_f1":
            parts.append(
                "观察：即使原始约束下存在候选，也没有可辨识回答质量差异，按统一规则保存 null 标签；原 Hard 因并列取 1。"
            )
    parts.extend(
        [
            "## 10. 局限、下一步与复现",
            "1. 这是单次模型/提示词/检索器在既定 k 下的观察。F1 曲线可能包含解码波动与答案指标局限，Soft Label 平滑不能恢复未观察的因果预算需求。",
            "2. Gold 支持句的保守字面覆盖不是语义证据充分性的真值。τ=0.5 只覆盖一半，τ=1 也可能漏掉替代证明或因文本匹配不确定被排除；需要人工复核并报告分类型检索失败。",
            "3. 相同或接近 F1 的预算获得不同概率，部分来自人为 λ/T 成本偏好；标签保留更多已观测预算信息，却不保证包含更多可由 Query 预测的信息。正值平坦与全零失败要区分。",
            "4. 统一排除全零再做门槛筛选造成非随机缺失，Bridge 比重尤其可能下降。无标签不等于容易/不需要检索，不能把该子集质量当成所有问题的部署表现。",
            "5. ε、τ 改变样本覆盖与候选数，λ/T 改变分布形状；本次没有挑选最优配置、显著性检验或证明某一配置优于其他配置。参数扫描统计是描述性的，不把k1减少视为成功指标。",
            "6. 当前数据来自分层抽取的开发集；若未来训练，不应在同一批数据上选参又宣称测试泛化。应先明确训练目标（答案效用还是Gold覆盖）、部署时输入、无标签策略、绝对/相对成本定义，再由用户决定独立 Train 自然随机采样、训练/验证/测试划分与Query-only可预测性检查。本轮未执行这些后续实验。",
            "7. 后续可先人工审阅30～50个 k1正确但证据不足、门槛无解、unknown及Bridge失败样本；核实替代证据和匹配偏差。若需要因果验证、k=0或Gold上下文对照，必须另设实验，本轮不补跑。",
            "代码和命令见同目录 README.md。metadata.json 保存四个输入的绝对路径/SHA256、配置与代码SHA256、版本、运行命令、Python/绘图库版本、成本和证据定义；validation.json 同时检查真实落盘文件。输出目录创建时拒绝覆盖，默认分配新实验编号。",
            "完整输出：labels.jsonl（{}行）、per_k_details.csv（{}行）、feasibility_summary.csv（{}行）、parameter_sweep.csv（{}行）、distribution_summary.json、examples.jsonl、report.md、metadata.json、validation.json与5张PNG。无论是否有有效标签，所有问题都保留。".format(
                len(questions), len(questions) * 15, len(feasibility_rows), len(sweep_rows)
            ),
        ]
    )
    (output / "report.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def validate_outputs(output, questions, reference, feasibility_rows, sweep_rows):
    """Read saved rows back; check curves, probabilities, partitions and summary totals."""
    from build_soft_oracle import read_rows, close

    saved = read_rows(output / "labels.jsonl")
    require(
        len(saved) == len(questions) and {r["question_id"] for r in saved} == {q.question_id for q in questions},
        "Saved label qids differ",
    )
    valid = 0
    for q, r in zip(questions, saved):
        require(q.question_id == r["question_id"], "Saved ordering changed")
        require(r["parameters"] == reference.to_dict(), "Saved parameter mismatch")
        for field in (
            "f1_curve",
            "em_curve",
            "evidence_recall_curve",
            "token_cost_curve",
            "candidate_mask",
            "utility_curve",
            "normalized_cost_curve",
        ):
            require(len(r[field]) == 15, "Bad saved curve: " + field)
        require(
            r["valid_candidate_count"] == sum(r["candidate_mask"]) == len(r["valid_candidates"]),
            "Saved candidate mismatch",
        )
        if r["label_status"] == "valid":
            valid += 1
            probs = r["soft_label"]
            require(len(probs) == 15 and all(math.isfinite(p) and p >= 0 for p in probs), "Bad saved probabilities")
            close(math.fsum(probs), 1, "Saved probability sum")
            require(all(p == 0 for p, m in zip(probs, r["candidate_mask"]) if not m), "Noncandidate has probability")
            require(r["argmax_soft_label_k"] in r["valid_candidates"], "Saved argmax invalid")
            require(
                (
                    max(q.f1) - q.f1[r["argmax_soft_label_k"] - 1] <= reference.epsilon + 1e-12
                    if reference.scheme in ("B", "D")
                    else True
                ),
                "F1 bound violated",
            )
        else:
            require(
                r["soft_label"] is None and r["argmax_soft_label_k"] is None, "Invalid label disguised as distribution"
            )
    detail = read_rows(output / "per_k_details.csv")
    require(
        len(detail) == len(questions) * 15 and len({(r["question_id"], r["k"]) for r in detail}) == len(detail),
        "Saved per-k key count mismatch",
    )
    lookup = {r["question_id"]: r for r in saved}
    for r in detail:
        label = lookup[r["question_id"]]
        j = int(r["k"]) - 1
        close(float(r["f1"]), label["f1_curve"][j], "Saved per-k F1")
        close(float(r["context_tokens"]), label["token_curves"]["context_tokens"][j], "Saved per-k tokens")
        if label["label_status"] == "valid":
            close(float(r["soft_label_probability"]), label["soft_label"][j], "Saved per-k probability")
        else:
            require(r["soft_label_probability"] == "", "Invalid saved per-k probability must be empty")
    for r in feasibility_rows:
        require(
            r["valid_n"] + r["empty_nonzero_n"] + r["all_zero_f1_n"] == r["population_n"], "Sweep partition mismatch"
        )
        require(
            sum(r["candidate_count_hist_all"].values()) == r["population_n"], "Candidate histogram denominator mismatch"
        )
        require(sum(r["candidate_count_hist_valid"].values()) == r["valid_n"], "Valid candidate histogram mismatch")
    for r in sweep_rows:
        require(
            sum(r["argmax_k_hist"].values()) == sum(r["hard_oracle_k_hist"].values()) == r["evaluated_n"],
            "Summary histogram mismatch",
        )
        if r["evaluated_n"]:
            close(sum(r["mean_probability_mass_by_k"]), 1, "Mean mass")
        if r["scheme"] in ("B", "D") and r["evaluated_n"]:
            require(
                r["argmax_f1_loss"] <= r["epsilon"] + 1e-12 and r["expected_f1_loss"] <= r["epsilon"] + 1e-12,
                "Aggregate F1 bound violated",
            )
    saved_sweep = read_rows(output / "parameter_sweep.csv")
    saved_feasible = read_rows(output / "feasibility_summary.csv")
    require(
        len(saved_sweep) == len(sweep_rows) and len(saved_feasible) == len(feasibility_rows),
        "Saved summary count mismatch",
    )
    for saved_row, expected_row in zip(saved_sweep, sweep_rows):
        for field in ("group", "scope", "scheme"):
            require(saved_row[field] == expected_row[field], "Saved summary identity mismatch")
        for field in ("evaluated_n", "own_valid_n", "population_n"):
            require(int(saved_row[field]) == expected_row[field], "Saved summary denominator mismatch")
        require(
            json.loads(saved_row["argmax_k_hist"]) == expected_row["argmax_k_hist"], "Saved argmax histogram mismatch"
        )
        for field in ("argmax_f1", "expected_f1", "argmax_context_tokens", "expected_context_tokens", "mean_entropy"):
            if expected_row[field] is not None:
                close(float(saved_row[field]), expected_row[field], "Saved summary metric: " + field)
    for saved_row, expected_row in zip(saved_feasible, feasibility_rows):
        require(int(saved_row["valid_n"]) == expected_row["valid_n"], "Saved feasibility mismatch")
    temperature_groups = {}
    for r in sweep_rows:
        key = tuple(r[k] for k in ("scheme", "epsilon", "tau", "lambda", "group", "scope"))
        signature = (r["evaluated_n"], r["argmax_k_hist"], r["argmax_f1"], r["argmax_context_tokens"])
        require(temperature_groups.setdefault(key, signature) == signature, "Temperature changed argmax")
    refrow = get_row(sweep_rows, reference)
    require(refrow["evaluated_n"] == valid, "Reference summary vs label count mismatch")
    for k in range(1, 16):
        close(
            mean(r["soft_label"][k - 1] for r in saved if r["label_status"] == "valid"),
            refrow["mean_probability_mass_by_k"][k - 1],
            "Reference mass consistency",
        )
    return {
        "passed": True,
        "labels_n": len(saved),
        "valid_labels_n": valid,
        "per_k_n": len(detail),
        "feasibility_rows": len(feasibility_rows),
        "parameter_sweep_rows": len(sweep_rows),
        "probabilities_and_null_states_checked": True,
        "summary_partitions_and_histograms_checked": True,
        "reference_probability_mass_recomputed_from_saved_labels": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--output", help="New directory only; default allocates experiment_NNN")
    args = parser.parse_args()
    # Resolve plotting dependency before creating any output.
    import matplotlib  # noqa: F401

    config, reference, questions, validation, metadata = context(args.config)
    require(
        reference.scheme == "D",
        "Full report requires reference scheme D; use build_soft_oracle.py for other reference schemes",
    )
    require(
        all(
            config["reference"][field] in config["sweep"][field]
            for field in ("epsilon", "tau", "lambda", "temperature")
        ),
        "Reference parameters must be in sweep for full analysis",
    )
    labels = [build_label(q, reference) for q in questions]
    output = allocate_output(args.output)
    print("Output: " + str(output), flush=True)
    save_labels(output, questions, labels)
    feasible, sweep, detailed = scan(questions, config, reference)
    write_csv(output / "feasibility_summary.csv", feasible)
    write_csv(output / "parameter_sweep.csv", sweep)
    baselines = {}
    for group in GROUPS:
        indices = group_indices(questions, group)
        baselines[group] = {
            "population_n": len(indices),
            "fixed_k": [fixed_metrics(questions, indices, k) for k in range(1, 16)],
            "hard_oracle_hist": histogram([questions[i].original_hard_oracle_k for i in indices], range(1, 16)),
            "hard_oracle_f1": mean(max(questions[i].f1) for i in indices),
            "hard_oracle_context_tokens": mean(
                questions[i].tokens["context_tokens"][questions[i].original_hard_oracle_k - 1] for i in indices
            ),
            "hard_oracle_evidence": mean(
                questions[i].evidence[questions[i].original_hard_oracle_k - 1][reference.evidence_metric]
                for i in indices
            ),
        }
    write_json(
        output / "distribution_summary.json",
        {
            "analysis_kind": "Offline Oracle / Label Analysis",
            "baselines_all_population": baselines,
            "results": detailed,
        },
    )
    examples = select_examples(questions, labels, reference)
    with (output / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for e in examples:
            handle.write(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n")
    metadata["plotting"] = make_figures(output, questions, labels, feasible, sweep, reference)
    make_report(output, questions, labels, validation, feasible, sweep, reference, examples, baselines)
    validation["output_validation"] = validate_outputs(output, questions, reference, feasible, sweep)
    assert_inputs_unchanged(metadata)
    metadata["input_hashes_unchanged_after_run"] = True
    metadata["sweep_numeric_parameter_combinations"] = (
        len(config["sweep"]["epsilon"])
        * len(config["sweep"]["tau"])
        * len(config["sweep"]["lambda"])
        * len(config["sweep"]["temperature"])
    )
    metadata["output_sha256"] = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in ("metadata.json", "validation.json")
    }
    write_json(output / "validation.json", validation)
    write_json(output / "metadata.json", metadata)
    print(json.dumps(validation["output_validation"], ensure_ascii=False), flush=True)
    print("Completed: " + str(output), flush=True)


if __name__ == "__main__":
    main()
