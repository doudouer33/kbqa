"""Compare fixed-k and two-stage label Oracles on the frozen Train and Validation splits."""

import csv
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from prepare_data import K_KEYS, K_VALUES, LEVELS, TYPES, read_rows, require, validate_result
from split_data import ANALYSIS_SPLITS, get_manifest, load_settings, read_metadata

FULL = "Oracle-Full"
MAPPED = "Oracle-Mapped"
CANDIDATE = "Oracle-Candidate"
POLICIES = (FULL, MAPPED, CANDIDATE)
MEANS = ("mean_f1", "mean_em", "mean_context_tokens", "mean_selected_k", "mean_utility")
DISPLAY_SPLITS = {"train": "Train", "val": "Validation"}
COLORS = {FULL: "#286493", MAPPED: "#b34b36", CANDIDATE: "#147c69"}
LABEL_COLORS = ("#38678c", "#52a7b1", "#7e9d55", "#dfab4b", "#ad5572")


@dataclass(frozen=True)
class Cohort:
    f1: np.ndarray
    em: np.ndarray
    tokens: np.ndarray
    levels: np.ndarray
    types: np.ndarray


def selection_config(config):
    full, low, fallback = config["full_k"], config["stage2_k"], config["fallback_k"]
    require(full == list(K_VALUES) and len(low) == 4 and low == list(range(len(low)))
            and type(fallback) is int and fallback in full and fallback > max(low), "Invalid k ranges")
    require(config["objectives"] == ["max_f1", "utility"] and config["tie_break"] == ["context_tokens", "k"],
            "Invalid objective or tie-break configuration")
    for field in ("tie_atol", "token_scale"):
        value = config[field]
        require(type(value) in (int, float) and math.isfinite(value), f"Invalid {field}")
    require(0 <= config["tie_atol"] < 1e-6 and config["token_scale"] > 0, "Invalid tolerance or token scale")
    lambdas = config["lambdas"]
    require(lambdas and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in lambdas)
            and lambdas == sorted(set(lambdas)), "Lambda values must be unique, sorted, nonnegative numbers")
    return full, low, low + [fallback]


def load_cohorts(metrics, source, manifest):
    membership = {qid: split for split in ANALYSIS_SPLITS for qid in manifest["splits"][split]["qids"]}
    rows = {split: [] for split in ANALYSIS_SPLITS}
    for row in read_rows(metrics, source["sha256"]):
        qid = row["qid"]
        if qid not in membership:
            continue
        results = row["results"]
        require(set(results) == K_KEYS, f"Incomplete or unexpected k values for {qid}")
        for k in K_VALUES:
            validate_result(results[str(k)], k, f"{qid} k={k}")
        rows[membership[qid]].append(row)
    cohorts = {}
    for split, subset in rows.items():
        require([row["qid"] for row in subset] == manifest["splits"][split]["qids"], f"Frozen QID order differs for {split}")
        arrays = [np.array([[row["results"][str(k)][metric] for k in K_VALUES] for row in subset])
                  for metric in ("f1", "em", "context_tokens")]
        cohorts[split] = Cohort(*arrays, np.array([row["level"] for row in subset]), np.array([row["type"] for row in subset]))
    return cohorts


def select_k(scores, tokens, ks, atol):
    # Compare every candidate with the actual maximum before applying tie-breaks.
    scores, tokens = scores[:, ks], tokens[:, ks]
    tied = scores.max(axis=1, keepdims=True) - scores <= atol
    cheapest = np.where(tied, tokens, np.iinfo(tokens.dtype).max).min(axis=1, keepdims=True)
    return np.asarray(ks)[np.argmax(tied & (tokens == cheapest), axis=1)]


def selected_metrics(cohort, chosen, utility):
    rows = np.arange(chosen.size)
    return {"mean_f1": cohort.f1[rows, chosen], "mean_em": cohort.em[rows, chosen],
            "mean_context_tokens": cohort.tokens[rows, chosen], "mean_selected_k": chosen,
            "mean_utility": None if utility is None else utility[rows, chosen]}


def means(selected, mask):
    return {name: float(values[mask].mean()) if values is not None and mask.any() else None
            for name, values in selected.items()}


def fraction(count, total):
    return count / total if total else None


def counts(values, ks):
    distribution = np.bincount(values, minlength=len(K_VALUES))
    return {k: int(distribution[k]) for k in ks}


def label_distributions(spec, cohort, all_zero, chosen, full, low, final):
    population = np.ones(all_zero.size, dtype=bool)
    groups = [("overall", "all", population)]
    groups.extend(("level", level, cohort.levels == level) for level in LEVELS)
    groups.extend(("type", kind, cohort.types == kind) for kind in TYPES)

    for scheme, policy in (("A", MAPPED), ("B", CANDIDATE)):
        labels = chosen[policy]
        high = labels == final[-1]
        for field, group, group_mask in groups:
            for population_name, subset in (("all", population), ("all_zero_f1", all_zero), ("any_positive_f1", ~all_zero)):
                mask = group_mask & subset
                size = int(mask.sum())
                prefix = {**spec, "scheme": scheme, "group_by": field, "group": group, "population": population_name,
                          "group_count": int(group_mask.sum()), "population_count": size,
                          "population_proportion": fraction(size, int(group_mask.sum()))}
                stages = [("final_k", labels[mask], final), ("stage1", high[mask].astype(int), (0, 1)),
                          ("stage2", labels[mask & ~high], low)]
                if scheme == "A":
                    stages.insert(0, ("original_full_k", chosen[FULL][mask], full))
                for stage, values, ks in stages:
                    for label, count in counts(values, ks).items():
                        yield {**prefix, "stage": stage, "label": label, "count": count, "denominator": values.size,
                               "proportion": fraction(count, values.size)}


def mapping_rows(spec, selected, chosen, all_zero, high, fallback, atol):
    changed = high & (chosen[FULL] != fallback)
    disagree = chosen[MAPPED] != chosen[CANDIDATE]
    delta_f1 = selected[MAPPED]["mean_f1"] - selected[FULL]["mean_f1"]
    down, up = delta_f1 < -atol, delta_f1 > atol
    for subset, mask in (("all", np.ones(high.size, dtype=bool)), ("full_high_k", high), ("changed_k", changed)):
        size = int(mask.sum())
        row = {**spec, "subset": subset, "split_count": high.size, "sample_count": size,
               "sample_proportion_of_split": fraction(size, high.size)}
        for name, event in (("original_high_k", high), ("changed_k", changed), ("all_zero_f1", all_zero),
                            ("f1_down", down), ("f1_same", ~(down | up)), ("f1_up", up), ("label_disagreement", disagree)):
            count = int((event & mask).sum())
            row[f"{name}_count"] = count
            row[f"{name}_proportion"] = fraction(count, size)
        summaries = {name: means(selected[policy], mask) for name, policy in
                     (("full", FULL), ("mapped", MAPPED), ("candidate", CANDIDATE))}
        for name, summary in summaries.items():
            row.update({f"{name}_{metric}": value for metric, value in summary.items()})
        for name in ("full", "candidate"):
            for metric in MEANS:
                a, b = summaries["mapped"][metric], summaries[name][metric]
                row[f"mapped_minus_{name}_{metric}"] = None if a is None else a - b
        yield row


def analyze(cohorts, config):
    full, low, final = selection_config(config)
    fallback = final[-1]
    atol, scale = config["tie_atol"], config["token_scale"]
    objectives = [("max_f1", None)] + [("utility", weight) for weight in config["lambdas"]]
    oracle, labels, mapping = [], [], []
    for split, cohort in cohorts.items():
        size = cohort.f1.shape[0]
        all_zero = cohort.f1.max(axis=1) == 0
        population = np.ones(size, dtype=bool)
        for objective, weight in objectives:
            spec = {"split": split, "objective": objective, "lambda": weight}
            utility = None if weight is None else cohort.f1 - weight * (cohort.tokens / scale)
            scores = cohort.f1 if utility is None else utility
            best = select_k(scores, cohort.tokens, full, atol)
            high = best > low[-1]
            chosen = {FULL: best, MAPPED: np.where(high, fallback, best),
                      CANDIDATE: select_k(scores, cohort.tokens, final, atol)}
            for k in final:
                chosen[f"Fixed k={k}"] = np.full(size, k)
            selected = {name: selected_metrics(cohort, choice, utility) for name, choice in chosen.items()}
            rows = np.arange(size)
            require(np.all(scores[rows, best] + atol >= scores[rows, chosen[CANDIDATE]]), "Full Oracle objective below Candidate")
            require(np.all(scores[rows, chosen[CANDIDATE]] + atol >= scores[rows, chosen[MAPPED]]),
                    "Candidate Oracle objective below Mapped")
            require(all(np.all(chosen[policy][all_zero] == 0) for policy in POLICIES), "All-zero F1 questions should select k=0")
            for name, choice in chosen.items():
                oracle.append({**spec, "policy": name, "sample_count": size, **means(selected[name], population),
                               "selected_k_distribution": json.dumps(counts(choice, full), separators=(",", ":"))})
            labels.extend(label_distributions(spec, cohort, all_zero, chosen, full, low, final))
            mapping.extend(mapping_rows(spec, selected, chosen, all_zero, high, fallback, atol))
    return oracle, labels, mapping


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(directory, oracle, labels, mapping, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    _, low, final = selection_config(config)
    lambdas = config["lambdas"]
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})

    def panels(title, rows=1, height=4.5):
        fig, axes = plt.subplots(rows, 2, figsize=(12, height), layout="constrained", squeeze=False, sharey="row")
        fig.suptitle(title, weight="bold")
        return fig, axes

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=150)
        plt.close(fig)

    def for_split(rows, split, **conditions):
        return [row for row in rows if row["split"] == split and all(row[key] == value for key, value in conditions.items())]

    def format_lambda(ax, x, tick_labels=None):
        if tick_labels is None:
            ax.set_xticks(x, [f"{v:g}" for v in x], rotation=90, ha="center", fontsize=8)
        else:
            ax.set_xticks(x, tick_labels, rotation=35, ha="right")
        unit = f"{config['token_scale']:,.0f}"
        ax.set_xlabel(f"λ (F1 penalty per {unit} tokens)" if tick_labels is None else "Objective and λ")
        ax.grid(axis="y", alpha=0.23)
        ax.set_axisbelow(True)

    overall = [row for row in labels if row["group_by"] == "overall" and row["population"] == "all" and row["stage"] == "final_k"]
    fig, axes = panels("Final k labels under maximum F1")
    for split, ax in zip(ANALYSIS_SPLITS, axes[0]):
        x = np.arange(len(final))
        for scheme, offset, color, name in (("A", -0.19, COLORS[MAPPED], "A · mapped"),
                                             ("B", 0.19, COLORS[CANDIDATE], "B · candidate")):
            rows = for_split(overall, split, objective="max_f1", scheme=scheme)
            ax.bar(x + offset, [row["proportion"] for row in rows], 0.36, color=color, label=name)
        ax.set(title=DISPLAY_SPLITS[split], xlabel="Final k", xticks=x, xticklabels=final)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.23)
    axes[0, 0].set_ylabel("Share of questions")
    axes[0, 1].legend(frameon=False)
    save(fig, "max_f1_label_distribution")

    for metric, title, filename in (("mean_f1", "Mean Answer F1", "f1_vs_lambda"),
                                     ("mean_context_tokens", "Mean Context Tokens", "context_tokens_vs_lambda")):
        fig, axes = panels(f"{title} across utility penalties")
        for split, ax in zip(ANALYSIS_SPLITS, axes[0]):
            for policy in POLICIES:
                rows = for_split(oracle, split, objective="utility", policy=policy)
                ax.plot(lambdas, [row[metric] for row in rows], "o-", ms=4, color=COLORS[policy], label=policy)
            ax.set_title(DISPLAY_SPLITS[split])
            format_lambda(ax, lambdas)
        axes[0, 0].set_ylabel(title)
        axes[0, 1].legend(frameon=False)
        save(fig, filename)

    fig, axes = panels("Answer quality and retrieval cost", height=5.1)
    for split, ax in zip(ANALYSIS_SPLITS, axes[0]):
        fixed = [row for row in for_split(oracle, split, objective="max_f1") if row["policy"].startswith("Fixed")]
        for policy in POLICIES:
            rows = for_split(oracle, split, objective="utility", policy=policy)
            ax.plot([row["mean_context_tokens"] for row in rows], [row["mean_f1"] for row in rows], "o-", ms=4,
                    color=COLORS[policy], label=policy)
        for row in fixed:
            x, y = row["mean_context_tokens"], row["mean_f1"]
            ax.scatter(x, y, color="#333333", marker="D", s=25)
            ax.annotate(f"k={row['mean_selected_k']:g}", (x, y), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=DISPLAY_SPLITS[split], xlabel="Mean Context Tokens")
        ax.grid(alpha=0.23)
        ax.margins(x=0.07, y=0.10)
    handles, names = axes[0, 1].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="D", linestyle="None", color="#333333"))
    axes[0, 1].legend(handles, names + ["Fixed k"], frameon=False)
    axes[0, 0].set_ylabel("Mean Answer F1")
    save(fig, "f1_vs_context_tokens")

    fig, axes = panels("Final k labels across utility penalties", rows=2, height=7.3)
    for (i, scheme), (j, split) in product(enumerate(("A", "B")), enumerate(ANALYSIS_SPLITS)):
        ax = axes[i, j]
        x = np.arange(len(lambdas))
        bottom = np.zeros(len(lambdas))
        for k, color in zip(final, LABEL_COLORS):
            rows = for_split(overall, split, objective="utility", scheme=scheme, label=k)
            heights = [row["proportion"] for row in rows]
            ax.bar(x, heights, bottom=bottom, color=color, label=f"k={k}")
            bottom += heights
        ax.set_title(f"{DISPLAY_SPLITS[split]} · Scheme {scheme}")
        ax.set_xticks(x, [f"{weight:g}" for weight in lambdas])
        ax.set_xlabel("λ")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        if j == 0:
            ax.set_ylabel("Share of questions")
    axes[0, 1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False, fontsize=8)
    save(fig, "labels_vs_lambda")

    fig, axes = panels(f"How mapping to k={final[-1]} changes per-question F1", rows=2, height=7.6)
    for (i, subset), (j, split) in product(enumerate(("all", "full_high_k")), enumerate(ANALYSIS_SPLITS)):
        ax = axes[i, j]
        rows = for_split(mapping, split, subset=subset)
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for field, name, color in (("f1_down_proportion", "Down", "#b34b36"), ("f1_same_proportion", "Same", "#9eabb5"),
                                    ("f1_up_proportion", "Up", "#147c69")):
            heights = [row[field] or 0 for row in rows]
            ax.bar(x, heights, bottom=bottom, color=color, label=name)
            bottom += heights
        for pos, row in enumerate(rows):
            if row["sample_count"] == 0:
                ax.text(pos, 0.04, "n=0", rotation=90, ha="center", fontsize=7)
        group = "All questions" if i == 0 else f"Full Oracle selected k > {low[-1]}"
        ax.set_title(f"{DISPLAY_SPLITS[split]} · {group}")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        format_lambda(ax, x, ["Max F1"] + [f"{weight:g}" for weight in lambdas])
        if j == 0:
            ax.set_ylabel("Share of questions")
    axes[0, 1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3)
    save(fig, "mapping_f1_change")


def main():
    metrics, report, config, split_dir, analysis_dir = load_settings(__doc__)
    selection = config["selection"]
    selection_config(selection)
    metadata, source = read_metadata(metrics, report)
    manifest = get_manifest(metadata, source, config["split"], split_dir)
    cohorts = load_cohorts(metrics, source, manifest)
    oracle, labels, mapping = analyze(cohorts, selection)
    figures = analysis_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=analysis_dir, suffix=".tmp") as temp:
        destination = Path(temp)
        write_csv(destination / "oracle_metrics.csv", oracle)
        write_csv(destination / "label_distribution.csv", labels)
        write_csv(destination / "mapping_analysis.csv", mapping)
        temp_figures = destination / "figures"
        temp_figures.mkdir()
        plot_results(temp_figures, oracle, labels, mapping, selection)
        for path in sorted(destination.glob("*.csv")):
            path.replace(analysis_dir / path.name)
        for path in sorted(temp_figures.glob("*.png")):
            path.replace(figures / path.name)
    print(f"Wrote {len(oracle)} Oracle/baseline, {len(labels)} label distribution and {len(mapping)} mapping rows")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, TypeError, ValueError) as error:
        sys.exit(f"Label analysis failed: {error}")
