"""Summarize fixed-k results on all 5,000 HotpotQA training questions."""

import csv
import json
import sys
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory

from prepare_data import K_KEYS, K_VALUES, LEVELS, QUESTION_COUNT, TYPES, load_settings, read_rows, require, validate_result

K = "k"
COUNT = "Sample Count"
F1 = "Mean F1"
EM = "Mean EM"
TOKENS = "Mean Context Tokens"
MEDIAN = "Median Context Tokens"
EMPTY = "Empty Answer Count"
CSV_FIELDS = (K, COUNT, F1, EM, TOKENS, MEDIAN, EMPTY)


def scores(results):
    return {COUNT: len(results), F1: mean(row["f1"] for row in results), EM: mean(row["em"] for row in results)}


def collect_metrics(path):
    by_k = {k: [] for k in K_VALUES}
    strata = {"level": {level: [] for level in LEVELS}, "type": {kind: [] for kind in TYPES}}
    seen = set()
    for row in read_rows(path):
        qid = row["qid"]
        require(isinstance(qid, str) and qid and qid not in seen, f"Invalid or duplicate question ID {qid!r}")
        seen.add(qid)
        results = row["results"]
        require(isinstance(results, dict) and set(results) == K_KEYS, f"Incomplete or unexpected k values for {qid}")
        for k in K_VALUES:
            result = results[str(k)]
            validate_result(result, k, f"{qid} k={k}")
            by_k[k].append(result)
        for field, groups in strata.items():
            require(row[field] in groups, f"Invalid {field} for {qid}: {row[field]!r}")
            groups[row[field]].append(results["0"])
    require(len(seen) == QUESTION_COUNT, f"Expected {QUESTION_COUNT} questions, found {len(seen)} in {path}")
    require(all(group for groups in strata.values() for group in groups.values()), "A level or type group is empty")

    fixed = []
    for k, results in by_k.items():
        tokens = [row["context_tokens"] for row in results]
        fixed.append({K: k, **scores(results), TOKENS: mean(tokens), MEDIAN: median(tokens),
                      EMPTY: sum(not row["prediction"].strip() for row in results)})
    zero_f1 = [row["f1"] for row in by_k[0]]
    zero, perfect = zero_f1.count(0), zero_f1.count(1)
    bands = {"f1_eq_0": zero, "f1_between_0_and_1": len(zero_f1) - zero - perfect, "f1_eq_1": perfect}
    k0 = {"overall": fixed[0], "f1_distribution": {name: {"count": count, "proportion": count / len(zero_f1)}
                                                 for name, count in bands.items()},
          **{f"by_{field}": {name: scores(results) for name, results in groups.items()} for field, groups in strata.items()}}
    return fixed, k0


def plot_metrics(fixed, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    ks = [row[K] for row in fixed]
    f1 = [row[F1] for row in fixed]
    tokens = [row[TOKENS] for row in fixed]
    color = "#185b84"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
    for column, label, filename in (
        (F1, "Mean Answer F1", "mean_f1_vs_k.png"),
        (EM, "Mean Exact Match", "mean_em_vs_k.png"),
        (TOKENS, "Mean Context Tokens", "mean_context_tokens_vs_k.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.plot(ks, [row[column] for row in fixed], marker="o", color=color, linewidth=2, markersize=4)
        ax.set(title=f"{label} by fixed k", xlabel="Fixed k", ylabel=label, xticks=ks)
        ax.grid(axis="y", alpha=0.25)
        if column == TOKENS:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        fig.savefig(directory / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(tokens, f1, color=color, linewidth=1.25, alpha=0.45)
    ax.scatter(tokens, f1, color=color, s=28, zorder=3)
    for k, x, y in zip(ks, tokens, f1):
        offset = 10 if k % 2 == 0 else -12
        ax.annotate(f"k={k}", (x, y), xytext=(0, offset), textcoords="offset points",
                    ha="center", va="bottom" if offset > 0 else "top", fontsize=8)
    ax.set(title="Answer F1 and context cost by fixed k", xlabel="Mean Context Tokens", ylabel="Mean Answer F1")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.margins(x=0.05, y=0.11)
    ax.grid(alpha=0.25)
    fig.savefig(directory / "mean_f1_vs_context_tokens.png", dpi=180)
    plt.close(fig)


def analyze(outputs):
    fixed, k0 = collect_metrics(outputs["per_question_metrics"])
    csv_path, figures = outputs["fixed_k_metrics"], outputs["figures"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=csv_path.parent, suffix=".tmp") as temp_csv, \
            TemporaryDirectory(dir=figures, suffix=".tmp") as temp_figures:
        csv_temp = Path(temp_csv) / csv_path.name
        with csv_temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(fixed)
        plot_metrics(fixed, Path(temp_figures))
        csv_temp.replace(csv_path)
        for image in sorted(Path(temp_figures).iterdir()):
            image.replace(figures / image.name)
    print(json.dumps({"k0": k0}, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        _, outputs, _ = load_settings(__doc__)
        analyze(outputs)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        sys.exit(f"Data analysis failed: {error}")
