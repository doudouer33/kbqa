"""Plot saved same-test-set Answer F1 and context cost without running RAG."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIXED_COLOR = "#176187"
BERT_COLOR = "#b4492b"
BERT_LABEL = "BERT Dynamic Top-K (argmax)"


def plot(rows, bert, x_key, xlabel, output):
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    x = [row[x_key] for row in rows]
    y = [100 * row["answer_f1"] for row in rows]
    ax.plot(x, y, "o-", color=FIXED_COLOR, markersize=4.5, linewidth=1.5,
            label="Fixed k", zorder=3)
    for i, (row, x_value, f1) in enumerate(zip(rows, x, y)):
        label = f"{f1:.2f}" if x_key == "k" else f"k={row['k']}"
        # Put labels below steep upward segments to keep them clear of the line.
        below = max(y[max(0, i - 1):i + 2]) - f1 > 1
        ax.annotate(label, (x_value, f1), xytext=(0, -7 if below else 7),
                    textcoords="offset points", ha="center",
                    va="top" if below else "bottom", fontsize=8, color=FIXED_COLOR)

    bert_f1 = 100 * bert["answer_f1"]
    if x_key == "k":
        ax.set_xticks(x)
        ax.axhline(bert_f1, color=BERT_COLOR, linestyle="--", linewidth=1.25,
                   label=f"{BERT_LABEL}, {bert_f1:.2f}%")
    else:
        ax.scatter(bert[x_key], bert_f1, marker="D", color=BERT_COLOR, s=40,
                   label=BERT_LABEL, zorder=4)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_title(f"HotpotQA Train5000 · Held-out test set (n={bert['test_count']:,})")
    ax.set(xlabel=xlabel, ylabel="Answer F1 (%)")
    ax.margins(x=0.045, y=0.13)
    ax.grid(axis="y", color="#e2e7eb", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    evaluation_path = REPORTS / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    methods = evaluation["all_test"]
    fixed = sorted(
        (int(name.split("_")[-1]), stats)
        for name, stats in methods.items()
        if name.startswith("fixed_k_")
    )
    bert = methods["bert"]
    if not fixed or any(stats["count"] != bert["count"] for _, stats in fixed):
        raise ValueError("Expected fixed-k and BERT metrics for the same test population")

    rows = [
        {
            "method": "fixed_k" if k is not None else "bert_dynamic_argmax",
            "k": k,
            "answer_f1": stats["argmax_answer_f1"],
            "average_context_tokens": stats["argmax_context_tokens"],
            "test_count": stats["count"],
            "test_ids_sha256": evaluation["test_ids_sha256"],
        }
        for k, stats in [*fixed, (None, bert)]
    ]
    output = REPORTS / "fixed_k_curves"
    output.mkdir(exist_ok=True)
    with (output / "fixed_k_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot(rows[:-1], rows[-1], "k", "Fixed retrieval k", output / "f1_vs_k")
    plot(rows[:-1], rows[-1], "average_context_tokens", "Average Context Tokens",
         output / "f1_vs_context_tokens")


if __name__ == "__main__":
    main()
