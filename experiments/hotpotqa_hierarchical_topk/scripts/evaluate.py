"""Evaluate frozen classifications and their offline HotpotQA F1/cost outcomes."""

import json

import numpy as np

from analyze_labels import CANDIDATE, FULL, MAPPED, select_k
from prepare_data import K_VALUES, provenance, read_json, read_rows, require
from training_common import (CLASSES, EXPERIMENTS, HEAD_CLASSES, classifier_metrics, e2e, entrypoint, head_path,
                             label_counts, load_data, read_labels, route, write_csv)
from analysis.hotpot_formal_runner.storage import atomic_json  # noqa: E402
from experiments.hotpotqa_hierarchical_topk.scripts.predict import RESULT_DIRS, options, protocol  # noqa: E402

METRICS = ("mean_f1", "mean_em", "mean_context_tokens", "mean_selected_k", "score")
PRIOR = "Random-Prior"


def prediction_rows(directory, split, experiment, labels, protocol_source):
    slug = EXPERIMENTS[experiment][0]
    path = directory / "predictions" / f"{slug}.jsonl"
    marker = read_json(path.with_suffix(".json"))
    require(marker == {"protocol": protocol_source, "split": split, "experiment": experiment,
                       "predictions": provenance(path), "count": len(labels)}, f"Prediction binding differs in {path}")
    rows = list(read_rows(path, marker["predictions"]["sha256"]))
    require([row["qid"] for row in rows] == [row["qid"] for row in labels], f"Prediction QIDs differ in {path}")
    return rows


def learned_metrics(rows, labels, heads):
    selected = np.asarray([row["selected_k"] for row in rows])
    truth = np.asarray([row["flat5_label"] for row in labels])
    if len(heads) == 1:
        predicted = np.asarray([row["flat5_pred"] for row in rows])
        require(np.isin(predicted, np.arange(len(CLASSES))).all() and np.array_equal(selected, np.asarray(CLASSES)[predicted]),
                "Flat5 class and executed k disagree")
        return selected, {"flat5": classifier_metrics(truth, predicted, CLASSES)}

    stage1_truth = np.asarray([row["stage1_label"] for row in labels])
    low = stage1_truth == 0
    stage1 = np.asarray([row["stage1_pred"] for row in rows])
    stage2 = np.asarray([row["stage2_routed_pred"] if row["stage2_routed_pred"] is not None else -1 for row in rows])
    standalone = np.asarray([row["stage2_standalone_pred"] if row["stage2_standalone_pred"] is not None else -1 for row in rows])
    require(np.array_equal(standalone >= 0, low) and np.array_equal(stage2 >= 0, stage1 == 0)
            and np.array_equal(selected, route(stage1, stage2)), "Stage 2 standalone subset or predicted route differs")
    require(np.isin(selected, CLASSES).all(), "Invalid hierarchical k")
    return selected, {"stage1": classifier_metrics(stage1_truth, stage1, HEAD_CLASSES["stage1"]),
                      "stage2": classifier_metrics(truth[low], standalone[low], HEAD_CLASSES["stage2"]),
                      "joint": classifier_metrics(truth, np.searchsorted(CLASSES, selected), CLASSES)}


def random_prior(cohort, train_rows, paths, seeds):
    counts = label_counts(train_rows, "flat5")
    probabilities = counts / counts.sum()
    runs = []
    for seed in seeds:
        draw = np.random.default_rng(seed).choice(CLASSES, size=len(cohort.f1), replace=True, p=probabilities)
        runs.append((seed, e2e(cohort, draw, paths)))
    mean = {metric: float(np.mean([row[metric] for _, row in runs])) for metric in METRICS}
    std = {metric: float(np.std([row[metric] for _, row in runs], ddof=1)) for metric in METRICS}
    mean["selected_k_distribution"] = {k: float(np.mean([row["selected_k_distribution"][k] for _, row in runs])) for k in K_VALUES}
    std["selected_k_distribution"] = {k: float(np.std([row["selected_k_distribution"][k] for _, row in runs], ddof=1)) for k in K_VALUES}
    return mean, std, runs


def report_rows(results, std, count, split, repeats):
    for name, result in results.items():
        row = {"split": split, "method": name, "sample_count": count, "repeats": repeats if name == PRIOR else 1}
        row.update({metric: result[metric] for metric in METRICS})
        row.update({f"{metric}_std": std[metric] if name == PRIOR else None for metric in METRICS})
        row["selected_k_distribution"] = json.dumps(result["selected_k_distribution"], separators=(",", ":"))
        row["selected_k_distribution_std"] = json.dumps(std["selected_k_distribution"], separators=(",", ":")) if name == PRIOR else None
        for k in (1, 2, 3, 10):
            fixed = results[f"Fixed k={k}"]
            row[f"delta_f1_vs_fixed_k{k}"] = result["mean_f1"] - fixed["mean_f1"]
            row[f"relative_context_tokens_vs_fixed_k{k}"] = result["mean_context_tokens"] / fixed["mean_context_tokens"] - 1
        yield row


def plot_results(paths, directory, split, results, std, classification):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    title = "Validation" if split == "val" else "Test"
    methods = list(results)
    display = [name.replace("Hierarchical", "Hier.") for name in methods]
    colors = ["#7d8b96" if name.startswith("Fixed") else "#a27b3f" if name == PRIOR else "#147c69" if name in EXPERIMENTS
              else "#375e93" for name in methods]
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    def save(fig, filename, dest=figures):
        fig.savefig(dest / f"{filename}.png", dpi=150)
        plt.close(fig)

    for metric, label, filename in (("mean_f1", "Mean Answer F1", "answer_f1"),
                                     ("mean_context_tokens", "Mean Context Tokens", "context_tokens")):
        fig, ax = plt.subplots(figsize=(8.6, 6.5), layout="constrained")
        y = np.arange(len(methods))
        ax.barh(y, [results[m][metric] for m in methods], color=colors, xerr=[std[metric] if m == PRIOR else 0 for m in methods],
                error_kw={"capsize": 3, "elinewidth": 1})
        ax.set(yticks=y, yticklabels=display, title=f"{title} · {label}", xlabel=label)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.22)
        ax.set_axisbelow(True)
        save(fig, filename)

    fig, ax = plt.subplots(figsize=(10, 6.4), layout="constrained")
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for k in K_VALUES:
        values = np.asarray([results[m]["selected_k_distribution"][k] / classification["sample_count"] for m in methods])
        if values.any():
            color = {0: "#38678c", 1: "#52a7b1", 2: "#7e9d55", 3: "#dfab4b", 10: "#ad5572"}.get(k, "#b1b7bc")
            ax.barh(y, values, left=left, color=color, label=f"k={k}" if k in CLASSES else "Other k" if k == 4 else "_nolegend_")
            left += values
    ax.set(yticks=y, yticklabels=display, title=f"{title} · Executed k distribution", xlabel="Share of questions")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.10), ncol=6)
    save(fig, "selected_k_distribution")

    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    markers = ("D", "D", "D", "D", "D", "X", "o", "s", "P", "^", "o", "s", "^" )
    for method, name, color, marker in zip(methods, display, colors, markers):
        r = results[method]
        ax.errorbar(r["mean_context_tokens"], r["mean_f1"], xerr=std["mean_context_tokens"] if method == PRIOR else None,
                    yerr=std["mean_f1"] if method == PRIOR else None, color=color, marker=marker, markersize=6,
                    linestyle="None", capsize=3, label=name)
    ax.set(title=f"{title} · Answer quality and retrieval cost", xlabel="Mean Context Tokens", ylabel="Mean Answer F1")
    ax.grid(alpha=.22)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, .5), fontsize=8)
    save(fig, "f1_vs_context_tokens")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    panels = [(experiment, head, stats) for experiment, heads in classification["models"].items() for head, stats in heads.items()]
    for ax, (experiment, head, stats) in zip(axes.flat, panels):
        cm = np.asarray(stats["confusion_matrix"])
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if normalized[i, j] > .55 else "#142536")
        names = ["Low", "High"] if head == "stage1" else stats["classes"]
        ax.set(title=f"{experiment.replace('Hierarchical', 'Hier.')} · {head}", xlabel="Predicted", ylabel="True",
               xticks=np.arange(len(names)), yticks=np.arange(len(names)), xticklabels=names, yticklabels=names)
    fig.suptitle(f"{title} · Confusion matrices (counts, shading normalized by true class)")
    save(fig, "confusion_matrices")

    if split == "val":
        dest = paths.training / "figures"
        dest.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        heads = [(experiment, head, weighted) for experiment, (_, group, weighted) in EXPERIMENTS.items() for head in group]
        for ax, (experiment, head, weighted) in zip(axes.flat, heads):
            rows = read_json(head_path(paths, experiment, head) / "history.json")["history"]
            x = [row["epoch"] for row in rows]
            ax.plot(x, [row["train_loss"] for row in rows], "o-", ms=3, color="#375e93", label="Train")
            ax.plot(x, [row["val_loss"] for row in rows], "o-", ms=3, color="#b34b36", label="Validation")
            ax.set(title=f"{experiment.replace('Hierarchical', 'Hier.')} · {head}", xlabel="Epoch", ylabel="Weighted CE" if weighted else "CE",
                   xticks=x)
            ax.grid(alpha=.22)
        axes[0, 0].legend(frameon=False)
        save(fig, "training_validation_loss", dest)


def evaluate(paths, args):
    split = args.split
    frozen, protocol_source = protocol(paths)
    directory = paths.evaluation / RESULT_DIRS[split]
    if split == "test":
        val = paths.evaluation / RESULT_DIRS["val"]
        require(read_json(val / "complete.json") == {"protocol": protocol_source, "results": provenance(val / "results.csv"),
                                                    "split": "val", "count": 500}, "Frozen Validation comparison required first")
    datasets, _ = load_data(paths, ("train", split))
    all_labels = read_labels(paths, datasets)
    cohort = datasets[split][1]
    labels = all_labels[split]
    size = len(labels)
    results = {f"Fixed k={k}": e2e(cohort, np.full(size, k), paths) for k in CLASSES}
    seeds = frozen["random_prior_seeds"]
    results[PRIOR], std, runs = random_prior(cohort, all_labels["train"], paths, seeds)
    classification = {"split": split, "sample_count": size, "true_final_k_counts": label_counts(labels, "flat5").tolist(),
                      "all_zero_f1_count": sum(row["all_zero_f1"] for row in labels), "models": {}}
    for experiment, (_, heads, _) in EXPERIMENTS.items():
        rows = prediction_rows(directory, split, experiment, labels, protocol_source)
        selected, stats = learned_metrics(rows, labels, heads)
        result = e2e(cohort, selected, paths)
        if split == "val":
            canonical = {**result, "selected_k_distribution": {str(k): v for k, v in result["selected_k_distribution"].items()}}
            require(canonical == frozen["choices"][experiment]["validation"], f"Validation differs from selection for {experiment}")
        results[experiment] = result
        classification["models"][experiment] = stats
    tol = paths.selection["tie_atol"]
    full = select_k(cohort.f1, cohort.tokens, K_VALUES, tol)
    candidate = select_k(cohort.f1, cohort.tokens, CLASSES, tol)
    require(candidate.tolist() == [row["optimal_k"] for row in labels], "Oracle-Candidate differs from labels")
    for method, choices in ((FULL, full), (MAPPED, np.where(full > max(CLASSES[:4]), CLASSES[-1], full)), (CANDIDATE, candidate)):
        results[method] = e2e(cohort, choices, paths)
    plot_results(paths, directory, split, results, std, classification)
    results_file = directory / "results.csv"
    write_csv(results_file, list(report_rows(results, std, size, split, len(seeds))))
    write_csv(directory / "random_prior_runs.csv", [{"seed": seed, **{m: row[m] for m in METRICS}} for seed, row in runs])
    atomic_json(directory / "classification_metrics.json", classification)
    atomic_json(directory / "complete.json", {"protocol": protocol_source, "results": provenance(results_file), "split": split, "count": size})
    print(f"EVALUATION COMPLETE {split}: {len(results)} methods on {size} questions", flush=True)


if __name__ == "__main__":
    entrypoint(evaluate, __doc__, options)
