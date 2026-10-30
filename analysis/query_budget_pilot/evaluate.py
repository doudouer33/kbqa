#!/usr/bin/env python3
"""Evaluate argmax budgets against saved per-k RAG curves; no new retrieval."""

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import PARTITIONS, fingerprint, fresh_directory, read_json, require, sha256, write_json, write_jsonl
from train import load_inputs, make_model

METRICS = ("answer_f1", "context_tokens", "prompt_tokens", "evidence_recall", "mean_k", "k1_ratio", "dominant_k_ratio")
FIXED_KS = (1, 3, 5, 13)


def summarize(ids, budgets, curves, labels):
    require(bool(ids), "Cannot summarize an empty group")
    counts = Counter(budgets[qid] for qid in ids)
    require(all(isinstance(k, int) and 1 <= k <= 15 for k in counts), "Invalid predicted k")
    result = {"n": len(ids), "valid_label_n": sum(labels[qid]["has_valid_label"] for qid in ids)}
    for metric, source in (
        ("answer_f1", "f1"),
        ("context_tokens", "context_tokens"),
        ("prompt_tokens", "prompt_tokens"),
        ("evidence_recall", "evidence_recall"),
    ):
        result[metric] = statistics.mean(curves[qid][source][budgets[qid] - 1] for qid in ids)
    result.update(
        mean_k=statistics.mean(budgets[qid] for qid in ids),
        k1_ratio=counts[1] / len(ids),
        k_distribution={str(k): counts[k] for k in range(1, 16)},
        dominant_k=min(counts, key=lambda k: (-counts[k], k)),
        dominant_k_ratio=max(counts.values()) / len(ids),
        unique_k_count=len(counts),
        distribution_entropy=-sum((n / len(ids)) * math.log(n / len(ids)) for n in counts.values()),
        collapse_flag=max(counts.values()) / len(ids) >= 0.95,
    )
    return result


def evaluate(run, output=None):
    import numpy as np
    import torch

    run = Path(run)
    training = read_json(run / "training.json")
    config, seed = training["configuration"], training["seed"]
    prepared, embeddings, metadata = (Path(training[key]) for key in ("prepared", "embeddings", "embedding_metadata"))
    require(
        training["input_sha256"] == fingerprint(prepared, embeddings, metadata, seed),
        "Training inputs changed before evaluation",
    )
    require(training["checkpoint_sha256"] == sha256(run / "best.pt"), "Checkpoint changed")
    queries, labels, curves, split, features = load_inputs(config, seed, prepared, embeddings, metadata)
    torch.set_num_threads(training["settings"]["num_threads"])
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    settings = training["settings"]
    model = make_model(training["model"], features.shape[1], settings["hidden_dim"], settings["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.from_numpy(features)), dim=1).numpy()
    require(bool(np.isfinite(probabilities).all()), "Non-finite predictions")
    predicted = {q["question_id"]: int(probabilities[i].argmax()) + 1 for i, q in enumerate(queries)}
    methods = {"model": predicted}
    methods.update({"fixed_k{}".format(k): {q["question_id"]: k for q in queries} for k in FIXED_KS})
    # A reference with real full-k information, defined for ALL IDs including invalid labels.
    methods["offline_max_f1_oracle"] = {
        qid: max(range(1, 16), key=lambda k: curve["f1"][k - 1]) for qid, curve in curves.items()
    }
    metrics = []
    for part in PARTITIONS:
        ids = split["partitions"][part]
        groups = {"combined": ids}
        groups.update({kind: [qid for qid in ids if labels[qid]["type"] == kind] for kind in ("bridge", "comparison")})
        for group, group_ids in groups.items():
            for cohort in ("all", "valid", "unlabeled"):
                selected = [
                    qid for qid in group_ids if cohort == "all" or labels[qid]["has_valid_label"] == (cohort == "valid")
                ]
                if not selected:
                    continue
                summaries = {method: summarize(selected, ks, curves, labels) for method, ks in methods.items()}
                for method, summary in summaries.items():
                    deltas = {
                        fixed: {key: summary[key] - summaries[fixed][key] for key in METRICS}
                        for fixed in summaries
                        if fixed.startswith("fixed_")
                    }
                    metrics.append(
                        {
                            "partition": part,
                            "group": group,
                            "cohort": cohort,
                            "method": method,
                            **summary,
                            "delta_vs_fixed": deltas,
                        }
                    )
    output = fresh_directory(output or run / "evaluation")
    membership = {qid: part for part, ids in split["partitions"].items() for qid in ids}
    write_jsonl(
        output / "predictions.jsonl",
        [
            {
                "question_id": q["question_id"],
                "partition": membership[q["question_id"]],
                "type": labels[q["question_id"]]["type"],
                "has_valid_label": labels[q["question_id"]]["has_valid_label"],
                "probabilities": probabilities[i].tolist(),
                "predicted_k": predicted[q["question_id"]],
            }
            for i, q in enumerate(queries)
        ],
    )
    result = {
        "model": training["model"],
        "label": training["label"],
        "seed": seed,
        "metrics": metrics,
        "test_ids": split["partitions"]["test"],
        "input_sha256": training["input_sha256"],
        "best_epoch": training["best_epoch"],
        "best_validation_loss": training["best_validation_loss"],
        "oracle_notice": "offline_max_f1_oracle uses true per-k information; not a deployable model. Its F1 is an upper bound on this saved table only.",
    }
    write_json(output / "metrics.json", result)
    return result


def mean_std(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def aggregate(matrix, output=None):
    matrix = Path(matrix)
    manifest = read_json(matrix / "matrix.json")
    require(manifest["status"] == "complete", "Matrix is incomplete")
    require(len(set(manifest["seeds"])) >= 3, "Aggregation requires at least three seeds")
    expected = {
        (seed, model, label)
        for seed in manifest["seeds"]
        for model in manifest["models"]
        for label in manifest["labels"]
    }
    results = [read_json(matrix / relative / "evaluation" / "metrics.json") for relative in manifest["runs"]]
    require(
        len(results) == len(expected) and {(r["seed"], r["model"], r["label"]) for r in results} == expected,
        "Missing or duplicate matrix runs",
    )
    same_test, shared_hashes, grouped = {}, None, defaultdict(list)
    baseline_seen = set()
    for result in results:
        seed = result["seed"]
        ids = result["test_ids"]
        require(ids == same_test.setdefault(seed, ids), "Models compared on different test IDs")
        hashes = {k: v for k, v in result["input_sha256"].items() if not k.startswith("seed_")}
        if shared_hashes is None:
            shared_hashes = hashes
        require(hashes == shared_hashes, "Matrix data or embeddings differ")
        for row in result["metrics"]:
            method = result["model"] + "/" + result["label"] if row["method"] == "model" else row["method"]
            key = (method, row["partition"], row["group"], row["cohort"])
            unique = (seed,) + key
            if row["method"] != "model":
                if unique in baseline_seen:
                    continue
                baseline_seen.add(unique)
            grouped[key].append(row)
    summary = []
    for key, rows in sorted(grouped.items()):
        require(len(rows) == len(manifest["seeds"]), "Unequal seed coverage in aggregate")
        item = dict(zip(("method", "partition", "group", "cohort"), key))
        item.update(repeats=len(rows), **{m: mean_std([r[m] for r in rows]) for m in METRICS})
        item["k_distribution_mean"] = {
            str(k): statistics.mean(r["k_distribution"][str(k)] for r in rows) for k in range(1, 16)
        }
        item["collapse_seed_count"] = sum(r["collapse_flag"] for r in rows)
        item["sample_count"] = mean_std([r["n"] for r in rows])
        item["delta_vs_fixed"] = {
            fixed: {m: mean_std([r["delta_vs_fixed"][fixed][m] for r in rows]) for m in METRICS}
            for fixed in rows[0]["delta_vs_fixed"]
        }
        summary.append(item)
    output = fresh_directory(output or matrix / "report")
    write_json(
        output / "summary.json",
        {
            "seeds": manifest["seeds"],
            "std_definition": "sample SD across repeated split+training seeds; not a confidence interval",
            "rows": summary,
        },
    )
    csv_rows = []
    for item in summary:
        row = {k: item[k] for k in ("method", "partition", "group", "cohort", "repeats", "collapse_seed_count")}
        row.update({m + "_" + stat: item[m][stat] for m in METRICS for stat in ("mean", "std")})
        csv_rows.append(row)
    with (output / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    make_figure(output, summary)
    make_report(output, summary, manifest, results)
    return summary


def make_figure(output, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, group in zip(axes, ("combined", "bridge", "comparison")):
        for row in summary:
            if (row["partition"], row["group"], row["cohort"]) != ("test", group, "all"):
                continue
            axis.errorbar(
                row["context_tokens"]["mean"],
                row["answer_f1"]["mean"],
                xerr=row["context_tokens"]["std"],
                yerr=row["answer_f1"]["std"],
                marker="x" if row["method"].startswith("offline") else "o",
                capsize=2,
                label=row["method"],
            )
        axis.set(title=group, xlabel="Mean context tokens", ylabel="Mean answer F1")
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(output / "f1_token.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_report(output, summary, manifest, results):
    def fmt(row, key, scale=1):
        value = row[key]
        return "{:.4f} ± {:.4f}".format(value["mean"] * scale, value["std"] * scale)

    lines = [
        "# Query-only Dynamic Top-K 探索性实验",
        "",
        "输入仅为冻结 BGE-M3 的 Query Dense Embedding，预算取 15 类概率的 argmax（并列取最小 k）。",
        "样本来自已用于方法设计的 HotpotQA Dev；本报告不构成独立泛化测试。每个 seed 内所有模型使用完全相同的测试 ID，主表包含全部无标签题。",
        "训练与早停只使用有标签的 Train / Validation；Test 不参与参数、检查点或固定预算选择。固定 k=13 为用户预先指定，历史上曾在整批 Dev 上表现最好，也应按探索性结果解释。",
        "均值 ± 样本标准差来自随机种子 {}，同时改变划分和初始化；跨 seed 测试集可重叠，不是独立置信区间。".format(
            manifest["seeds"]
        ),
        "",
        "offline_max_f1_oracle 使用真实逐 k F1，是保存结果表上的 F1 参考上界，不能部署；其成本不是成本下界。",
        "",
    ]
    for part in PARTITIONS:
        lines.extend(
            [
                "## {} 全部样本".format(part),
                "",
                "| 方法 | 分组 | N | F1 | Context Tokens | Prompt Tokens | Evidence Recall | k1比例 | ≥95%同类的seed数 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in summary:
            if row["partition"] == part and row["cohort"] == "all":
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        row["method"],
                        row["group"],
                        fmt(row, "sample_count"),
                        fmt(row, "answer_f1"),
                        fmt(row, "context_tokens"),
                        fmt(row, "prompt_tokens"),
                        fmt(row, "evidence_recall"),
                        fmt(row, "k1_ratio"),
                        row["collapse_seed_count"],
                    )
                )
        lines.append("")
    lines.extend(
        [
            "## Test 相对固定预算的配对差",
            "",
            "同一 seed、同一完整测试集先相减，再跨 seed 汇总。ΔF1>0 表示质量增加，ΔTokens<0 表示成本减少。",
            "",
            "| 模型/标签 | 固定基线 | ΔF1 | ΔContext Tokens |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in summary:
        if (row["partition"], row["group"], row["cohort"]) == ("test", "combined", "all") and "/" in row["method"]:
            for fixed, delta in row["delta_vs_fixed"].items():
                lines.append(
                    "| {} | {} | {} | {} |".format(
                        row["method"], fixed, fmt(delta, "answer_f1"), fmt(delta, "context_tokens")
                    )
                )
    lines.extend(
        [
            "",
            "## 训练与预测诊断",
            "",
            "| 模型 | 标签 | Seed | 最佳轮数 | Validation Loss |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        lines.append(
            "| {} | {} | {} | {} | {:.6f} |".format(
                result["model"], result["label"], result["seed"], result["best_epoch"], result["best_validation_loss"]
            )
        )
    lines.extend(
        [
            "",
            "每次运行的 history.json 保存逐轮 Train/Validation Loss；evaluation/predictions.jsonl 保存全部 ID 的概率和 k。summary.json 同时提供各 k 平均计数、k1比例、有效/无标签分组和相对固定 k 的差。95%同类标志只是预测塌缩的描述性阈值；固定 k 天然会触发此标志。",
            "",
            "![F1–Token](f1_token.png)",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path)
    mode.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.run:
        evaluate(args.run, args.output)
        print("Offline evaluation saved")
    else:
        aggregate(args.matrix, args.output)
        print("Aggregate report saved")


if __name__ == "__main__":
    main()
