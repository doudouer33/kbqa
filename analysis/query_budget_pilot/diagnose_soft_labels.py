#!/usr/bin/env python3
"""Read saved Original Soft Label probabilities; no training, embeddings, or RAG imports."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (13, 42, 2026)
MODELS = ("linear", "mlp")
METHODS = ("mean_distribution",) + MODELS
EPSILON = 1e-12
WIN_TOLERANCE = 1e-12
PARAMETERS = {"lambda": 0.1, "temperature": 0.1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_index(path):
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["question_id"]
            require(isinstance(qid, str) and qid and qid not in result, "Invalid/duplicate question ID: " + str(path))
            result[qid] = row
    return result


def probabilities(values):
    """Validate before normalizing float32 serialization roundoff in float64."""
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 2 and array.shape[0] > 0 and array.shape[1] == 15, "Expected nonempty N x 15 probabilities")
    require(np.isfinite(array).all() and (array >= 0).all() and (array <= 1).all(), "Invalid probability value")
    totals = array.sum(axis=1, keepdims=True)
    require(np.all(np.abs(totals - 1) <= 1e-6), "Probabilities do not sum to one")
    return array / totals


def loss_vectors(target, prediction):
    """Return per-query KL(P||Q), CE, H(P); floor Q only, then renormalize."""
    target, prediction = probabilities(target), probabilities(prediction)
    require(target.shape == prediction.shape, "Loss inputs must have the same shape")
    safe_prediction = np.maximum(prediction, EPSILON)
    safe_prediction /= safe_prediction.sum(axis=1, keepdims=True)
    log_target = np.zeros_like(target)
    np.log(target, out=log_target, where=target > 0)
    entropy = -(target * log_target).sum(axis=1)
    ce = -(target * np.log(safe_prediction)).sum(axis=1)
    kl = ce - entropy
    require(np.isfinite(kl).all() and (kl >= -1e-12).all(), "Invalid KL divergence")
    return kl, ce, entropy


def center(values):
    # Subtracting an anchor first keeps repeated constant distributions exactly constant.
    return values - (values[0] + (values - values[0]).mean(axis=0))


def sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) <= 1e-14 or np.ptp(y) <= 1e-14:
        return None
    x, y = center(x), center(y)
    return float(np.clip(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1))


def average_ranks(values):
    """Average ranks for ties, as required for Spearman correlation."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def correlations(x, y):
    if len(x) < 2 or np.ptp(x) <= 1e-14 or np.ptp(y) <= 1e-14:
        return {"pearson": None, "spearman": None}
    return {"pearson": pearson(x, y), "spearman": pearson(average_ranks(x), average_ranks(y))}


def tail_mass(values, threshold):
    mass = values[:, threshold - 1 :].sum(axis=1)
    # All support in the tail means exactly 1, independent of summation roundoff.
    # Otherwise tiny normalization errors spuriously break Spearman ties at 1.
    mass[(values[:, : threshold - 1] == 0).all(axis=1)] = 1.0
    return np.clip(mass, 0, 1)


def train_mean(labels, split):
    # This is the ONLY source of the deployed constant baseline. Never include val/test.
    ids = [qid for qid in split["partitions"]["train"] if labels[qid]["has_valid_label"]]
    require(bool(ids), "No valid Original Soft Labels in train")
    targets = probabilities([labels[qid]["targets"]["original_soft"]["probabilities"] for qid in ids])
    return ids, targets.mean(axis=0)


def validate_data(prepared, seeds):
    metadata = read_json(prepared / "metadata.json")
    require(
        metadata["pilot_configuration"]["labels"]["original_soft"] == PARAMETERS, "Wrong Original Soft Label parameters"
    )
    labels = read_index(prepared / "labels.jsonl")
    require(len(labels) == metadata["query_count"], "Unexpected label count")
    for row in labels.values():
        target = row["targets"]["original_soft"]
        require(isinstance(row["has_valid_label"], bool), "Non-boolean validity flag")
        if row["has_valid_label"]:
            require(target["status"] == "valid", "Inconsistent valid target")
            probabilities([target["probabilities"]])
        else:
            require(
                target["probabilities"] is None and target["status"] in ("all_zero_f1", "empty_candidates"),
                "Invalid target must be null",
            )
    splits = {}
    for seed in seeds:
        relative = "splits/seed_{}.json".format(seed)
        for artifact in ("labels.jsonl", relative):
            require(
                sha256(prepared / artifact) == metadata["artifacts_sha256"][artifact],
                "Prepared artifact hash mismatch: " + artifact,
            )
        split = read_json(prepared / relative)
        require(split["seed"] == seed, "Wrong split seed")
        require(set(split["partitions"]) == {"train", "validation", "test"}, "Wrong split partitions")
        ids = [qid for part in split["partitions"].values() for qid in part]
        require(len(ids) == len(set(ids)) and set(ids) == set(labels), "Split overlap, duplicate, or missing IDs")
        splits[seed] = split
    return labels, splits


def load_prediction(run, prepared, seed, model, labels, split, test_ids):
    training = read_json(run / "training.json")
    require(
        (training["seed"], training["model"], training["label"]) == (seed, model, "original_soft"),
        "Run identity mismatch: " + str(run),
    )
    require(
        training["configuration"]["labels"]["original_soft"] == PARAMETERS, "Run used different soft label parameters"
    )
    for relative in ("metadata.json", "labels.jsonl", "splits/seed_{}.json".format(seed)):
        require(
            training["input_sha256"][Path(relative).name] == sha256(prepared / relative),
            "Training input hash mismatch: " + relative,
        )
    predictions = read_index(run / "evaluation" / "predictions.jsonl")
    missing = sorted(set(test_ids) - set(predictions))
    require(not missing, "Missing valid Test predictions (no intersection filtering): " + str(missing))
    membership = {qid: part for part, ids in split["partitions"].items() for qid in ids}
    for qid, row in predictions.items():
        require(qid in labels, "Unknown prediction question ID: " + qid)
        require(row["partition"] == membership[qid], "Prediction partition mismatch: " + qid)
        require(
            row["has_valid_label"] == labels[qid]["has_valid_label"] and row["type"] == labels[qid]["type"],
            "Prediction label metadata mismatch: " + qid,
        )
        vector = probabilities([row["probabilities"]])
        require(row["predicted_k"] == int(vector[0].argmax()) + 1, "Saved argmax mismatch: " + qid)
    array = probabilities([predictions[qid]["probabilities"] for qid in test_ids])
    return array, training["best_epoch"]


def summarize(seed, method, target, prediction, baseline, baseline_kl, best_epoch=None):
    kl, ce, entropy = loss_vectors(target, prediction)
    mean_prediction = prediction.mean(axis=0)
    own_mean_kl = loss_vectors(target, np.tile(mean_prediction, (len(target), 1)))[0]
    delta = baseline_kl - kl
    centered = center(prediction)
    query_std = np.sqrt((centered**2).sum(axis=0) / max(len(target) - 1, 1))
    result = {
        "seed": seed,
        "method": method,
        "n_test": len(target),
        "best_epoch": best_epoch,
        "kl": float(kl.mean()),
        "soft_ce": float(ce.mean()),
        "target_entropy": float(entropy.mean()),
        "kl_gain": float(delta.mean()),
        "relative_kl_gain": float(delta.mean() / baseline_kl.mean()) if baseline_kl.mean() > EPSILON else None,
        "win_fraction": float((delta > WIN_TOLERANCE).mean()),
        "tie_fraction": float((np.abs(delta) <= WIN_TOLERANCE).mean()),
        "query_tv_to_own_mean": float((0.5 * np.abs(centered).sum(axis=1)).mean()),
        "query_tv_to_train_mean": float((0.5 * np.abs(prediction - baseline).sum(axis=1)).mean()),
        "mean_shift_tv": float(0.5 * np.abs(mean_prediction - baseline).sum()),
        "mean_per_k_query_std": float(query_std.mean()),
        "own_mean_kl": float(own_mean_kl.mean()),
        "query_kl_gain_vs_own_mean": float((own_mean_kl - kl).mean()),
        "argmax_k1_k2_fraction": float((prediction.argmax(axis=1) < 2).mean()),
        "floored_prediction_entries": int((prediction < EPSILON).sum()),
    }
    query_rows = []
    for threshold in (5, 8):
        true_mass, predicted_mass = tail_mass(target, threshold), tail_mass(prediction, threshold)
        prefix = "mass_ge{}_".format(threshold)
        result.update(
            {
                prefix + key: value
                for key, value in {
                    "target_mean": float(true_mass.mean()),
                    "target_std": sample_std(true_mass),
                    "pred_mean": float(predicted_mass.mean()),
                    "pred_std": sample_std(predicted_mass),
                    **correlations(true_mass, predicted_mass),
                }.items()
            }
        )
    for i in range(len(target)):
        query_rows.append(
            {
                "seed": seed,
                "method": method,
                "kl": float(kl[i]),
                "soft_ce": float(ce[i]),
                "target_entropy": float(entropy[i]),
                "baseline_kl": float(baseline_kl[i]),
                "kl_gain": float(delta[i]),
                "predicted_k": int(prediction[i].argmax()) + 1,
                **{
                    "{}_mass_ge{}".format(name, k): float(tail_mass(value[i : i + 1], k)[0])
                    for name, value in (("target", target), ("pred", prediction))
                    for k in (5, 8)
                },
            }
        )
    return result, query_rows, query_std


def aggregate(rows):
    results = []
    keys = [key for key in rows[0] if key not in ("seed", "method", "best_epoch")]
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        item = {"method": method, "n_seeds": len(selected)}
        for key in keys:
            values = [row[key] for row in selected if row[key] is not None]
            item[key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": sample_std(values) if values else None,
                "n": len(values),
            }
        results.append(item)
    return results


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figures(output, rows, per_query, per_k):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"mean_distribution": "#555555", "linear": "#3073b8", "mlp": "#dd8136", "target": "#398547"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        axes[0].plot(range(len(SEEDS)), [r["kl"] for r in selected], "o-", label=method, color=colors[method])
    axes[0].set(xticks=range(len(SEEDS)), xticklabels=SEEDS, xlabel="Seed", ylabel="Mean KL (nats)")
    for offset, method in zip((-0.18, 0.18), MODELS):
        axes[1].bar(
            np.arange(len(SEEDS)) + offset,
            [r["relative_kl_gain"] * 100 for r in rows if r["method"] == method],
            width=0.36,
            label=method,
            color=colors[method],
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xticks=range(len(SEEDS)), xticklabels=SEEDS, xlabel="Seed", ylabel="KL gain vs train mean (%)")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "kl_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for row_index, method in enumerate(MODELS):
        for col_index, threshold in enumerate((5, 8)):
            axis = axes[row_index, col_index]
            for seed in SEEDS:
                selected = [r for r in per_query if r["method"] == method and r["seed"] == seed]
                axis.scatter(
                    [r["target_mass_ge{}".format(threshold)] for r in selected],
                    [r["pred_mass_ge{}".format(threshold)] for r in selected],
                    s=9,
                    alpha=0.35,
                    label=str(seed),
                )
            axis.plot([0, 1], [0, 1], "k--", linewidth=0.7)
            axis.set(
                title="{}: mass(k >= {})".format(method, threshold),
                xlabel="True soft-label mass",
                ylabel="Predicted mass",
                xlim=(-0.02, 1.02),
                ylim=(-0.02, 1.02),
            )
            axis.grid(alpha=0.15)
    axes[0, 0].legend(title="Seed", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "tail_mass_scatter.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for method in ("target",) + METHODS:
        selected = [r for r in per_k if r["method"] == method]
        for axis, field in zip(axes, ("mean_probability", "query_std")):
            values = [np.mean([r[field] for r in selected if r["k"] == k]) for k in range(1, 16)]
            axis.plot(range(1, 16), values, "o-", markersize=3, label=method, color=colors[method])
    axes[0].set(xlabel="k", ylabel="Mean probability", title="Mean across seeds")
    axes[1].set(xlabel="k", ylabel="Across-query sample SD", title="Mean SD across seeds")
    for axis in axes:
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "distribution_variation.png", dpi=150)
    plt.close(fig)


def make_report(output, rows, aggregated, baselines):
    def number(value, scale=1, digits=4):
        return "—" if value is None else ("{: ." + str(digits) + "f}").format(value * scale).strip()

    def avg(row, key, scale=1, digits=4):
        if row[key]["mean"] is None:
            return "—"
        return "{} ± {}".format(number(row[key]["mean"], scale, digits), number(row[key]["std"], scale, digits))

    lines = [
        "# Query-specific Soft Label 可学习性诊断",
        "",
        "仅复用 `matrix_001` 中 Linear / MLP 的 Original Soft Label（λ=0.1、T=0.1）已有预测；未训练、未加载权重或 Embedding，未运行 RAG / LLM。",
        "",
        "## 比较口径",
        "",
        "Mean Distribution Baseline 仅对各 seed 的有效 **Train** Original Soft Label 求算术平均；不使用 Validation / Test 标签。每个 seed 的三种方法严格按相同有效 Test ID 连接，缺失预测即报错，不通过取交集缩小测试集。无效标签不进入本诊断。",
        "",
        "| Seed | Train 全部 / 有效 | Validation 全部 / 有效 | Test 全部 / 有效 |",
        "| --- | --- | --- | --- |",
    ]
    for baseline in baselines:
        counts = baseline["partition_counts"]
        lines.append(
            "| {} | {} / {} | {} / {} | {} / {} |".format(
                baseline["seed"],
                *[v for part in ("train", "validation", "test") for v in (counts[part]["all"], counts[part]["valid"])]
            )
        )
    lines.extend(
        [
            "",
            "KL 为 D_KL(P_oracle || P_pred)，自然对数，单位 nats。Soft CE = KL + H(P_oracle)，同一 seed 的目标熵完全相同。概率先校验并以 float64 归一化；预测概率下限 1e-12 后重新归一化，目标零项贡献为 0。因此若预测有零概率，所报为下限平滑后的有限 KL。当前共 {} 个预测概率项低于下限。".format(
                sum(r["floored_prediction_entries"] for r in rows)
            ),
            "",
            "所有 ± 为三个 seed 的**样本标准差**（ddof=1），相对改善先逐 seed 计算再汇总。种子同时改变划分与初始化，测试 ID 可重叠；这些波动不是置信区间，本文不作统计显著性声明。",
            "",
            "## KL 与 Soft Cross Entropy",
            "",
            "改善量 = KL_baseline − KL_model；相对改善 = 改善量 / KL_baseline。逐题胜率使用同一题的 KL，差值 >1e-12 算胜；负改善表示差于基线。",
            "",
            "| Seed | 方法 | KL ↓ | Soft CE ↓ | KL 改善 ↑ | 相对改善 (%) ↑ | 逐题胜率 (%) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                r["seed"],
                r["method"],
                *[
                    number(r[key], scale)
                    for key, scale in (
                        ("kl", 1),
                        ("soft_ce", 1),
                        ("kl_gain", 1),
                        ("relative_kl_gain", 100),
                        ("win_fraction", 100),
                    )
                ]
            )
        )
    for r in aggregated:
        lines.append(
            "| 均值 ± SD | {} | {} | {} | {} | {} | {} |".format(
                r["method"],
                *[
                    avg(r, key, scale)
                    for key, scale in (
                        ("kl", 1),
                        ("soft_ce", 1),
                        ("kl_gain", 1),
                        ("relative_kl_gain", 100),
                        ("win_fraction", 100),
                    )
                ]
            )
        )
    lines.extend(
        [
            "",
            "![KL 比较](kl_comparison.png)",
            "",
            "## 分布是否随 Query 改变",
            "",
            "TV(P,Q)=½Σ|P−Q|。逐题 TV 至自身均值衡量变化幅度；至 Train 均值包含整体偏移。为区分整体校准与逐题信息，还把每个模型所有 Test **预测**取平均并广播，计算“自身均值 KL”；它不使用 Test 标签构造概率，仅用于消除逐题变化的描述性对照，不能替代 Train Mean Baseline。逐题增益 = 自身均值 KL − 原模型 KL，正值才表示保留逐题变化有帮助。",
            "",
            "| Seed | 方法 | TV 至自身均值 | TV 至 Train 均值 | 自身均值 KL | 逐题增益 | argmax 为 1/2 (%) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                r["seed"],
                r["method"],
                *[
                    number(r[key], scale, 6 if key == "query_kl_gain_vs_own_mean" else 4)
                    for key, scale in (
                        ("query_tv_to_own_mean", 1),
                        ("query_tv_to_train_mean", 1),
                        ("own_mean_kl", 1),
                        ("query_kl_gain_vs_own_mean", 1),
                        ("argmax_k1_k2_fraction", 100),
                    )
                ]
            )
        )
    for r in aggregated:
        lines.append(
            "| 均值 ± SD | {} | {} | {} | {} | {} | {} |".format(
                r["method"],
                *[
                    avg(r, key, scale, digits=6 if key == "query_kl_gain_vs_own_mean" else 4)
                    for key, scale in (
                        ("query_tv_to_own_mean", 1),
                        ("query_tv_to_train_mean", 1),
                        ("own_mean_kl", 1),
                        ("query_kl_gain_vs_own_mean", 1),
                        ("argmax_k1_k2_fraction", 100),
                    )
                ]
            )
        )
    lines.extend(
        [
            "",
            "![平均概率与逐题变化幅度](distribution_variation.png)",
            "",
            "左图为跨 seed 平均概率，右图为每个 seed 内逐题概率标准差的跨 seed 均值。完整逐 k 均值/标准差见 `per_k.csv`。",
            "",
            "## 大 k 概率质量与相关性",
            "",
            "质量是 Σ_{k≥阈值}P(k)，不是 argmax 落入大 k 的比例。若全部非零概率都在阈值内，质量精确记 1，避免浮点求和误差拆散 Spearman 并列秩。下表中的均值 / SD 是同一 seed 内不同 Query 的质量均值 / 样本标准差；末尾汇总的 ± 才是跨 seed 波动。Pearson / Spearman 逐 seed 计算，不混合跨 seed 重复问题；常量基线的相关性未定义，记 —（JSON 为 null）。",
            "",
            "| Seed | 方法 | 阈值 | 真实质量 均值 / SD | 预测质量 均值 / SD | Pearson | Spearman |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in rows:
        for k in (5, 8):
            p = "mass_ge{}_".format(k)
            lines.append(
                "| {} | {} | k≥{} | {} / {} | {} / {} | {} | {} |".format(
                    r["seed"],
                    r["method"],
                    k,
                    *[
                        number(r[p + key])
                        for key in ("target_mean", "target_std", "pred_mean", "pred_std", "pearson", "spearman")
                    ]
                )
            )
    lines.extend(
        [
            "",
            "| 方法 | 阈值 | 真实质量均值 ± SD | 预测质量均值 ± SD | Pearson ± SD | Spearman ± SD |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in aggregated:
        for k in (5, 8):
            lines.append(
                "| {} | k≥{} | {} | {} | {} | {} |".format(
                    r["method"],
                    k,
                    *[
                        avg(r, "mass_ge{}_".format(k) + key)
                        for key in ("target_mean", "pred_mean", "pearson", "spearman")
                    ]
                )
            )
    lines.extend(
        [
            "",
            "![真实与预测的大 k 质量](tail_mass_scatter.png)",
            "",
            "虚线为理想的真实质量=预测质量；各 seed 的点仅用于可视化，相关系数仍逐 seed 计算。",
            "",
            "## 结论与下一步",
            "",
        ]
    )
    for r in aggregated:
        if r["method"] in MODELS:
            wins = sum(row["kl_gain"] > WIN_TOLERANCE for row in rows if row["method"] == r["method"])
            lines.append(
                "- **{}**：相对 Train Mean 的 KL 改善为 {}%，{} / 3 个 seed 为正；逐题变化相对自身均值的 KL 增益仅 {} nats。".format(
                    r["method"], avg(r, "relative_kl_gain", 100), wins, avg(r, "query_kl_gain_vs_own_mean", digits=6)
                )
            )
    model_rows = [r for r in rows if r["method"] in MODELS]
    corr_values = [
        r["mass_ge{}_{}".format(k, name)] for r in model_rows for k in (5, 8) for name in ("pearson", "spearman")
    ]
    corr_values = [v for v in corr_values if v is not None]
    all_average_gains_nonpositive = all(r["kl_gain"]["mean"] <= 0 for r in aggregated if r["method"] in MODELS)
    lines.append(
        "- **是否明显优于平均分布？** "
        + (
            "当前没有：两个模型的跨 seed 平均 KL 改善均不为正，不能宣称有稳定收益。"
            if all_average_gains_nonpositive
            else "请结合上表逐 seed 的改善幅度与方向判断；少量重复种子不足以作统计显著性声明。"
        )
    )
    if corr_values:
        lines.append(
            "- **能否预测大 k 偏好？** 模型逐 seed 的 Pearson / Spearman 范围为 {} 至 {}。{}均值质量接近不能替代逐题相关性。".format(
                number(min(corr_values)),
                number(max(corr_values)),
                "当前相关性接近零，证据很弱；" if max(abs(v) for v in corr_values) < 0.1 else "",
            )
        )
    lines.extend(
        [
            "- **是否有 Query-specific 分布信息？** "
            + (
                "输出并非完全恒定，但变化幅度较小；消除逐题变化的对照仅有极小 KL 增益，且没有稳定超越 Train Mean。最多支持很弱的迹象，尚不足以证明有实用的逐题预算分布学习能力。"
                if all_average_gains_nonpositive and corr_values and max(abs(v) for v in corr_values) < 0.1
                else "输出随 Query 改变本身不构成学会逐题分布的证据；还需与 Train Mean 的配对改善、大 k 相关性及消除逐题变化后的 KL 对照一起解释。"
            )
            + "此结论针对当前模型、数据量与标签。",
            "- **概率学习与决策分开看。** argmax 集中在 1、2 本身不能判定 KL 学习失败；平均分布仍给大 k 分配大量概率。这里的判断来自配对 KL、变化幅度和大 k 相关性，而不是 argmax 频次。",
            "- **扩大数据的启示。** 不宜直接假设扩大规模会改善动态预算。后续若扩充数据，应保留 Train Mean 与相同测试 ID 的配对评价，观察相对 KL 改善和大 k 相关性是否随有效训练样本增加而稳定提升；当前结果无法区分数据不足、Embedding 信号弱或标签噪声等原因。无需据此扩展本次诊断或立即启动新训练。",
            "",
            "本次仅诊断有效标签的各 seed Test 子集，不能推广到无有效标签的问题。现有 HotpotQA Dev 已用于方法设计，结果为探索性。六次既有训练的最佳检查点在第 {}–{} 轮，不能仅凭本诊断判定训练数据是唯一瓶颈。".format(
                min(r["best_epoch"] for r in model_rows), max(r["best_epoch"] for r in model_rows)
            ),
            "",
            "## 复现与产物",
            "",
            "`diagnose_soft_labels.py` 仅读取标签、保存的划分、训练元数据和 `predictions.jsonl`。所需文件齐全；来源路径、SHA256 及运行后未改动校验见 `input_audit.json`。缺失文件时只写缺失报告并返回退出码 2，绝不补训。",
            "",
            "- `baseline_distributions.json`：各 seed 有效 Train/Test ID、训练均值 15 维分布及划分计数。",
            "- `per_seed.csv` / `summary.csv` / `summary.json`：逐 seed 指标与跨 seed 均值、样本标准差；不把 450 个跨 seed Test 行当作独立样本。",
            "- `per_query.csv`：每题 KL、CE、基线差、大 k 质量与 argmax；`per_k.csv`：分布逐维均值与标准差。",
            "- `verification.json`：实际运行后的计算测试与只读核验记录。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_diagnostic(prepared, matrix, output):
    prepared, matrix, output = (Path(p).resolve() for p in (prepared, matrix, output))
    require(not output.exists(), "Output exists; choose a new diagnostic directory: " + str(output))
    require(
        all(source not in output.parents for source in (prepared, matrix)),
        "Diagnostic output must be outside existing prepared/training results",
    )
    files = [prepared / "metadata.json", prepared / "labels.jsonl", matrix / "matrix.json"]
    for seed in SEEDS:
        files.append(prepared / "splits" / "seed_{}.json".format(seed))
        for model in MODELS:
            run = matrix / "seed_{}".format(seed) / (model + "_original_soft")
            files.extend([run / "training.json", run / "evaluation" / "predictions.jsonl"])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        output.mkdir(parents=True)
        result = {
            "status": "missing_inputs",
            "missing_files": missing,
            "action": "No training or inference was attempted.",
        }
        write_json(output / "input_audit.json", result)
        (output / "report.md").write_text(
            "# 诊断未完成：已有文件缺失\n\n无法在相同 Test 有效标签子集上完成全部模型比较。未重新训练、生成 Embedding 或运行 RAG / LLM。\n\n缺失文件：\n\n"
            + "\n".join("- `" + path + "`" for path in missing)
            + "\n",
            encoding="utf-8",
        )
        return result
    before = {str(path): sha256(path) for path in files}
    manifest = read_json(matrix / "matrix.json")
    require(
        manifest["status"] == "complete" and set(SEEDS) <= set(manifest["seeds"]), "Incomplete matrix or missing seeds"
    )
    require(
        all("seed_{}/{}_original_soft".format(seed, model) in manifest["runs"] for seed in SEEDS for model in MODELS),
        "Original Soft Label run missing from matrix manifest",
    )
    labels, splits = validate_data(prepared, SEEDS)
    rows, per_query, per_k, baselines = [], [], [], []
    for seed in SEEDS:
        split = splits[seed]
        train_ids, mean = train_mean(labels, split)
        test_ids = [qid for qid in split["partitions"]["test"] if labels[qid]["has_valid_label"]]
        require(bool(test_ids), "No valid Test labels")
        target = probabilities([labels[qid]["targets"]["original_soft"]["probabilities"] for qid in test_ids])
        predictions = {"mean_distribution": np.tile(mean, (len(target), 1))}
        epochs = {"mean_distribution": None}
        for model in MODELS:
            run = matrix / "seed_{}".format(seed) / (model + "_original_soft")
            predictions[model], epochs[model] = load_prediction(run, prepared, seed, model, labels, split, test_ids)
        baseline_kl = loss_vectors(target, predictions["mean_distribution"])[0]
        baselines.append(
            {
                "seed": seed,
                "source": "valid Original Soft Labels from train partition ONLY",
                "valid_train_ids": train_ids,
                "valid_test_ids": test_ids,
                "probabilities": mean.tolist(),
                "partition_counts": {
                    part: {"all": len(ids), "valid": sum(labels[qid]["has_valid_label"] for qid in ids)}
                    for part, ids in split["partitions"].items()
                },
            }
        )
        for method, prediction in predictions.items():
            summary, query_rows, std = summarize(seed, method, target, prediction, mean, baseline_kl, epochs[method])
            rows.append(summary)
            per_query.extend(dict(question_id=qid, **row) for qid, row in zip(test_ids, query_rows))
            per_k.extend(
                {
                    "seed": seed,
                    "method": method,
                    "k": k + 1,
                    "mean_probability": float(prediction[:, k].mean()),
                    "query_std": float(std[k]),
                }
                for k in range(15)
            )
        std = np.sqrt((center(target) ** 2).sum(axis=0) / max(len(target) - 1, 1))
        per_k.extend(
            {
                "seed": seed,
                "method": "target",
                "k": k + 1,
                "mean_probability": float(target[:, k].mean()),
                "query_std": float(std[k]),
            }
            for k in range(15)
        )
    aggregated = aggregate(rows)
    output.mkdir(parents=True)
    write_json(output / "baseline_distributions.json", baselines)
    write_json(
        output / "summary.json",
        {
            "seeds": SEEDS,
            "parameters": PARAMETERS,
            "epsilon": EPSILON,
            "win_tolerance": WIN_TOLERANCE,
            "std_definition": "sample SD across seeds (ddof=1); not a confidence interval",
            "per_seed": rows,
            "aggregate": aggregated,
        },
    )
    write_csv(output / "per_seed.csv", rows)
    write_csv(
        output / "summary.csv",
        [
            {
                "method": r["method"],
                "n_seeds": r["n_seeds"],
                **{
                    key + "_" + statistic: value[statistic]
                    for key, value in r.items()
                    if isinstance(value, dict)
                    for statistic in ("mean", "std", "n")
                },
            }
            for r in aggregated
        ],
    )
    write_csv(output / "per_query.csv", per_query)
    write_csv(output / "per_k.csv", per_k)
    make_figures(output, rows, per_query, per_k)
    make_report(output, rows, aggregated, baselines)
    after = {str(path): sha256(path) for path in files}
    require(before == after, "An input file changed during diagnosis")
    audit = {
        "status": "complete",
        "prepared": str(prepared),
        "matrix": str(matrix),
        "inputs_unchanged": True,
        "input_sha256_before": before,
        "input_sha256_after": after,
        "code_sha256": sha256(Path(__file__)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "scope": "Saved probabilities only; no model/embedding loading, training, RAG, or LLM.",
    }
    write_json(output / "input_audit.json", audit)
    write_json(
        output / "verification.json",
        {
            "status": "passed",
            "ce_equals_kl_plus_target_entropy_max_abs_error": max(
                abs(r["soft_ce"] - r["kl"] - r["target_entropy"]) for r in per_query
            ),
            "all_methods_identical_valid_test_ids": all(
                [r["question_id"] for r in per_query if r["seed"] == seed and r["method"] == method]
                == baseline["valid_test_ids"]
                for seed, baseline in zip(SEEDS, baselines)
                for method in METHODS
            ),
            "test_counts": {str(b["seed"]): len(b["valid_test_ids"]) for b in baselines},
            "input_files_unchanged": True,
        },
    )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=HERE / "data/prepared_001")
    parser.add_argument("--matrix", type=Path, default=HERE / "outputs/matrix_001")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory outside the existing training/prepared results"
    )
    args = parser.parse_args()
    try:
        result = run_diagnostic(args.prepared, args.matrix, args.output)
    except (ValueError, KeyError, OSError) as error:
        print("Diagnosis failed; no retraining attempted: {}".format(error), file=sys.stderr)
        return 1
    if result["status"] == "missing_inputs":
        print("Missing saved inputs; no retraining attempted:\n" + "\n".join(result["missing_files"]), file=sys.stderr)
        return 2
    print("Saved offline diagnosis: {}".format(args.output / "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
