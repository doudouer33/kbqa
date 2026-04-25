import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 配置
# =========================
INPUT_PATH = "kbqa_classifier/data/merged/dev_500_topk_labels.json"
OUTPUT_DIR = "kbqa_classifier/output/topk_curve_analysis"
EPS = 0.05  # 小于这个幅度的变化视为“不变”
EXAMPLES_PER_TREND_TYPE = 5


# =========================
# 工具函数
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_filename(name):
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def get_sorted_k_list(score_dict):
    return sorted([int(k) for k in score_dict.keys()])


def extract_curve(sample, metric="f1"):
    score_dict = sample["scores"]
    ks = get_sorted_k_list(score_dict)
    ys = [float(score_dict[str(k)][metric]) for k in ks]
    return ks, np.array(ys, dtype=float)


def count_sign_changes(diff, eps=0.05):
    signs = []
    for d in diff:
        if d > eps:
            signs.append(1)
        elif d < -eps:
            signs.append(-1)
        else:
            signs.append(0)

    changes = 0
    last = 0
    for s in signs:
        if s == 0:
            continue
        if last != 0 and s != last:
            changes += 1
        last = s
    return changes


def get_stability_label(diff, eps=0.05):
    sign_changes = count_sign_changes(diff, eps=eps)
    volatility = float(np.sum(np.abs(diff)))

    if sign_changes <= 1 and volatility <= 0.3:
        return "stable"
    elif sign_changes <= 3:
        return "mildly_oscillating"
    else:
        return "highly_oscillating"


def get_first_reach_best_k(ks, y, eps=1e-8):
    max_y = np.max(y)
    for k, v in zip(ks, y):
        if abs(v - max_y) <= eps:
            return k
    return ks[int(np.argmax(y))]


def get_plateau_length_from_first_best(y, eps=0.05):
    max_y = np.max(y)
    first_idx = int(np.argmax(y == max_y)) if np.any(y == max_y) else int(np.argmax(y))
    count = 0
    for i in range(first_idx, len(y)):
        if y[i] >= max_y - eps:
            count += 1
        else:
            break
    return count


def get_post_peak_drop(y, best_idx):
    if best_idx == len(y) - 1:
        return 0.0
    tail_min = float(np.min(y[best_idx + 1:]))
    return float(y[best_idx] - tail_min)


def classify_trend(y, eps=0.05):
    """
    返回更关注“主趋势”的标签：
    - flat
    - monotone_up
    - rise_then_plateau
    - monotone_down
    - fall_then_plateau
    - rise_then_fall
    - oscillating
    - other
    """
    diff = np.diff(y)

    total_range = float(np.max(y) - np.min(y))
    if total_range <= eps:
        return "flat"

    # 差分符号
    pos = np.sum(diff > eps)
    neg = np.sum(diff < -eps)
    sign_changes = count_sign_changes(diff, eps=eps)

    # 基本单调上升：没有明显下降
    if np.all(diff >= -eps):
        # 后半段是否基本平台
        tail = diff[-5:] if len(diff) >= 5 else diff
        if np.sum(np.abs(tail) <= eps) >= max(1, len(tail) - 1):
            return "rise_then_plateau"
        return "monotone_up"

    # 基本单调下降：没有明显上升
    if np.all(diff <= eps):
        tail = diff[-5:] if len(diff) >= 5 else diff
        if np.sum(np.abs(tail) <= eps) >= max(1, len(tail) - 1):
            return "fall_then_plateau"
        return "monotone_down"

    best_idx = int(np.argmax(y))
    before = y[:best_idx + 1]
    after = y[best_idx:]

    has_clear_rise_before_peak = (len(before) >= 2 and (np.max(before) - np.min(before) > eps))
    has_clear_drop_after_peak = (len(after) >= 2 and (y[best_idx] - np.min(after[1:]) > eps if len(after) > 1 else False))

    # 单峰主趋势
    if has_clear_rise_before_peak and has_clear_drop_after_peak:
        if sign_changes <= 2:
            # 再细分：峰后是否先平台再掉
            peak_val = y[best_idx]
            plateau_count = 0
            for i in range(best_idx + 1, len(y)):
                if abs(y[i] - peak_val) <= eps:
                    plateau_count += 1
                else:
                    break
            if plateau_count >= 2:
                return "rise_then_plateau_then_fall"
            return "rise_then_fall"

    if sign_changes >= 3:
        return "oscillating"

    # 常见情况：先升后平台，但夹杂局部噪声
    first_half_gain = np.max(y[: len(y)//2 + 1]) - y[0]
    late_close_to_best = np.mean(y[len(y)//2:]) >= np.max(y) - 2 * eps
    if first_half_gain > eps and late_close_to_best:
        return "rise_then_plateau"

    return "other"


def compute_features(sample):
    ks, y = extract_curve(sample, metric="f1")
    diff = np.diff(y)

    best_idx = int(np.argmax(y))
    best_k = ks[best_idx]
    max_f1 = float(np.max(y))
    f1_at_0 = float(y[0])

    first_reach_best_k = get_first_reach_best_k(ks, y)
    plateau_length = get_plateau_length_from_first_best(y, eps=EPS)
    post_peak_drop = get_post_peak_drop(y, best_idx)
    sign_changes = count_sign_changes(diff, eps=EPS)
    volatility = float(np.sum(np.abs(diff)))
    trend = classify_trend(y, eps=EPS)
    stability = get_stability_label(diff, eps=EPS)

    prefix_best = np.maximum.accumulate(y)

    return {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "dataset_name": sample.get("dataset_name"),
        "best_k": best_k,
        "max_f1": max_f1,
        "f1_at_0": f1_at_0,
        "gain_to_best": max_f1 - f1_at_0,
        "first_reach_best_k": first_reach_best_k,
        "plateau_length": plateau_length,
        "post_peak_drop": post_peak_drop,
        "sign_changes": sign_changes,
        "volatility": volatility,
        "trend_type": trend,
        "stability_type": stability,
        "curve": y.tolist(),
        "prefix_best_curve": prefix_best.tolist(),
    }


# =========================
# 画图
# =========================
def plot_all_curves(results, ks, save_path):
    plt.figure(figsize=(10, 6))
    for r in results:
        y = np.array(r["curve"], dtype=float)
        plt.plot(ks, y, alpha=0.12, linewidth=1)

    mean_curve = np.mean(np.array([r["curve"] for r in results]), axis=0)
    plt.plot(ks, mean_curve, linewidth=3, label="Mean F1")

    plt.xlabel("top-k")
    plt.ylabel("F1")
    plt.title("All Question Curves (F1 vs top-k)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_mean_curves_by_type(results, ks, save_path):
    groups = defaultdict(list)
    for r in results:
        groups[r["trend_type"]].append(np.array(r["curve"], dtype=float))

    plt.figure(figsize=(10, 6))
    for trend_type, curves in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        arr = np.stack(curves, axis=0)
        mean_curve = arr.mean(axis=0)
        plt.plot(ks, mean_curve, linewidth=2, label=f"{trend_type} (n={len(curves)})")

    plt.xlabel("top-k")
    plt.ylabel("Mean F1")
    plt.title("Mean Curve by Trend Type")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_prefix_best_mean_curves_by_type(results, ks, save_path):
    groups = defaultdict(list)
    for r in results:
        groups[r["trend_type"]].append(np.array(r["prefix_best_curve"], dtype=float))

    plt.figure(figsize=(10, 6))
    for trend_type, curves in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        arr = np.stack(curves, axis=0)
        mean_curve = arr.mean(axis=0)
        plt.plot(ks, mean_curve, linewidth=2, label=f"{trend_type} (n={len(curves)})")

    plt.xlabel("top-k")
    plt.ylabel("Mean Prefix-Best F1")
    plt.title("Mean Prefix-Best Curve by Trend Type")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_best_k_distribution(results, save_path):
    best_ks = [r["best_k"] for r in results]
    counts = Counter(best_ks)
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]

    plt.figure(figsize=(9, 5))
    plt.bar(xs, ys)
    plt.xlabel("best_k")
    plt.ylabel("Count")
    plt.title("Distribution of best_k")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_trend_distribution(results, save_path):
    counts = Counter([r["trend_type"] for r in results])
    labels = list(counts.keys())
    values = list(counts.values())

    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Trend Type Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_dataset_trend_stacked(results, save_path):
    datasets = sorted(set(r["dataset_name"] for r in results if r["dataset_name"] is not None))
    trend_types = sorted(set(r["trend_type"] for r in results))

    if len(datasets) == 0:
        return

    data = {ds: Counter() for ds in datasets}
    for r in results:
        ds = r["dataset_name"]
        if ds is not None:
            data[ds][r["trend_type"]] += 1

    bottom = np.zeros(len(datasets))
    plt.figure(figsize=(10, 6))
    for trend in trend_types:
        vals = np.array([data[ds][trend] for ds in datasets], dtype=float)
        plt.bar(datasets, vals, bottom=bottom, label=trend)
        bottom += vals

    plt.ylabel("Count")
    plt.title("Trend Type Distribution by Dataset")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def export_trend_type_examples(samples, results, output_dir, limit=5):
    examples_dir = os.path.join(output_dir, "trend_type_examples")
    ensure_dir(examples_dir)

    grouped_examples = defaultdict(list)
    for sample, result in zip(samples, results):
        trend_type = result["trend_type"]
        if len(grouped_examples[trend_type]) >= limit:
            continue

        example = dict(sample)
        example["curve_analysis"] = {
            "trend_type": result["trend_type"],
            "stability_type": result["stability_type"],
            "best_k": result["best_k"],
            "max_f1": result["max_f1"],
            "f1_at_0": result["f1_at_0"],
            "gain_to_best": result["gain_to_best"],
            "first_reach_best_k": result["first_reach_best_k"],
            "plateau_length": result["plateau_length"],
            "post_peak_drop": result["post_peak_drop"],
            "sign_changes": result["sign_changes"],
            "volatility": result["volatility"],
            "trend_curve": result["curve"],
            "prefix_best_curve": result["prefix_best_curve"],
        }
        grouped_examples[trend_type].append(example)

    export_payload = {trend_type: grouped_examples[trend_type] for trend_type in sorted(grouped_examples.keys())}

    with open(os.path.join(output_dir, "trend_type_examples.json"), "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)

    for trend_type, examples in export_payload.items():
        trend_path = os.path.join(examples_dir, f"{sanitize_filename(trend_type)}_examples.json")
        with open(trend_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)


# =========================
# 主函数
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    data = load_json(INPUT_PATH)
    if not isinstance(data, list):
        raise ValueError(f"期望 {INPUT_PATH} 是一个 list[dict] 的 JSON 文件。")

    if len(data) == 0:
        raise ValueError("输入文件为空。")

    # 取第一个样本确定 k 轴
    ks, _ = extract_curve(data[0], metric="f1")

    results = []
    valid_samples = []
    for sample in data:
        try:
            result = compute_features(sample)
            results.append(result)
            valid_samples.append(sample)
        except Exception as e:
            print(f"[WARN] 跳过样本 {sample.get('id', 'UNKNOWN')}，原因: {e}")

    if len(results) == 0:
        raise ValueError("没有成功解析任何样本。")

    # 导出明细表
    df = pd.DataFrame([
        {
            "id": r["id"],
            "question": r["question"],
            "dataset_name": r["dataset_name"],
            "best_k": r["best_k"],
            "max_f1": r["max_f1"],
            "f1_at_0": r["f1_at_0"],
            "gain_to_best": r["gain_to_best"],
            "first_reach_best_k": r["first_reach_best_k"],
            "plateau_length": r["plateau_length"],
            "post_peak_drop": r["post_peak_drop"],
            "sign_changes": r["sign_changes"],
            "volatility": r["volatility"],
            "trend_type": r["trend_type"],
            "stability_type": r["stability_type"],
        }
        for r in results
    ])
    df.to_csv(os.path.join(OUTPUT_DIR, "per_question_curve_analysis.csv"), index=False, encoding="utf-8-sig")

    # 导出带原始曲线的 json
    with open(os.path.join(OUTPUT_DIR, "per_question_curve_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 导出每种趋势类型的实例（每类最多 5 个）
    export_trend_type_examples(
        valid_samples,
        results,
        OUTPUT_DIR,
        limit=EXAMPLES_PER_TREND_TYPE,
    )

    # 汇总统计
    trend_counter = Counter(r["trend_type"] for r in results)
    stability_counter = Counter(r["stability_type"] for r in results)
    dataset_counter = Counter(r["dataset_name"] for r in results)

    summary = {
        "num_questions": len(results),
        "datasets": dict(dataset_counter),
        "trend_type_distribution": dict(trend_counter),
        "stability_type_distribution": dict(stability_counter),
        "avg_best_k": float(df["best_k"].mean()),
        "avg_max_f1": float(df["max_f1"].mean()),
        "avg_f1_at_0": float(df["f1_at_0"].mean()),
        "avg_gain_to_best": float(df["gain_to_best"].mean()),
        "avg_post_peak_drop": float(df["post_peak_drop"].mean()),
        "avg_sign_changes": float(df["sign_changes"].mean()),
        "avg_volatility": float(df["volatility"].mean()),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 画图
    plot_all_curves(results, ks, os.path.join(OUTPUT_DIR, "all_curves.png"))
    plot_mean_curves_by_type(results, ks, os.path.join(OUTPUT_DIR, "mean_curves_by_trend_type.png"))
    plot_prefix_best_mean_curves_by_type(results, ks, os.path.join(OUTPUT_DIR, "prefix_best_mean_curves_by_trend_type.png"))
    plot_best_k_distribution(results, os.path.join(OUTPUT_DIR, "best_k_distribution.png"))
    plot_trend_distribution(results, os.path.join(OUTPUT_DIR, "trend_type_distribution.png"))
    plot_dataset_trend_stacked(results, os.path.join(OUTPUT_DIR, "dataset_trend_distribution.png"))

    # 打印简要结果
    print("=" * 80)
    print(f"总问题数: {len(results)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("-" * 80)
    print("趋势类型分布:")
    for k, v in trend_counter.most_common():
        print(f"  {k}: {v}")
    print("-" * 80)
    print("稳定性类型分布:")
    for k, v in stability_counter.most_common():
        print(f"  {k}: {v}")
    print("-" * 80)
    print(f"平均 best_k: {summary['avg_best_k']:.4f}")
    print(f"平均 max_f1: {summary['avg_max_f1']:.4f}")
    print(f"平均 f1@0: {summary['avg_f1_at_0']:.4f}")
    print(f"平均 gain_to_best: {summary['avg_gain_to_best']:.4f}")
    print(f"平均 post_peak_drop: {summary['avg_post_peak_drop']:.4f}")
    print(f"平均 sign_changes: {summary['avg_sign_changes']:.4f}")
    print(f"平均 volatility: {summary['avg_volatility']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
