#!/usr/bin/env python3
"""Analyze the fixed-k HotpotQA 500+500 ONER sweep."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make local project modules importable when invoked as
# ``python analysis/analyze_hotpot_500_500.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
from transformers import PreTrainedTokenizerFast

from official_evaluation.hotpotqa.hotpot_evaluate_v1 import exact_match_score, f1_score
from reconstruction.reconstruct_oner import para_to_text, remove_wh_words


EXPERIMENT_PATTERN = re.compile(r"bm25_retrieval_count__(\d+)___")
GROUPS = ("bridge", "comparison", "combined")


def percentile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution_statistics(best_ks):
    counts = Counter(best_ks)
    total = len(best_ks)
    probabilities = [count / total for count in counts.values()] if total else []
    return {
        "count": total,
        "distribution": {str(k): counts.get(k, 0) for k in range(1, 16)},
        "majority_ratio": max(probabilities) if probabilities else None,
        "entropy_bits": -sum(p * math.log2(p) for p in probabilities),
        "mean_best_k": statistics.mean(best_ks) if best_ks else None,
        "median_best_k": statistics.median(best_ks) if best_ks else None,
        "std_best_k": statistics.pstdev(best_ks) if best_ks else None,
    }


def find_prediction_files(predictions_dir, k_min=1, k_max=15):
    result = {}
    for directory in Path(predictions_dir).glob("oner_qa_qwen_hotpotqa*"):
        match = EXPERIMENT_PATTERN.search(directory.name)
        if not match:
            continue
        k = int(match.group(1))
        if not k_min <= k <= k_max:
            continue
        matches = list(directory.glob("prediction__hotpotqa_to_hotpotqa__dev_1000_subsampled.json"))
        if len(matches) == 1:
            if k in result:
                raise ValueError("Multiple prediction directories for k={}".format(k))
            result[k] = matches[0]
    missing = sorted(set(range(k_min, k_max + 1)) - set(result))
    if missing:
        raise FileNotFoundError("Missing prediction files for k={}".format(missing))
    return result


def load_samples(path):
    samples = {}
    with Path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            sample = json.loads(line)
            qid = sample["question_id"]
            if qid in samples:
                raise ValueError("Duplicate question_id: " + qid)
            samples[qid] = sample
    return samples


def gold_answers(sample):
    answers = []
    for answer_object in sample["answers_objects"]:
        answers.extend(str(span) for span in answer_object.get("spans", []))
    return answers


def score_predictions(samples, prediction_files):
    results = {qid: {"type": sample["type"], "scores": {}} for qid, sample in samples.items()}
    for k, path in sorted(prediction_files.items()):
        predictions = json.loads(path.read_text(encoding="utf-8"))
        if set(predictions) != set(samples):
            raise ValueError("Prediction IDs do not match dataset at k={}".format(k))
        for qid, sample in samples.items():
            prediction = predictions[qid]
            if isinstance(prediction, list):
                prediction = " ".join(str(answer) for answer in prediction)
            if not isinstance(prediction, str):
                raise ValueError("Invalid prediction for " + qid)
            gold = gold_answers(sample)
            if len(gold) != 1:
                raise ValueError("Expected one official HotpotQA gold answer for " + qid)
            results[qid]["scores"][k] = {
                "em": int(exact_match_score(prediction, gold[0])),
                "f1": f1_score(prediction, gold[0])[0],
                "prediction": prediction,
            }
    for item in results.values():
        item["best_k"] = max(item["scores"], key=lambda k: (item["scores"][k]["f1"], -k))
        best_f1 = item["scores"][item["best_k"]]["f1"]
        item["tied_best_ks"] = sorted(k for k in item["scores"] if item["scores"][k]["f1"] == best_f1)
    return results


def load_api_tokens(samples, prediction_files):
    result = {qid: {} for qid in samples}
    for k, prediction_file in sorted(prediction_files.items()):
        path = prediction_file.parent / "token_usage.jsonl"
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            qid = row["qid"]
            if qid not in samples or qid in seen or row["k"] != k:
                raise ValueError("Foreign or duplicate token record in " + str(path))
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if not isinstance(row[field], int) or row[field] < 0:
                    raise ValueError("Invalid API usage")
            if row["prompt_cropped"] or row["finish_reason"] != "stop":
                raise ValueError("Invalid generation in " + str(path))
            seen.add(qid)
            result[qid][k] = row
        if seen != set(samples):
            raise ValueError("Token IDs do not match dataset at k={}".format(k))
    return result


def count_saved_context_tokens(samples, prediction_files, api_tokens, tokenizer):
    """Reuse count_context_tokens on the actual selected contexts for each k."""
    counts = {qid: {} for qid in samples}
    for k, path in sorted(prediction_files.items()):
        retrievals = {}
        for qid in samples:
            row = json.loads((path.parent / "checkpoints" / (qid + ".json")).read_text())
            if row["qid"] != qid or row["k"] != k:
                raise ValueError("Foreign context checkpoint")
            documents = row["contexts"]
            context = "\n\n".join(para_to_text(d["title"], d["paragraph_text"], 350) for d in documents)
            digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
            if digest != api_tokens[qid][k]["context_sha256"]:
                raise ValueError("Saved context differs from model context")
            if row["result"][1] != api_tokens[qid][k]["prediction"]:
                raise ValueError("Checkpoint/token prediction mismatch")
            retrievals[qid] = documents
        current = count_context_tokens(retrievals, tokenizer, ks=(k,))
        for qid in samples:
            counts[qid][k] = current[qid][k]
    return counts


def retrieve_contexts(samples, endpoint, workers):
    def retrieve(item):
        qid, sample = item
        payload = {
            "retrieval_method": "retrieve_from_elasticsearch",
            "query_text": remove_wh_words(sample["question_text"]),
            "max_hits_count": 15,
            "corpus_name": "hotpotqa",
            "document_type": "title_paragraph_text",
        }
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        return qid, response.json()["retrieval"]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(retrieve, samples.items()))


def count_context_tokens(retrievals, tokenizer, ks=range(1, 16)):
    counts = {qid: {} for qid in retrievals}
    for qid, documents in retrievals.items():
        for k in ks:
            context = "\n\n".join(
                para_to_text(document["title"], document["paragraph_text"], 350)
                for document in documents[:k]
            )
            counts[qid][k] = len(tokenizer.encode(context, add_special_tokens=False))
    return counts


def selected_ids(results, group):
    return [qid for qid, item in results.items() if group == "combined" or item["type"] == group]


def build_summaries(results, token_counts, api_tokens=None):
    fixed_rows = []
    oracle = {}
    for group in GROUPS:
        ids = selected_ids(results, group)
        oracle[group] = distribution_statistics([results[qid]["best_k"] for qid in ids])
        best_values = [results[qid]["scores"][results[qid]["best_k"]]["f1"] for qid in ids]
        tied_counts = [len(results[qid]["tied_best_ks"]) for qid in ids]
        oracle[group].update({
            "all_zero_f1_ratio": sum(value == 0 for value in best_values) / len(ids),
            "tied_best_k_ratio": sum(value > 1 for value in tied_counts) / len(ids),
            "tied_best_k_count_distribution": dict(sorted(Counter(tied_counts).items())),
            "oracle_average_f1": statistics.mean(best_values),
        })
        for k in range(1, 16):
            f1_values = [results[qid]["scores"][k]["f1"] for qid in ids]
            tokens = [token_counts[qid][k] for qid in ids]
            fixed_rows.append(
                {
                    "group": group,
                    "k": k,
                    "count": len(ids),
                    "average_f1": statistics.mean(f1_values),
                    "average_em": statistics.mean(results[qid]["scores"][k]["em"] for qid in ids),
                    "average_context_tokens": statistics.mean(tokens),
                    "median_context_tokens": statistics.median(tokens),
                    "p90_context_tokens": percentile(tokens, 0.90),
                    "p95_context_tokens": percentile(tokens, 0.95),
                }
            )
            if api_tokens is not None:
                for metric in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds"):
                    values = [api_tokens[qid][k][metric] for qid in ids]
                    fixed_rows[-1].update({
                        "average_" + metric: statistics.mean(values),
                        "median_" + metric: statistics.median(values),
                        "p90_" + metric: percentile(values, 0.90),
                        "p95_" + metric: percentile(values, 0.95),
                        "min_" + metric: min(values),
                        "max_" + metric: max(values),
                    })
        best_fixed = max((row for row in fixed_rows if row["group"] == group), key=lambda row: (row["average_f1"], -row["k"]))
        oracle[group].update(best_fixed_k=best_fixed["k"], best_fixed_average_f1=best_fixed["average_f1"],
            oracle_fixed_f1_gap=oracle[group]["oracle_average_f1"] - best_fixed["average_f1"])
    return fixed_rows, oracle


def write_outputs(output_dir, results, token_counts, fixed_rows, oracle, api_tokens=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fixed_k_results.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(fixed_rows[0]))
        writer.writeheader()
        writer.writerows(fixed_rows)

    with (output_dir / "oracle_distribution.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["group", "k", "count", "proportion"])
        writer.writeheader()
        for group in GROUPS:
            total = oracle[group]["count"]
            for k in range(1, 16):
                count = oracle[group]["distribution"][str(k)]
                writer.writerow({"group": group, "k": k, "count": count, "proportion": count / total})

    (output_dir / "hotpot_basic_statistics.json").write_text(
        json.dumps({"scoring": "official HotpotQA", "oracle_rule": "max F1; smallest k on ties", "oracle": oracle, "fixed_k": fixed_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "per_question_curves.jsonl").open("w", encoding="utf-8") as output_file:
        for qid, item in results.items():
            row = {"question_id": qid, **item, "context_tokens": token_counts[qid]}
            if api_tokens is not None:
                row["api_token_usage"] = {k: {key: value for key, value in usage.items() if key != "raw_generation"} for k, usage in api_tokens[qid].items()}
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_figures(output_dir, fixed_rows, oracle):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for all analysis artifacts") from error

    for metric, filename, ylabel in (
        ("average_f1", "fixed_k_f1.png", "Average F1"),
        ("average_context_tokens", "token_curve.png", "Average context tokens"),
    ):
        for group in GROUPS:
            rows = [row for row in fixed_rows if row["group"] == group]
            plt.plot([row["k"] for row in rows], [row[metric] for row in rows], marker="o", label=group)
        plt.xlabel("k")
        plt.ylabel(ylabel)
        plt.xticks(range(1, 16))
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    for group in GROUPS:
        counts = oracle[group]["distribution"]
        plt.bar(range(1, 16), [counts[str(k)] for k in range(1, 16)])
        plt.xlabel("Oracle best k")
        plt.ylabel("Question count")
        plt.xticks(range(1, 16))
        plt.tight_layout()
        plt.savefig(output_dir / (group + "_oracle_distribution.png"), dpi=180)
        plt.close()


def write_summary(output_dir, oracle, fixed_rows):
    combined_rows = [row for row in fixed_rows if row["group"] == "combined"]
    best_fixed = max(combined_rows, key=lambda row: row["average_f1"])
    lines = [
        "# HotpotQA 500 bridge + 500 comparison 分析",
        "",
        "- 样本：bridge 500，comparison 500，共 1000。",
        "- k 范围：1–15；Oracle 并列时选择较小 k。",
        "- F1/EM：官方 HotpotQA；Oracle 首先最大化官方 F1，EM 不参与选择。",
        "- context_tokens 使用正式运行保存的实际检索上下文，与 API prompt_tokens 分开统计。",
        "- 历史 DROP F1 与本轮官方 F1 不能直接比较；历史产物未覆盖。",
        "- combined 最优 fixed-k：k={}，平均 F1={:.6f}。".format(best_fixed["k"], best_fixed["average_f1"]),
        "",
        "| group | majority ratio | entropy (bits) | mean best-k | median | std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = oracle[group]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:.4f} |".format(
                group, item["majority_ratio"], item["entropy_bits"], item["mean_best_k"],
                item["median_best_k"], item["std_best_k"]
            )
        )
    lines.extend(["", "| group | Oracle F1 | best fixed k | fixed F1 | gap | zero F1 ratio | ties ratio |", "|---|---:|---:|---:|---:|---:|---:|"])
    for group in GROUPS:
        item = oracle[group]
        lines.append("| {} | {:.6f} | {} | {:.6f} | {:.6f} | {:.4f} | {:.4f} |".format(group, item["oracle_average_f1"], item["best_fixed_k"], item["best_fixed_average_f1"], item["oracle_fixed_f1_gap"], item["all_zero_f1_ratio"], item["tied_best_k_ratio"]))
        lines.append("\n{} 并列最优 k 数量分布：`{}`\n".format(group, json.dumps(item["tied_best_k_count_distribution"], ensure_ascii=False)))
    lines.extend(["", "| group | k | F1 | EM | context tokens | API prompt | completion | total | seconds |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in fixed_rows:
        lines.append("| {} | {} | {:.6f} | {:.6f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.3f} |".format(row["group"], row["k"], row["average_f1"], row["average_em"], row["average_context_tokens"], row.get("average_prompt_tokens", 0), row.get("average_completion_tokens", 0), row.get("average_total_tokens", 0), row.get("average_latency_seconds", 0)))
    (output_dir / "hotpot_500_500_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="processed_data/hotpotqa/dev_1000_subsampled.jsonl")
    parser.add_argument("--predictions-dir", default="predictions/dev_1000")
    parser.add_argument("--output-dir", required=True, help="New directory; historical reports must not be overwritten")
    parser.add_argument("--retriever-endpoint", default="http://127.0.0.1:8000/retrieve/")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    samples = load_samples(args.data)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new analysis output directory: " + str(output_dir))
    if Counter(sample["type"] for sample in samples.values()) != Counter({"bridge": 500, "comparison": 500}):
        raise ValueError("Expected exactly 500 bridge and 500 comparison samples")
    prediction_files = find_prediction_files(args.predictions_dir)
    results = score_predictions(samples, prediction_files)
    api_tokens = load_api_tokens(samples, prediction_files)
    for qid, item in results.items():
        for k, score in item["scores"].items():
            if score["prediction"] != api_tokens[qid][k]["prediction"]:
                raise ValueError("Token/prediction mismatch")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    token_counts = count_saved_context_tokens(samples, prediction_files, api_tokens, tokenizer)
    fixed_rows, oracle = build_summaries(results, token_counts, api_tokens)
    write_outputs(output_dir, results, token_counts, fixed_rows, oracle, api_tokens)
    write_figures(output_dir, fixed_rows, oracle)
    write_summary(output_dir, oracle, fixed_rows)
    print(json.dumps(oracle, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
