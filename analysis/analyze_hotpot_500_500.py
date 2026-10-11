#!/usr/bin/env python3
"""Analyze the fixed-k HotpotQA 500+500 ONER sweep."""

import argparse
import csv
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

from evaluate_per_question import prepare_prediction_answers, score_per_question
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
            answers = prepare_prediction_answers(predictions[qid])
            score = score_per_question("hotpotqa", answers, gold_answers(sample))
            results[qid]["scores"][k] = {
                "em": score["em"],
                "f1": score["f1"],
                "prediction": answers[0],
            }
    for item in results.values():
        item["best_k"] = max(item["scores"], key=lambda k: (item["scores"][k]["em"], item["scores"][k]["f1"], -k))
    return results


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


def count_context_tokens(retrievals, tokenizer):
    counts = {qid: {} for qid in retrievals}
    for qid, documents in retrievals.items():
        for k in range(1, 16):
            context = "\n\n".join(
                para_to_text(document["title"], document["paragraph_text"], 350)
                for document in documents[:k]
            )
            counts[qid][k] = len(tokenizer.encode(context, add_special_tokens=False))
    return counts


def selected_ids(results, group):
    return [qid for qid, item in results.items() if group == "combined" or item["type"] == group]


def build_summaries(results, token_counts):
    fixed_rows = []
    oracle = {}
    for group in GROUPS:
        ids = selected_ids(results, group)
        oracle[group] = distribution_statistics([results[qid]["best_k"] for qid in ids])
        for k in range(1, 16):
            f1_values = [results[qid]["scores"][k]["f1"] for qid in ids]
            tokens = [token_counts[qid][k] for qid in ids]
            fixed_rows.append(
                {
                    "group": group,
                    "k": k,
                    "count": len(ids),
                    "average_f1": statistics.mean(f1_values),
                    "average_context_tokens": statistics.mean(tokens),
                    "median_context_tokens": statistics.median(tokens),
                    "p90_context_tokens": percentile(tokens, 0.90),
                    "p95_context_tokens": percentile(tokens, 0.95),
                }
            )
    return fixed_rows, oracle


def write_outputs(output_dir, results, token_counts, fixed_rows, oracle):
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
        json.dumps({"oracle": oracle, "fixed_k": fixed_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "per_question_curves.jsonl").open("w", encoding="utf-8") as output_file:
        for qid, item in results.items():
            row = {"question_id": qid, **item, "context_tokens": token_counts[qid]}
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_figures(output_dir, fixed_rows, oracle):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping PNG figures.")
        return

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
    (output_dir / "hotpot_500_500_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="processed_data/hotpotqa/dev_1000_subsampled.jsonl")
    parser.add_argument("--predictions-dir", default="predictions/dev_1000")
    parser.add_argument("--output-dir", default="analysis/hotpot_500_500")
    parser.add_argument("--retriever-endpoint", default="http://127.0.0.1:8000/retrieve/")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    samples = load_samples(args.data)
    if Counter(sample["type"] for sample in samples.values()) != Counter({"bridge": 500, "comparison": 500}):
        raise ValueError("Expected exactly 500 bridge and 500 comparison samples")
    prediction_files = find_prediction_files(args.predictions_dir)
    results = score_predictions(samples, prediction_files)
    retrievals = retrieve_contexts(samples, args.retriever_endpoint, args.workers)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    token_counts = count_context_tokens(retrievals, tokenizer)
    fixed_rows, oracle = build_summaries(results, token_counts)
    output_dir = Path(args.output_dir)
    write_outputs(output_dir, results, token_counts, fixed_rows, oracle)
    write_figures(output_dir, fixed_rows, oracle)
    write_summary(output_dir, oracle, fixed_rows)
    print(json.dumps(oracle, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
