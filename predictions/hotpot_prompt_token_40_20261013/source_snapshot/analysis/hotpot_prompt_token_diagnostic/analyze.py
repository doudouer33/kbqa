"""Paired offline diagnostics with the unchanged repository and official scorers."""
import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from evaluate_per_question import prepare_prediction_answers, score_per_question
from official_evaluation.hotpotqa.hotpot_evaluate_v1 import f1_score, exact_match_score
from analysis.hotpot_stop_diagnostic.analyze import official_answer


def score(prediction, gold):
    evaluated = prepare_prediction_answers(prediction)
    repository = score_per_question("hotpotqa", evaluated, gold)
    return {"repository_f1": repository["f1"], "repository_em": repository["em"],
            "official_f1": max(f1_score(official_answer(prediction), span)[0] for span in gold),
            "official_em": max(float(exact_match_score(official_answer(prediction), span)) for span in gold)}


def observations(log, gold):
    response = log["api_responses"][0]["body"]
    choice = response["choices"][0]
    raw = choice["message"]["content"]
    # Observe the last marker rather than letting the diagnostic capture span
    # several markers. The production extractor remains completely unchanged.
    suffix = re.search(r"\bSo the answer is: ((?:(?!So the answer is:)[^\r\n])+)\.$", raw)
    answer = suffix.group(1) if suffix else None
    yes_no = gold[0].lower() in ("yes", "no")
    has_newline = "\n" in raw or "\r" in raw
    # Syntax/yes-no diagnostics only. Semantic correctness and entity verbosity
    # are reviewed separately; this regex never changes the saved prediction.
    syntax_ok = bool(suffix) and not has_newline
    unique_final_marker = raw.count("So the answer is:") == 1
    yes_no_pure = str(log["final_answer"]) in ("yes", "no") if yes_no else None
    format_ok = syntax_ok and (not yes_no or answer in ("yes", "no"))
    return {"raw": raw, "raw_has_newline": has_newline, "exact_marker": "So the answer is:" in raw,
            "primary_match": log["extractor"]["primary_match"], "regex_capture": log["extractor"]["matched_group"],
            "observed_suffix_answer": answer, "syntax_ok": syntax_ok, "format_ok": format_ok,
            "unique_final_marker": unique_final_marker, "single_marker_format_ok": format_ok and unique_final_marker,
            "reasoning_before_answer": bool(suffix and raw[:suffix.start()].strip()), "yes_no_pure": yes_no_pure,
            "extraction_matches_suffix": str(log["final_answer"]) == answer if suffix else None,
            "prediction": log["final_answer"], "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"], "finish_reason": choice["finish_reason"],
            "stop_reason": choice.get("stop_reason"), "runtime_cropped": log["prompt_fit"]["changed"],
            "fewshot_count": log["prompt_fit"]["after_demonstrations"],
            "test_changed": log["prompt_fit"]["test_example_before"].strip() != log["prompt_fit"]["test_example_after"].strip(),
            "json_escape_pollution": bool(re.search(r'\\(?:u[0-9a-fA-F]{4}|n|r|t|\")', str(log["final_answer"]))),
            "raw_preserved": log["generator_outputs"][0][0] == raw and json.loads(log["extractor"]["serialized_input"]) == raw.strip()}


def summarize(rows, group, k):
    result = {"group": group, "k": k, "count": len(rows)}
    for prefix in ("old", "new"):
        metrics = {}
        for name in ("official_f1", "official_em", "repository_f1", "repository_em"):
            metrics[name] = statistics.mean(row[prefix + "_" + name] for row in rows)
        for name in ("raw_has_newline", "exact_marker", "primary_match", "syntax_ok", "format_ok", "single_marker_format_ok", "reasoning_before_answer",
                     "runtime_cropped", "test_changed", "json_escape_pollution"):
            metrics[name + "_count"] = sum(bool(row[prefix + "_" + name]) for row in rows)
        metrics["yes_no_count"] = sum(row["yes_no"] for row in rows)
        metrics["yes_no_pure_count"] = sum(row[prefix + "_yes_no_pure"] is True for row in rows)
        metrics["yes_no_correct_count"] = sum(row["yes_no"] and row[prefix + "_official_em"] == 1 for row in rows)
        metrics["extraction_mismatch_count"] = sum(row[prefix + "_extraction_matches_suffix"] is False for row in rows)
        metrics["fewshot_counts"] = dict(Counter(row[prefix + "_fewshot_count"] for row in rows))
        metrics["finish_reasons"] = dict(Counter(row[prefix + "_finish_reason"] for row in rows))
        for name in ("prompt_tokens", "completion_tokens"):
            values = [row[prefix + "_" + name] for row in rows]
            metrics[name] = {"min": min(values), "max": max(values), "mean": statistics.mean(values), "sum": sum(values)}
        result[prefix] = metrics
    result["official_f1_improved"] = sum(row["new_official_f1"] > row["old_official_f1"] for row in rows)
    result["official_f1_decreased"] = sum(row["new_official_f1"] < row["old_official_f1"] for row in rows)
    result["official_f1_unchanged"] = sum(row["new_official_f1"] == row["old_official_f1"] for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    baseline = Path(manifest["baseline_dir"])
    preflight = json.loads((output / "preflight.json").read_text())
    old_logs = {(row["k"], row["qid"]): row for row in map(json.loads, (baseline / "diagnostics.jsonl").read_text().splitlines())}
    logs = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
    samples = {row["question_id"]: row for row in map(json.loads, (output / "samples.jsonl").read_text().splitlines())}
    assert (output / "samples.jsonl").read_bytes() == (baseline / "samples.jsonl").read_bytes()
    assert len(logs) == len(old_logs) == 120 and len(samples) == 40
    assert [(log["k"], log["qid"]) for log in logs] == [(k, qid) for k in (1, 8, 15) for qid in manifest["question_ids"]]
    saved_metrics = []
    rows = []
    for log in logs:
        sample = samples[log["qid"]]
        old = old_logs[(log["k"], log["qid"])]
        gold = [span for answer in sample["answers_objects"] for span in answer["spans"]]
        row = {"k": log["k"], "qid": log["qid"], "type": sample["type"], "question": sample["question_text"], "gold": gold,
               "yes_no": gold[0].lower() in ("yes", "no"), "fewshot_ids": log["fewshot_ids"],
               "instructions_present": log["instructions_present"], "initialization_cropped": log["initialization_cropped"]}
        for prefix, record in (("old", old), ("new", log)):
            row.update({prefix + "_" + key: value for key, value in observations(record, gold).items()})
            row.update({prefix + "_" + key: value for key, value in score(record["final_answer"], gold).items()})
        row["retrieval_identical"] = log["retrieval_calls"][0]["response"]["retrieval"] == old["retrieval_calls"][0]["response"]["retrieval"]
        row["current_question_context_identical"] = log["prompt_fit"]["test_example_after"] == old["prompt_fit"]["test_example_before"]
        row["token_problem"] = row["new_runtime_cropped"] or row["new_finish_reason"] == "length"
        row["model_format_failure"] = not row["new_format_ok"]
        row["extraction_mismatch"] = row["new_extraction_matches_suffix"] is False
        row["formatted_zero_score"] = row["new_format_ok"] and not row["extraction_mismatch"] and row["new_official_f1"] == 0
        rows.append(row)
        log["scores"] = {"official_f1": row["new_official_f1"], "official_em": row["new_official_em"],
                         "repository_f1": row["new_repository_f1"], "repository_em": row["new_repository_em"]}
        log["output_format"] = {key: row["new_" + key] for key in ("syntax_ok", "format_ok", "observed_suffix_answer", "yes_no_pure", "extraction_matches_suffix")}
    assert all(row["retrieval_identical"] and row["current_question_context_identical"] for row in rows)
    assert all(row["instructions_present"] and not row["initialization_cropped"] and not row["new_runtime_cropped"] and row["new_fewshot_count"] == 15 for row in rows)
    assert all(row["new_raw_preserved"] for row in rows)
    summary = []
    for group in ("combined", "bridge", "comparison", "yes_no", "entity"):
        for k in (1, 8, 15, "all"):
            selected = [row for row in rows if (k == "all" or row["k"] == k) and (
                group == "combined" or row["type"] == group or group == "yes_no" and row["yes_no"] or group == "entity" and not row["yes_no"])]
            summary.append(summarize(selected, group, k))

    for k in (1, 8, 15):
        directory = next(output.glob("*retrieval_count__{}___*".format(k)))
        filename = "prediction__hotpotqa_to_hotpotqa__" + output.name + ".json"
        saved = json.loads((directory / filename).read_text())
        selected = [row for row in rows if row["k"] == k]
        assert saved == {row["qid"]: row["new_prediction"] for row in selected}
        aggregate = next(row for row in summary if row["group"] == "combined" and row["k"] == k)
        official = json.loads(next(directory.glob("official_evaluation_metrics__*.json")).read_text())
        repository = json.loads(next(directory.glob("evaluation_metrics__*.json")).read_text())
        assert abs(official["f1"] - round(aggregate["new"]["official_f1"], 5)) < 1e-10
        assert abs(official["em"] - round(aggregate["new"]["official_em"], 5)) < 1e-10
        # evaluate.py accumulates in set order; at a .0005 boundary floating
        # addition/NumPy rounding can differ by one last saved decimal.
        assert abs(repository["f1"] - float(aggregate["new"]["repository_f1"])) <= 0.0005 + 1e-10
        assert abs(repository["em"] - round(aggregate["new"]["repository_em"], 3)) < 1e-10
        saved_metrics.append({"k": k, "official": official, "repository": repository,
                              "repository_per_question_f1_mean": float(aggregate["new"]["repository_f1"]),
                              "repository_agrees_within_saved_precision": True})

    (output / "diagnostics.jsonl").write_text("".join(json.dumps(log, ensure_ascii=False) + "\n" for log in logs))
    (output / "comparison.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as stream:
        fields = [key for key in rows[0] if key not in ("old_raw", "new_raw")]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    review = ["# 120 条配对原始输出", "", "只展示原始结果；不修改预测、提取器或评分输入。"]
    for row in rows:
        review += ["", "## k={} / {} / {}".format(row["k"], row["type"], row["qid"]), "", row["question"], "",
                   "Gold: " + json.dumps(row["gold"], ensure_ascii=False), "",
                   "首选正则命中：{}→{}；官方 F1：{:.2%}→{:.2%}；EM：{}→{}".format(row["old_primary_match"], row["new_primary_match"], row["old_official_f1"], row["new_official_f1"], row["old_official_em"], row["new_official_em"]),
                   "", "上轮原始输出：", "", "```text", row["old_raw"], "```", "", "本轮原始输出：", "", "```text", row["new_raw"], "```", "",
                   "原正则捕获：" + json.dumps(row["new_regex_capture"], ensure_ascii=False), "",
                   "最终预测：" + json.dumps(row["new_prediction"], ensure_ascii=False), "",
                   "finish_reason={}；stop_reason={}；prompt_tokens={}；completion_tokens={}；fewshot={}".format(row["new_finish_reason"], row["new_stop_reason"], row["new_prompt_tokens"], row["new_completion_tokens"], row["new_fewshot_count"])]
    (output / "raw_output_review.md").write_text("\n".join(review) + "\n")
    validation = dict(manifest["validation"], sample_bytes_identical=True, question_order_identical=True, retrieval_results_identical=True,
                      current_question_context_identical=True, original_15_example_contents_unchanged=True, all_k_example_ids_and_order_identical=True,
                      scores_match_original_evaluators=True, saved_metrics=saved_metrics,
                      server_chat_template_overheads=sorted(set(row["chat_template_overhead"] for row in preflight["requests"])),
                      output_length_truncations=sum(row["new_finish_reason"] == "length" for row in rows))
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([row for row in summary if row["group"] == "combined"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
