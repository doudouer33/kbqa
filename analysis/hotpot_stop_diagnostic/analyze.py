"""Offline comparison only: never generate, alter predictions, or repair extraction."""
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


def old_chains(path):
    text = path.read_text()
    matches = list(re.finditer(r"(?m)^([0-9a-f]{24})\n", text))
    result = {}
    for index, match in enumerate(matches):
        block = text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        answers = [line[3:] for line in block.splitlines() if line.startswith("A: ")]
        assert len(answers) == 3, (match.group(1), answers)
        result[match.group(1)] = {"serialized_cot": answers[1], "raw_cot": json.loads(answers[1]), "extractor_output": answers[2], "retrieved_titles": json.loads(answers[0])}
    return result


def primary_match(serialized):
    query = serialized[1:-1] if serialized.startswith('"') and serialized.endswith('"') else serialized
    return bool(re.match(r".* answer is:? (.*)\.?", query))


def marker(text):
    # Diagnostic observation only; the pipeline does not use this regex.
    match = re.search(r"\banswer is\s*:?\s*(\S[^\n]*)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def official_answer(prediction):
    # Match evaluate.py's official path: flatten lists, with no extra extractor.
    return " ".join(str(item) for item in prediction) if isinstance(prediction, list) else str(prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    samples = {row["question_id"]: row for row in map(json.loads, (output / "samples.jsonl").read_text().splitlines())}
    logs = list(map(json.loads, (output / "diagnostics.jsonl").read_text().splitlines()))
    assert len(logs) == 120 and len(samples) == 40
    rows = []
    for k in (1, 8, 15):
        historical = ROOT / "predictions/dev_1000" / ("oner_qa_qwen_hotpotqa____prompt_set_1___bm25_retrieval_count__{}___distractor_count__1".format(k))
        old_predictions = json.loads(next(historical.glob("prediction__*subsampled.json")).read_text())
        old = old_chains(next(historical.glob("prediction__*chains.txt")))
        new_dir = next(output.glob("*retrieval_count__{}___*".format(k)))
        saved = json.loads(next(new_dir.glob("prediction__*.json")).read_text())
        assert set(saved) == set(samples)
        for log in (item for item in logs if item["k"] == k):
            qid = log["qid"]
            assert saved[qid] == log["final_answer"]
            sample = samples[qid]
            gold = [span for answer in sample["answers_objects"] for span in answer["spans"]]
            old_eval = prepare_prediction_answers(old_predictions[qid])
            new_eval = prepare_prediction_answers(log["final_answer"])
            old_score = score_per_question("hotpotqa", old_eval, gold)
            new_score = score_per_question("hotpotqa", new_eval, gold)
            response = log["api_responses"][0]["body"]
            choice = response["choices"][0]
            raw = choice["message"]["content"]
            fit = log["prompt_fit"]
            row = {
                "k": k, "qid": qid, "type": sample["type"], "question": sample["question_text"], "gold": gold,
                "yes_no": gold[0].lower() in ("yes", "no"),
                "old_prediction": old_predictions[qid], "new_prediction": log["final_answer"],
                "old_evaluated_answer": old_eval, "new_evaluated_answer": new_eval,
                "old_cot": old[qid]["raw_cot"], "new_cot": raw,
                "old_primary_match": primary_match(old[qid]["serialized_cot"]), "new_primary_match": log["extractor"]["primary_match"],
                "old_marker": marker(old[qid]["raw_cot"]), "new_marker": marker(raw),
                "old_f1": old_score["f1"], "new_f1": new_score["f1"], "f1_delta": new_score["f1"] - old_score["f1"],
                "old_em": old_score["em"], "new_em": new_score["em"],
                "old_official_f1": max(f1_score(official_answer(old_predictions[qid]), span)[0] for span in gold),
                "new_official_f1": max(f1_score(official_answer(log["final_answer"]), span)[0] for span in gold),
                "old_official_em": max(float(exact_match_score(official_answer(old_predictions[qid]), span)) for span in gold),
                "new_official_em": max(float(exact_match_score(official_answer(log["final_answer"]), span)) for span in gold),
                "prompt_tokens": response["usage"]["prompt_tokens"], "completion_tokens": response["usage"]["completion_tokens"],
                "finish_reason": choice["finish_reason"], "stop_reason": choice.get("stop_reason"),
                "requested_max_tokens": log["api_requests"][0]["body"]["max_tokens"],
                "raw_has_newline": "\n" in raw,
                "prompt_changed": fit["changed"], "test_example_changed": fit["test_example_before"].strip() != fit["test_example_after"].strip(),
                "prompt_before_tokens": fit["before_tokens"], "prompt_after_tokens": fit["after_tokens"],
                "demonstrations_before": fit["before_demonstrations"], "demonstrations_after": fit["after_demonstrations"],
                "raw_preserved_by_generator": log["generator_outputs"][0][0] == raw,
                "raw_preserved_before_extraction": json.loads(log["extractor"]["serialized_input"]) == raw.strip(),
                "prediction_has_literal_newline_escape": "\\n" in str(log["final_answer"]),
                "retrieval_titles_changed": old[qid]["retrieved_titles"] != json.loads(log["reasoning_chain"].split("\nA: ", 1)[1].split("\n", 1)[0]),
            }
            rows.append(row)
    summary = []
    for group in ("combined", "bridge", "comparison", "yes_no", "entity"):
        for k in (1, 8, 15):
            selected = [row for row in rows if row["k"] == k and (group == "combined" or row["type"] == group or group == "yes_no" and row["yes_no"] or group == "entity" and not row["yes_no"])]
            entry = {"group": group, "k": k, "count": len(selected)}
            for field in ("old_f1", "new_f1", "f1_delta", "old_official_f1", "new_official_f1", "old_em", "new_em", "old_primary_match", "new_primary_match"):
                entry[field] = statistics.mean(row[field] for row in selected) if selected else None
            for field in ("old_marker", "new_marker", "raw_has_newline", "prompt_changed", "test_example_changed", "prediction_has_literal_newline_escape", "retrieval_titles_changed"):
                entry[field + "_count"] = sum(bool(row[field]) for row in selected)
            entry["finish_reasons"] = dict(Counter(row["finish_reason"] for row in selected))
            entry["prompt_tokens"] = {"min": min(row["prompt_tokens"] for row in selected), "max": max(row["prompt_tokens"] for row in selected), "mean": statistics.mean(row["prompt_tokens"] for row in selected)} if selected else None
            entry["completion_tokens"] = {"min": min(row["completion_tokens"] for row in selected), "max": max(row["completion_tokens"] for row in selected), "mean": statistics.mean(row["completion_tokens"] for row in selected)} if selected else None
            entry["lowered_max_tokens_count"] = sum(row["requested_max_tokens"] < 300 for row in selected)
            entry["generation_content_loss_count"] = sum(not row["raw_preserved_by_generator"] or not row["raw_preserved_before_extraction"] for row in selected)
            summary.append(entry)
    (output / "comparison.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as stream:
        fields = [key for key in rows[0] if key not in ("old_cot", "new_cot")]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    review = ["# 全部 120 条输出的离线复核材料", "", "此文件只展示已生成结果，不改变预测或评分。"]
    for row in rows:
        review.extend(["", "## k={} / {} / {}".format(row["k"], row["type"], row["qid"]), "", row["question"], "", "Gold: " + json.dumps(row["gold"], ensure_ascii=False), "", "旧答案: " + str(row["old_prediction"]), "", "新答案: " + str(row["new_prediction"]), "", "停止原因: {}; completion_tokens={}; regex_match={}; F1={:.2f}→{:.2f}".format(row["finish_reason"], row["completion_tokens"], row["new_primary_match"], row["old_f1"], row["new_f1"]), "", "```text", row["new_cot"], "```"])
    (output / "raw_output_review.md").write_text("\n".join(review) + "\n")
    print(json.dumps([entry for entry in summary if entry["group"] == "combined"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
