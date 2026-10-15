"""Compare old/new configured regexes using saved text and the current JSON decoding."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

import _jsonnet

from analysis.hotpot_unicode_json_diagnostic.replay import (
    aggregate,
    replay_raw,
    run_official,
    scores,
    sha256,
    write_json,
)
from evaluate import evaluate_by_dicts
from run import instatiate_config


HISTORICAL_CONFIG = ROOT / "base_configs/oner_qa_qwen_hotpotqa_prompt_token_diagnostic.jsonnet"
CANDIDATE_CONFIG = ROOT / "base_configs/oner_qa_qwen_hotpotqa_formal.jsonnet"
OLD_REGEX = r".* answer is:? (.*)\.?"
NEW_REGEX = r"(?s).* answer is:? ([^\r\n]*)\.?"


def configured(path, k):
    content = instatiate_config(path.read_text(encoding="utf-8"), {"bm25_retrieval_count": str(k)})
    return json.loads(
        _jsonnet.evaluate_snippet(str(path), content, ext_vars={"RETRIEVER_HOST": "offline", "RETRIEVER_PORT": "0"})
    )


def extractor_settings(config):
    settings = dict(config["models"]["extract_answer"])
    assert settings.pop("name") == "answer_extractor"
    return settings


def protected_hashes():
    # Include the original experiment, the Unicode fix/regression, configs,
    # prompts, public code and evaluators; exclude only the new candidate.
    files = []
    for folder in (
        "commaqa",
        "metrics",
        "official_evaluation",
        "prompts",
        "base_configs",
        "instantiated_configs",
        "analysis/hotpot_unicode_json_diagnostic",
        "predictions/hotpot_prompt_token_40_20261013",
        "predictions/hotpot_unicode_json_replay_120_20261014",
        "predictions/hotpot_unicode_json_replay_120_20261014_repeat",
    ):
        files.extend((ROOT / folder).rglob("*"))
    files.extend(
        ROOT / name
        for name in (
            "run.py",
            "predict.py",
            "evaluate.py",
            "evaluate_per_question.py",
            "raw_data/hotpotqa/hotpot_dev_distractor_v1.json",
            "tests/test_answer_extractor_json.py",
        )
    )
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(set(files))
        if path.is_file() and "__pycache__" not in path.parts and path != CANDIDATE_CONFIG
    }


def run(source, baseline, output):
    source, baseline, output = source.resolve(), baseline.resolve(), output.resolve()
    assert not output.exists(), "Use a new output directory; never overwrite results"
    assert source != output and baseline != output and source not in output.parents and baseline not in output.parents
    fingerprints = protected_hashes()
    logs = [json.loads(line) for line in (source / "diagnostics.jsonl").read_text().splitlines()]
    prior = {
        (row["k"], row["qid"]): row for row in map(json.loads, (baseline / "comparison.jsonl").read_text().splitlines())
    }
    manifest = json.loads((source / "manifest.json").read_text())
    assert len(logs) == len(prior) == 120
    assert [(row["k"], row["qid"]) for row in logs] == [
        (k, qid) for k in (1, 8, 15) for qid in manifest["question_ids"]
    ]
    configurations = {}
    for k in range(1, 16):
        old, new = configured(HISTORICAL_CONFIG, k), configured(CANDIDATE_CONFIG, k)
        assert old["models"]["extract_answer"]["regex"] == OLD_REGEX
        assert new["models"]["extract_answer"]["regex"] == NEW_REGEX
        restored = json.loads(json.dumps(new))
        restored["models"]["extract_answer"]["regex"] = OLD_REGEX
        assert restored == old, "Candidate must differ only in the extractor regex"
        assert new["models"]["generate_titles"]["retrieval_count"] == k
        configurations[k] = (old, new)
    output.mkdir(parents=True)
    for k, (_, new) in configurations.items():
        directory = output / "evaluated_candidate_configs"
        directory.mkdir(exist_ok=True)
        write_json(directory / "k_{}.json".format(k), new)
    rows = []
    with patch("httpx.Client.send", side_effect=AssertionError("Offline replay forbids HTTP")), patch(
        "requests.sessions.Session.request", side_effect=AssertionError("Offline replay forbids HTTP")
    ):
        for log in logs:
            raw = log["api_responses"][0]["body"]["choices"][0]["message"]["content"]
            assert log["generator_outputs"] == [[raw, 0]]
            previous = prior[(log["k"], log["qid"])]
            assert previous["raw_output"] == raw
            before = replay_raw(raw, extractor_settings(configurations[log["k"]][0]))
            after = replay_raw(raw, extractor_settings(configurations[log["k"]][1]))
            assert before == previous["after"], "Old regex must reproduce the Unicode-fixed pipeline"
            for field in ("serialized_input", "llm_answer", "regex_input"):
                assert before[field] == after[field]
            assert after["regex_input"] == raw.strip()
            # Gold enters only scoring and integrity checks, never replay_raw().
            gold = log["gold"]
            assert gold == previous["gold"]
            before_scores, after_scores = scores(before["prediction"], gold), scores(after["prediction"], gold)
            assert before_scores == previous["after_scores"]
            rows.append(
                {
                    "k": log["k"],
                    "qid": log["qid"],
                    "type": log["type"],
                    "question": log["question"],
                    "gold": gold,
                    "raw_output": raw,
                    "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "before": before,
                    "after": after,
                    "before_scores": before_scores,
                    "after_scores": after_scores,
                    "changed": before["prediction"] != after["prediction"],
                    "known_unicode_case": previous["known_unicode_pollution"],
                    "yes_no": gold[0].lower() in ("yes", "no"),
                }
            )
    changed = [row for row in rows if row["changed"]]
    # General invariant checks: no question-specific extraction/correction.
    assert len(changed) == 1 and all("\n" in row["raw_output"] or "\r" in row["raw_output"] for row in changed)
    assert all(row["after"]["primary_match"] for row in rows)
    unicode_rows = [row for row in rows if row["known_unicode_case"]]
    assert len(unicode_rows) == 5 and all(not row["changed"] for row in unicode_rows)
    assert all(not row["changed"] for row in rows if row["yes_no"])
    summary = {
        "all": aggregate(rows),
        "by_k": {str(k): aggregate([row for row in rows if row["k"] == k]) for k in (1, 8, 15)},
        "unicode_cases_unchanged": len(unicode_rows),
        "yes_no_records": sum(row["yes_no"] for row in rows),
        "unchanged_predictions": sum(not row["changed"] for row in rows),
        "model_api_calls": 0,
    }
    raw_gold_path = ROOT / "raw_data/hotpotqa/hotpot_dev_distractor_v1.json"
    raw_gold = {
        row["_id"]: row for row in json.loads(raw_gold_path.read_text()) if row["_id"] in manifest["question_ids"]
    }
    assert len(raw_gold) == 40
    commands = []
    for k in (1, 8, 15):
        selected = [row for row in rows if row["k"] == k]
        assert all(row["gold"] == [raw_gold[row["qid"]]["answer"]] for row in selected)
        for version in ("before", "after"):
            directory = output / "k_{}".format(k) / version
            directory.mkdir(parents=True)
            predictions = {row["qid"]: row[version]["prediction"] for row in selected}
            write_json(directory / "predictions.json", predictions)
            gold = {row["qid"]: row["gold"] for row in selected}
            repository = evaluate_by_dicts("answer", gold, predictions, "hotpotqa")
            write_json(directory / "repository_metrics.json", repository)
            official, command = run_official(predictions, [raw_gold[row["qid"]] for row in selected], directory)
            commands.append(command)
            means = summary["by_k"][str(k)][version]
            for metric in ("f1", "em"):
                assert math.isclose(official[metric], means["official_" + metric], abs_tol=1e-12)
                assert abs(repository[metric] - means["repository_" + metric]) <= 0.0005 + 1e-12
            if version == "before":
                assert predictions == json.loads((baseline / "k_{}".format(k) / "after/predictions.json").read_text())
    write_json(output / "summary.json", summary)
    write_json(output / "changed_predictions.json", changed)
    (output / "comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "k",
                "qid",
                "changed",
                "before_prediction",
                "after_prediction",
                "before_official_f1",
                "after_official_f1",
                "before_official_em",
                "after_official_em",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "k": row["k"],
                    "qid": row["qid"],
                    "changed": row["changed"],
                    "before_prediction": row["before"]["prediction"],
                    "after_prediction": row["after"]["prediction"],
                    **{
                        version + "_" + name: row[version + "_scores"][name]
                        for version in ("before", "after")
                        for name in ("official_f1", "official_em")
                    },
                }
            )
    assert protected_hashes() == fingerprints, "Protected historical code/data changed"
    write_json(output / "protected_hashes.json", fingerprints)
    write_json(
        output / "validation.json",
        {
            "records": 120,
            "baseline_traces_and_scores_match_unicode_fix": 120,
            "identical_raw_and_serialized_inputs": 120,
            "changed_predictions": len(changed),
            "unchanged_predictions": 119,
            "unicode_cases_unchanged": 5,
            "yes_no_unchanged": True,
            "candidate_k_1_to_15_only_regex_differs": True,
            "protected_files_unchanged": len(fingerprints),
            "public_extractor_unchanged": True,
            "official_cli_runs": 6,
            "repository_evaluator_runs": 6,
            "model_api_calls": 0,
        },
    )
    paths = [
        CANDIDATE_CONFIG,
        HISTORICAL_CONFIG,
        Path(__file__).resolve(),
        ROOT / "tests/test_hotpot_multiline_answer.py",
    ]
    for path in paths:
        target = output / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(path), str(target))
    write_json(
        output / "manifest.json",
        {
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip(),
            "source": str(source),
            "unicode_fixed_baseline": str(baseline),
            "source_diagnostics_sha256": sha256(source / "diagnostics.jsonl"),
            "public_extractor_sha256": sha256(ROOT / "commaqa/inference/ircot.py"),
            "old_regex": OLD_REGEX,
            "new_regex": NEW_REGEX,
            "python": sys.executable,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "official_commands": commands,
            "snapshots_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "predictions/hotpot_prompt_token_40_20261013")
    parser.add_argument("--baseline", type=Path, default=ROOT / "predictions/hotpot_unicode_json_replay_120_20261014")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.baseline, args.output)
