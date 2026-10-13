"""Replay saved model text through both answer pipelines without model/network calls."""

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

import _jsonnet
import requests

from commaqa.inference.data_instances import QuestionAnsweringStep, QuestionGenerationStep, StructuredDataInstance
from commaqa.inference.ircot import AnswerExtractor
from commaqa.inference.model_search import ParticipantModel, QuestionSearchBase, SearchState
from commaqa.inference.participant_qa import LLMQAParticipantModel
from evaluate import evaluate_by_dicts
from evaluate_per_question import prepare_prediction_answers, score_per_question
from official_evaluation.hotpotqa.hotpot_evaluate_v1 import exact_match_score, f1_score


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_baseline(path):
    # Compile the exact saved class, without executing module-level imports or clients.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AnswerExtractor")
    namespace = {
        "re": re,
        "json": json,
        "ParticipantModel": ParticipantModel,
        "QuestionAnsweringStep": QuestionAnsweringStep,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["AnswerExtractor"]


class SavedGenerator:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def generate_text_sequence(self, prompt):
        self.calls += 1
        return [(self.raw, 0)]


class RecordedRegex:
    def __init__(self, regex):
        self.regex = regex
        self.calls = []

    def match(self, text):
        result = self.regex.match(text)
        self.calls.append((text, result))
        return result


def replay_raw(raw, settings, extractor_class=AnswerExtractor):
    """Gold is deliberately absent from this interface."""
    generator = SavedGenerator(raw)
    with patch("commaqa.execution.llm_qa_model.GPT3Generator", return_value=generator):
        participant = LLMQAParticipantModel(next_model="extract_answer", gen_model="gpt3")
    # Prompt construction is irrelevant to a saved completion. No prompt reader,
    # tokenizer, retrieval or API generation is run by this offline harness.
    participant.qa_model.prompt = ""
    data = StructuredDataInstance(
        {"qid": "replay", "question": "Offline replay", "titles": [], "paras": [], "metadata": {}}
    )
    data.add_qgen(QuestionGenerationStep(question="Offline replay", score=0, participant="copy_question"))
    state = participant.query(SearchState(data, "answer_main_question"))
    serialized = state.data.get_last_answer()
    extraction = extractor_class(**settings)
    recorded = RecordedRegex(extraction.regex)
    extraction.regex = recorded
    result = extraction.query(state)
    assert isinstance(result, SearchState), "Extractor failed to produce a state"
    search = QuestionSearchBase(None)
    with patch.object(search, "find_answer_decomp", return_value=(result, [])):
        prediction = search.return_qid_prediction({"qid": "replay", "question": "Offline replay"}, silent=True)[1]
    query, primary = recorded.calls[0]
    assert generator.calls == participant.qa_model.num_calls == participant.num_calls == 1
    return {
        "serialized_input": serialized,
        "llm_answer": json.loads(serialized),
        "regex_input": query,
        "primary_match": primary is not None,
        "matched_group": primary.group(1) if primary else None,
        "fallback": primary is None and extraction.match_all_on_failure,
        "extractor_output": result.data.get_last_answer(),
        "prediction": prediction,
    }


def scores(prediction, gold):
    repository = score_per_question("hotpotqa", prepare_prediction_answers(prediction), gold)
    # Same list flattening as evaluate.py's official path.
    official = " ".join(str(x) for x in prediction) if isinstance(prediction, list) else str(prediction)
    return {
        "official_em": max(float(exact_match_score(official, span)) for span in gold),
        "official_f1": max(f1_score(official, span)[0] for span in gold),
        "repository_em": repository["em"],
        "repository_f1": float(repository["f1"]),
    }


def aggregate(rows):
    result = {"count": len(rows), "changed_predictions": sum(row["changed"] for row in rows)}
    for version in ("before", "after"):
        result[version] = {
            metric: statistics.mean(row[version + "_scores"][metric] for row in rows)
            for metric in ("official_em", "official_f1", "repository_em", "repository_f1")
        }
        result[version]["primary_matches"] = sum(row[version]["primary_match"] for row in rows)
    result["delta"] = {metric: result["after"][metric] - result["before"][metric] for metric in result["before"]}
    return result


def run_official(predictions, raw_gold, directory):
    write_json(
        directory / "official_predictions.json", {"answer": predictions, "sp": {qid: [["", 0]] for qid in predictions}}
    )
    write_json(directory / "official_gold.json", raw_gold)
    command = [
        sys.executable,
        str(ROOT / "official_evaluation/hotpotqa/hotpot_evaluate_v1.py"),
        str(directory / "official_predictions.json"),
        str(directory / "official_gold.json"),
    ]
    completed = subprocess.run(
        command, cwd=str(ROOT), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    (directory / "official_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    metrics = ast.literal_eval(completed.stdout.strip())
    write_json(directory / "official_metrics.json", metrics)
    return metrics, command


def run(source, output):
    source, output = source.resolve(), output.resolve()
    assert source != output and source not in output.parents, "Use a separate output directory"
    assert not output.exists(), "Refusing to overwrite an existing replay"
    protected = json.loads((Path(__file__).resolve().parent / "protected_before.json").read_text())
    hashes_at_start = {name: sha256(ROOT / name) for name in protected}
    expected_changes = ["commaqa/inference/ircot.py"]
    assert [name for name in protected if hashes_at_start[name] != protected[name]] == expected_changes
    logs = [json.loads(line) for line in (source / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((source / "manifest.json").read_text())
    samples = {row["question_id"]: row for row in map(json.loads, (source / "samples.jsonl").read_text().splitlines())}
    raw_gold_source = ROOT / "raw_data/hotpotqa/hotpot_dev_distractor_v1.json"
    raw_gold = {row["_id"]: row for row in json.loads(raw_gold_source.read_text()) if row["_id"] in samples}
    assert len(logs) == 120 and len(samples) == len(raw_gold) == 40
    assert [(log["k"], log["qid"]) for log in logs] == [
        (k, qid) for k in (1, 8, 15) for qid in manifest["question_ids"]
    ]
    baseline_path = Path(__file__).resolve().parent / "ircot_before.py.txt"
    baseline = load_baseline(baseline_path)
    # The source snapshot was captured before editing and matches the code hash
    # that the original experiment recorded for this module.
    original_hashes = manifest["protected_sha256"]
    assert sha256(baseline_path) == original_hashes[str(ROOT / "commaqa/inference/ircot.py")]
    configurations = {}
    for k in (1, 8, 15):
        directory = next(source.glob("*retrieval_count__{}___*".format(k)))
        config_path = next(directory.glob("config__*.jsonnet"))
        config = json.loads(
            _jsonnet.evaluate_file(str(config_path), ext_vars={"RETRIEVER_HOST": "offline", "RETRIEVER_PORT": "0"})
        )
        settings = dict(config["models"]["extract_answer"])
        assert settings.pop("name") == "answer_extractor"
        configurations[k] = (directory, settings)
    output.mkdir(parents=True)
    rows = []
    # Fail on any accidental network call from the replay process.
    with patch("httpx.Client.send", side_effect=AssertionError("Offline replay forbids HTTP")), patch.object(
        requests.sessions.Session, "request", side_effect=AssertionError("Offline replay forbids HTTP")
    ):
        for log in logs:
            raw = log["api_responses"][0]["body"]["choices"][0]["message"]["content"]
            assert log["generator_outputs"] == [[raw, 0]]
            settings = configurations[log["k"]][1]
            assert settings["regex"] == log["extractor"]["regex"]
            before = replay_raw(raw, settings, baseline)
            after = replay_raw(raw, settings)
            for field in ("serialized_input", "regex_input", "primary_match", "matched_group", "fallback"):
                assert before[field] == log["extractor"][field], (log["k"], log["qid"], field)
            assert before["extractor_output"] == log["extractor"]["output"]
            assert before["prediction"] == log["final_answer"]
            assert before["serialized_input"] == after["serialized_input"]
            assert before["llm_answer"] == after["llm_answer"] == raw.strip()
            assert after["regex_input"] == raw.strip()
            gold = log["gold"]
            sample_gold = [span for answer in samples[log["qid"]]["answers_objects"] for span in answer["spans"]]
            assert gold == sample_gold == [raw_gold[log["qid"]]["answer"]]
            before_scores, after_scores = scores(before["prediction"], gold), scores(after["prediction"], gold)
            assert all(math.isclose(before_scores[key], log["scores"][key], abs_tol=1e-12) for key in before_scores)
            changed = before["prediction"] != after["prediction"]
            unicode_pollution = bool(re.search(r"\\u[0-9a-fA-F]{4}", before["prediction"]))
            reason = (
                "Unicode JSON escapes restored"
                if changed and unicode_pollution
                else "Real newline restored before unchanged regex/fallback" if changed else "unchanged"
            )
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
                    "changed": changed,
                    "known_unicode_pollution": unicode_pollution,
                    "reason": reason,
                }
            )
    unicode_rows = [row for row in rows if row["known_unicode_pollution"]]
    other_changes = [row for row in rows if row["changed"] and not row["known_unicode_pollution"]]
    assert len(unicode_rows) == 5 and all(row["changed"] for row in unicode_rows)
    assert all(not re.search(r"\\u[0-9a-fA-F]{4}", row["after"]["prediction"]) for row in unicode_rows)
    assert len(other_changes) == 1 and "\n" in other_changes[0]["raw_output"]
    summary = {
        "all": aggregate(rows),
        "by_k": {str(k): aggregate([row for row in rows if row["k"] == k]) for k in (1, 8, 15)},
        "known_unicode_records_fixed": 5,
        "other_115_unchanged": 114,
        "other_115_changed": len(other_changes),
        "gold_used_in_prediction": False,
        "model_api_calls": 0,
    }
    commands = []
    for k in (1, 8, 15):
        selected = [row for row in rows if row["k"] == k]
        historical_directory = configurations[k][0]
        saved = json.loads(next(historical_directory.glob("prediction__*.json")).read_text())
        assert saved == {row["qid"]: row["before"]["prediction"] for row in selected}
        for version in ("before", "after"):
            directory = output / ("k_{}".format(k)) / version
            directory.mkdir(parents=True)
            predictions = {row["qid"]: row[version]["prediction"] for row in selected}
            gold = {row["qid"]: row["gold"] for row in selected}
            write_json(directory / "predictions.json", predictions)
            repository = evaluate_by_dicts("answer", gold, predictions, "hotpotqa")
            write_json(directory / "repository_metrics.json", repository)
            official, command = run_official(predictions, [raw_gold[row["qid"]] for row in selected], directory)
            commands.append(command)
            means = summary["by_k"][str(k)][version]
            assert math.isclose(official["f1"], means["official_f1"], abs_tol=1e-12)
            assert math.isclose(official["em"], means["official_em"], abs_tol=1e-12)
            for metric in ("f1", "em"):
                assert abs(repository[metric] - means["repository_" + metric]) <= 0.0005 + 1e-12
            if version == "before":
                saved_official = json.loads(
                    next(historical_directory.glob("official_evaluation_metrics__*.json")).read_text()
                )
                saved_repository = json.loads(next(historical_directory.glob("evaluation_metrics__*.json")).read_text())
                assert all(round(official[key], 5) == saved_official[key] for key in ("f1", "em"))
                assert all(abs(repository[key] - saved_repository[key]) <= 0.001 + 1e-12 for key in ("f1", "em"))
    (output / "comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    write_json(output / "changed_predictions.json", [row for row in rows if row["changed"]])
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "k",
            "qid",
            "type",
            "question",
            "gold",
            "before_prediction",
            "after_prediction",
            "changed",
            "reason",
            "before_official_em",
            "after_official_em",
            "before_official_f1",
            "after_official_f1",
            "before_repository_em",
            "after_repository_em",
            "before_repository_f1",
            "after_repository_f1",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields if key in row}
            for version in ("before", "after"):
                flat[version + "_prediction"] = row[version]["prediction"]
                flat.update({version + "_" + metric: value for metric, value in row[version + "_scores"].items()})
            writer.writerow(flat)
    write_json(output / "summary.json", summary)
    hashes_at_end = {name: sha256(ROOT / name) for name in protected}
    assert hashes_at_end == hashes_at_start, "Replay changed a protected file"
    write_json(
        output / "validation.json",
        {
            "records": 120,
            "unique_questions": 40,
            "ordered_inputs_identical": True,
            "baseline_predictions_match_saved": 120,
            "baseline_traces_and_scores_match_saved": 120,
            "fixed_regex_inputs_equal_raw_stripped": 120,
            "official_cli_runs": 6,
            "repository_evaluator_runs": 6,
            "protected_files": len(protected),
            "changed_from_original": expected_changes,
            "protected_files_unchanged_during_replay": True,
            "history_unchanged": True,
            "model_api_calls": 0,
        },
    )
    write_json(
        output / "protected_hashes.json",
        {"before_task": protected, "before_replay": hashes_at_start, "after_replay": hashes_at_end},
    )
    (output / "fix.diff").write_bytes(
        subprocess.check_output(["git", "diff", "--", "commaqa/inference/ircot.py"], cwd=str(ROOT))
    )
    snapshots = [
        baseline_path,
        Path(__file__).resolve().parent / "protected_before.json",
        Path(__file__).resolve(),
        ROOT / "commaqa/inference/ircot.py",
        ROOT / "tests/test_answer_extractor_json.py",
    ]
    for path in snapshots:
        target = output / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(path), str(target))
    write_json(
        output / "manifest.json",
        {
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip(),
            "source": str(source),
            "source_diagnostics_sha256": sha256(source / "diagnostics.jsonl"),
            "source_samples_sha256": sha256(source / "samples.jsonl"),
            "official_gold_sha256": sha256(raw_gold_source),
            "baseline_ircot_sha256": sha256(baseline_path),
            "fixed_ircot_sha256": sha256(ROOT / "commaqa/inference/ircot.py"),
            "python": sys.executable,
            "python_version": sys.version,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "versions": {
                name: importlib.metadata.version(name) for name in ("openai", "httpx", "numpy", "ujson", "ftfy")
            },
            "settings": {str(k): configurations[k][1] for k in configurations},
            "official_commands": commands,
            "replay_command": [
                sys.executable,
                str(Path(__file__).resolve()),
                "--source",
                str(source),
                "--output",
                str(output),
            ],
            "generation": "Saved text only; generator stub; parent HTTP calls fail closed; no model, retrieval or tokenizer run",
            "snapshot_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in snapshots},
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # evaluate.py iterates a set of question IDs. Fix iteration order so its
    # existing three-decimal rounding is reproducible at half-way boundaries.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "predictions/hotpot_prompt_token_40_20261013")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
