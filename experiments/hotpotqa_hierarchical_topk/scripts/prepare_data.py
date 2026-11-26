"""Validate frozen HotpotQA runs and assemble per-question answer scores and costs."""

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
K_VALUES = tuple(range(16))
K_KEYS = {str(k) for k in K_VALUES}
QUESTION_COUNT = 5000
LEVELS = ("easy", "medium", "hard")
TYPES = ("bridge", "comparison")
DEFAULT_SETTINGS = HERE / "configs/settings.json"

sys.path.insert(0, str(ROOT))
from analysis.hotpot_formal_runner.storage import sha256, text_hash  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key {key!r}")
        value[key] = item
    return value


def parse_object(raw, location):
    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
        require(isinstance(value, dict), "Expected a JSON object")
    except ValueError as error:
        raise ValueError(f"{location}: {error}") from error
    return value


def read_json(path):
    return parse_object(path.read_bytes(), path)


def read_rows(path, expected_digest=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            yield parse_object(raw, f"{path}:{number}")
    if expected_digest is not None:
        require(digest.hexdigest() == expected_digest, f"SHA-256 mismatch in {path}")


def load_settings(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    config = read_json(parser.parse_args().settings)
    source = ROOT / config["source_dir"]
    inputs = {name: (source / path).resolve() for name, path in config["inputs"].items()}
    outputs = {name: (HERE / path).resolve() for name, path in config["outputs"].items()}
    require(all(HERE in path.parents for path in outputs.values()), "Outputs must stay inside the new experiment")
    require(len(set(outputs.values())) == len(outputs), "Output paths must be distinct")
    return inputs, outputs, config["tokenizer"]


def provenance(path, digest=None):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest if digest is not None else sha256(path)}


def check_ids(expected, actual, source):
    missing, extra = expected - actual, actual - expected
    require(not (missing or extra), f"{source}: {len(missing)} missing IDs {sorted(missing)[:5]}, "
            f"{len(extra)} unexpected IDs {sorted(extra)[:5]}")


def load_questions(path, digest):
    questions = {}
    for row in read_rows(path, digest):
        qid = row["question_id"]
        require(isinstance(qid, str) and qid and qid not in questions, f"Invalid or duplicate question ID {qid!r}")
        require(row["dataset"] == "hotpotqa" and isinstance(row["question_text"], str) and row["question_text"].strip(),
                f"Invalid HotpotQA question {qid}")
        require(row["level"] in LEVELS and row["type"] in TYPES, f"Invalid level or type for {qid}")
        answers = row["answers_objects"]
        require(isinstance(answers, list) and len(answers) == 1 and isinstance(answers[0]["spans"], list),
                f"Expected a single HotpotQA gold answer object for {qid}")
        spans = answers[0]["spans"]
        require(len(spans) == 1 and isinstance(spans[0], str) and spans[0].strip(),
                f"Expected a single nonempty HotpotQA gold span for {qid}")
        questions[qid] = {"qid": qid, "question": row["question_text"], "gold_answers": spans,
                          "level": row["level"], "type": row["type"], "results": {}}
    require(len(questions) == QUESTION_COUNT, f"Expected {QUESTION_COUNT} questions, got {len(questions)}")
    return questions


def validate_result(result, k, location):
    f1, em, tokens = result["f1"], result["em"], result["context_tokens"]
    require(isinstance(result["prediction"], str), f"{location}: prediction must be a string")
    require(type(f1) in (int, float) and 0 <= f1 <= 1, f"{location}: invalid F1 {f1!r}")
    require(type(em) is int and em in (0, 1), f"{location}: invalid EM {em!r}")
    require(type(tokens) is int and (tokens > 0 if k else tokens == 0),
            f"{location}: invalid Context Tokens {tokens!r}")


def load_tokenizer(config):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from huggingface_hub import snapshot_download
    from transformers import PreTrainedTokenizerFast

    snapshot = Path(snapshot_download(config["model"], revision=config["revision"], local_files_only=True))
    files = {name: sha256(snapshot / name) for name in ("tokenizer.json", "tokenizer_config.json")}
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(snapshot), local_files_only=True, model_max_length=120000)
    require(tokenizer.encode("", add_special_tokens=False) == [], "Qwen tokenizer should encode empty context to zero tokens")
    return tokenizer, {
        **config, "class": type(tokenizer).__name__, "files_sha256": files,
        "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers"),
        "field": "effective_context_text", "add_special_tokens": False, "truncation": False,
        "formula": "len(tokenizer.encode(effective_context_text, add_special_tokens=False))",
    }


def validate_context(row, question, expected, prediction, length_policy):
    from experiments.hotpotqa_train_5000_bert.src.metrics import render_context

    require(row["binding"] == expected and type(row["k"]) is int and row["k"] == expected["k"],
            "Context/run binding differs")
    require(row["input_question"] == question["question"].strip(), "Context question text differs")
    require(row["prediction"] == prediction, "Prediction and context record differ")
    require(row["effective_context_verified_in_prompt"] is True, "Effective context was not verified in the prompt")
    raw, reader, effective = row["raw_contexts"], row["reader_contexts"], row["effective_context_text"]
    require(isinstance(raw, list) and isinstance(reader, list), "Invalid raw/reader context lists")
    require(row["context_sha256"] == text_hash(effective), "Effective context hash differs")
    require(effective == "\n\n".join(reader) and reader == [render_context(doc) for doc in raw],
            "Raw, Reader and effective contexts disagree")
    if expected["k"] == 0:
        require(raw == reader == [] and effective == row["context_suffix"] == "", "k=0 retrieval context is not empty")
    else:
        require(0 < len(raw) <= expected["k"] and effective.strip(), "Invalid retrieval context count or text")
    require(row["chain_sha256"] == text_hash(row["chain"]), "Context chain hash differs")

    finish = row["api"]["finish_reasons"]
    skip_reason = row.get("skip_reason")
    if skip_reason is not None or finish == ["length"]:
        require(expected["k"] == 0 and skip_reason == "generation_length" and finish == ["length"] and prediction == "",
                "Unexpected generation length skip")
        source = row["api"]["finish_reason_source"]
        if source == "failed_attempt_log":
            evidence = length_policy["known_truncation"]
            require(row["cache_hit"] is True and row["api"]["usage"] is None and evidence["qid"] == row["qid"]
                    and evidence["call_sha256"] == row["call_sha256"]
                    and evidence["raw_generation_sha256"] == text_hash(row["raw_generation"]),
                    "Length skip evidence differs from the frozen policy")
        else:
            require(source == ("saved_api_response" if row["cache_hit"] else "api_response"),
                    "Invalid length skip provenance")
    else:
        require(finish in (None, ["stop"]), "Unexpected generation finish reason")
    return skip_reason is not None


def process_k(k, paths, manifest, manifest_digest, questions, moves, length_policy, count_tokens):
    from official_evaluation.hotpotqa.hotpot_evaluate_v1 import exact_match_score, f1_score

    base = paths["predictions_k0" if k == 0 else "predictions"]
    config = manifest["configs"][str(k)]
    config_path = ROOT / moves.get(config["path"], config["path"])
    require(sha256(config_path) == config["sha256"], f"Frozen config hash mismatch in {config_path}")
    run_dir = base / config_path.stem
    expected = {"manifest_sha256": manifest_digest, "input_sha256": manifest["datasets"]["full"]["sha256"],
                "config_sha256": config["sha256"], "k": k, "set_name": manifest["set_names"]["full"]}
    marker_path = run_dir / "complete.json"
    marker = read_json(marker_path)
    require(marker["status"] == "complete" and type(marker["predict_exit_code"]) is int and marker["predict_exit_code"] == 0
            and marker["binding"] == expected, f"Invalid completion marker {marker_path}")
    attempt_path = (ROOT / marker["attempt"]).resolve()
    require(attempt_path.parent == (run_dir / "attempts").resolve(), f"Attempt directory differs in {marker_path}")
    attempt_file = attempt_path / "attempt.json"
    attempt = read_json(attempt_file)
    require(attempt["binding"] == expected and attempt["status"] == "complete" and attempt["predict_exit_code"] == 0,
            f"Invalid completed attempt {attempt_file}")

    eval_name = f"hotpotqa_to_hotpotqa__{paths['questions'].stem}"
    stem = f"prediction__{eval_name}"
    prediction_name, chains_name = f"{stem}.json", f"{stem}_chains.txt"
    config_name = f"config__{eval_name}.jsonnet"
    input_name, replacements_name = f"{stem}_full_eval_path.txt", f"{stem}_variable_replacements.json"
    hashes = marker["output_hashes"]
    require(set(hashes) == {prediction_name, "contexts.jsonl", chains_name, config_name, input_name,
                            replacements_name, f"{stem}_time_taken.txt"}, "Completion output list differs")
    for name, digest in hashes.items():
        if name != "contexts.jsonl":  # Checked against the exact bytes parsed below.
            require(sha256(run_dir / name) == digest, f"Output SHA-256 mismatch in {run_dir / name}")
    require(hashes[config_name] == config["sha256"], "Prediction config snapshot does not match the manifest")
    input_path = (run_dir / input_name).read_text(encoding="utf-8")
    require((ROOT / moves.get(input_path, input_path)).resolve() == paths["questions"], "Prediction input path differs")
    require((run_dir / replacements_name).read_bytes() == b"", "Prediction used unexpected variable replacements")

    prediction_path = run_dir / prediction_name
    contexts_path = run_dir / "contexts.jsonl"
    predictions = read_json(prediction_path)
    check_ids(questions.keys(), predictions.keys(), prediction_path)
    require(all(isinstance(answer, str) for answer in predictions.values()), f"Non-string prediction in {prediction_path}")
    empty_count = sum(not answer.strip() for answer in predictions.values())
    if "empty_prediction_count" in marker:
        require(marker["empty_prediction_count"] == empty_count, "Completion marker empty-answer count differs")
    seen, skipped = set(), []
    chains = hashlib.sha256()
    for row in read_rows(contexts_path, hashes["contexts.jsonl"]):
        qid = row["qid"]
        require(isinstance(qid, str) and qid in questions and qid not in seen,
                f"Unexpected or duplicate context ID {qid!r} in {contexts_path}")
        seen.add(qid)
        question, prediction = questions[qid], predictions[qid]
        try:
            if validate_context(row, question, expected, prediction, length_policy):
                skipped.append(qid)
            chains.update((row["chain"] + "\n").encode("utf-8"))
            gold = question["gold_answers"][0]
            result = {"prediction": prediction, "f1": f1_score(prediction, gold)[0],
                      "em": int(exact_match_score(prediction, gold)), "context_tokens": count_tokens(row)}
            validate_result(result, k, qid)
            question["results"][str(k)] = result
        except (ValueError, TypeError, AttributeError, KeyError) as error:
            raise ValueError(f"{contexts_path} qid={qid}: {error}") from error
    check_ids(questions.keys(), seen, contexts_path)
    require(chains.hexdigest() == hashes[chains_name], "Chain file and context rows disagree")

    summary = {"binding": expected, "prediction_count": len(predictions), "context_count": len(seen),
               "empty_answer_count": empty_count, "complete": provenance(marker_path), "attempt": provenance(attempt_file),
               "prediction": provenance(prediction_path, hashes[prediction_name]),
               "contexts": provenance(contexts_path, hashes["contexts.jsonl"]), "output_hashes": hashes}
    if k == 0:
        length_digest = sha256(paths["k0_length_policy"])
        require(marker["length_policy_sha256"] == attempt["length_policy_sha256"] == length_digest
                and marker["skipped_length_qids"] == attempt["skipped_length_qids"] == skipped,
                "k=0 length policy or skipped-question summary differs")
        manual = marker.get("manual_empty_prediction_qids", [])
        require(len(set(manual)) == len(manual) and all(predictions[qid] == "" and qid not in skipped for qid in manual),
                "k=0 manually recorded empty predictions disagree with the completion marker")
        summary.update(skipped_length_qids=skipped, manual_empty_prediction_qids=manual)
    return summary


def write_results(path, report_path, questions, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, suffix=".tmp") as data_dir, \
            TemporaryDirectory(dir=report_path.parent, suffix=".tmp") as report_dir:
        data_file = Path(data_dir) / path.name
        audit_file = Path(report_dir) / report_path.name
        with data_file.open("w", encoding="utf-8") as data:
            for question in questions.values():
                data.write(json.dumps(question, ensure_ascii=False, allow_nan=False) + "\n")
        report["output"] = provenance(path, sha256(data_file))
        with audit_file.open("w", encoding="utf-8") as audit:
            json.dump(report, audit, ensure_ascii=False, indent=2, allow_nan=False)
            audit.write("\n")
        data_file.replace(path)
        audit_file.replace(report_path)


def prepare(paths, outputs, tokenizer_config):
    sample = read_json(paths["sample_manifest"])
    parent, zero = (read_json(paths[name]) for name in ("run_manifest", "run_manifest_k0"))
    migration = read_json(paths["directory_migration"])
    policy = read_json(paths["k0_length_policy"])
    moves = migration["moves"]
    input_sources = {name: provenance(path) for name, path in paths.items() if not name.startswith("predictions")}
    parent_digest = input_sources["run_manifest"]["sha256"]
    zero_digest = input_sources["run_manifest_k0"]["sha256"]
    compatibility_digest = input_sources["k0_compatibility"]["sha256"]
    require(parent_digest == zero["parent_manifest_sha256"] == migration["run_manifest_sha256"]
            and zero_digest == policy["run_manifest_k0_sha256"]
            and compatibility_digest == zero["compatibility_sha256"] == policy["compatibility_sha256"],
            "Parent, k=0 run manifest, migration and policy identities disagree")
    require(set(parent["configs"]) == K_KEYS - {"0"} and set(zero["configs"]) == {"0"}, "Manifest k coverage differs")
    full = parent["datasets"]["full"]
    input_digest = input_sources["questions"]["sha256"]
    require(input_digest == full["sha256"] == zero["datasets"]["full"]["sha256"] == sample["sample"]["sha256"],
            "Input SHA-256 does not match the frozen manifests")
    for entry in (full, zero["datasets"]["full"], sample["sample"]):
        require((ROOT / moves.get(entry["path"], entry["path"])).resolve() == paths["questions"], "Manifest input path differs")
    require(full["count"] == zero["datasets"]["full"]["count"] == sample["sample_size"] == QUESTION_COUNT,
            "Frozen input counts differ")
    questions = load_questions(paths["questions"], input_digest)
    require(list(questions) == full["question_ids"] == sample["question_ids"], "Frozen input ID order differs")

    for manifest, digest, name in ((parent, parent_digest, "predictions"), (zero, zero_digest, "predictions_k0")):
        base = paths[name]
        identity = base / "run_identity.json"
        require(base.name == manifest["set_names"]["full"]
                and read_json(identity) == {"manifest_sha256": digest, "mode": "full"}, f"Invalid full run identity {identity}")
        input_sources[name + "_identity"] = provenance(identity)

    tokenizer, tokenizer_report = load_tokenizer(tokenizer_config)
    counts = {}

    def count_tokens(row):
        key = row["context_sha256"]
        if key not in counts:
            counts[key] = len(tokenizer.encode(row["effective_context_text"], add_special_tokens=False))
        return counts[key]

    per_k = {}
    for k in K_VALUES:
        manifest, digest = (zero, zero_digest) if k == 0 else (parent, parent_digest)
        try:
            per_k[str(k)] = process_k(k, paths, manifest, digest, questions, moves, policy, count_tokens)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"k={k}: {error}") from error
        print(f"k={k:2d} validated {per_k[str(k)]['prediction_count']} predictions and contexts", flush=True)

    require(all(set(question["results"]) == K_KEYS for question in questions.values()), "Incomplete per-question k coverage")
    record_count = sum(len(question["results"]) for question in questions.values())
    require(record_count == QUESTION_COUNT * len(K_VALUES), f"Wrong total record count {record_count}")
    report = {
        "status": "passed", "question_count": len(questions), "question_k_record_count": record_count,
        "k_values": list(K_VALUES), "inputs": input_sources, "per_k": per_k,
        "empty_answer_count": sum(row["empty_answer_count"] for row in per_k.values()),
        "checks": {
            "unique_question_ids_and_frozen_order": True, "every_k_has_exact_input_ids_in_predictions_and_contexts": True,
            "run_identities_completion_attempts_and_output_hashes_validated": True,
            "context_question_prediction_config_and_chain_agree": True, "effective_context_verified_in_prompt": True,
            "complete_result_matrix_and_valid_metric_ranges": True, "k0_empty_retrieval_context_and_zero_tokens": True,
        },
        "scoring": {"implementation": provenance(ROOT / "official_evaluation/hotpotqa/hotpot_evaluate_v1.py"),
                    "f1": "f1_score(prediction, gold)[0]", "em": "int(exact_match_score(prediction, gold))",
                    "gold": "one span from answers_objects[0].spans per original HotpotQA question",
                    "empty_answers": "Empty/whitespace-only predictions are counted by not prediction.strip(); "
                                     "the original string is retained and evaluated by the official functions."},
        "tokenizer": tokenizer_report,
    }
    write_results(outputs["per_question_metrics"], outputs["validation_report"], questions, report)
    print(f"PASS: {len(questions)} questions, {record_count} question-k records, {report['empty_answer_count']} empty answers")


if __name__ == "__main__":
    try:
        prepare(*load_settings(__doc__))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        sys.exit(f"Data preparation failed: {error}")
