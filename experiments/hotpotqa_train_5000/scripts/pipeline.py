"""Prepare, run and strictly validate the existing HotpotQA ONER pipeline."""
import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from common import (HERE, ROOT, DATA, RUNS, SCRIPTS, SMOKE, LOGS, artifact_path, atomic_json, atomic_text, binding, configured, environment, now,
                    prediction_path, read_json, read_rows, require, settings, sha256, text_hash)
from sample import distributions, verify as verify_sample

PACKAGES = ("openai", "httpx", "diskcache", "transformers", "jsonnet", "torch")


def code_files():
    return (sorted((ROOT / "commaqa").rglob("*.py")) + sorted(SCRIPTS.glob("*.py")) + sorted((HERE / "tests").glob("*.py")) +
            [ROOT / name for name in ("run.py", "predict.py", "lib.py",
                                     ".retriever_address.jsonnet", ".llm_server_address.jsonnet",
                                     "analysis/hotpot_formal_runner/storage.py")] +
            [HERE / "run.sh", HERE / "settings.json"])


def execute(command, log, env, lock_fd=None):
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n{} COMMAND {}\n".format(now(), json.dumps(command)))
        stream.flush()
        child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=() if lock_fd is None else (lock_fd,))
        try:
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            # Include predict.py's shell and inference grandchild in interruption.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise


def load_manifest():
    manifest = read_json(RUNS / "run_manifest.json")
    migration_path = RUNS / "directory_migration.json"
    migration = read_json(migration_path) if migration_path.exists() else None
    if migration:
        require(sha256(RUNS / "run_manifest.json") == migration["run_manifest_sha256"], "Original run manifest changed")
    policy_path = RUNS / "empty_answer_policy.json"
    policy = read_json(policy_path) if policy_path.exists() else None
    if policy:
        require(policy["run_manifest_sha256"] == sha256(RUNS / "run_manifest.json"), "Policy manifest differs")
        require(policy["directory_migration_sha256"] == sha256(migration_path), "Policy migration differs")
    for key, expected in manifest["file_hashes"].items():
        if migration and key in migration["changed_files"]:
            entry = migration["changed_files"][key]
            require(entry["original_sha256"] == expected, "Migration source hash differs: " + key)
            expected = entry["sha256"]
        if policy and key in policy["changed_files"]:
            entry = policy["changed_files"][key]
            require(entry["previous_sha256"] == expected, "Policy source hash differs: " + key)
            expected = entry["sha256"]
        require(sha256(artifact_path(key)) == expected, "Frozen experiment file changed: " + key)
    require(str(Path(sys.executable).resolve()) == manifest["python"], "Python interpreter changed")
    require({name: importlib.metadata.version(name) for name in PACKAGES} == manifest["versions"], "Runtime packages changed")
    return manifest


def prepare():
    if (RUNS / "run_manifest.json").exists():
        load_manifest()
        verify_sample()
        print("Frozen configuration and sample verified; no rewriting.")
        return
    cfg = settings()
    os.environ.update(environment())
    rows = verify_sample()  # Preparation NEVER samples.
    base = ROOT / cfg["baseline"]
    config_dir = HERE / "configs"
    config_dir.mkdir(exist_ok=True)
    unique_base = config_dir / "oner_qa_qwen_hotpotqa_train_5000.jsonnet"
    atomic_text(unique_base, base.read_text())
    command = [sys.executable, "run.py", "write", str(unique_base.relative_to(ROOT)),
               "--instantiation_scheme", "oner_qa", "--prompt_set", "1", "--topk_values", "1-15",
               "--set_name", cfg["set_name"], "--llm_port_num", cfg["llm_port"], "--no_diff"]
    LOGS.mkdir(parents=True, exist_ok=True)
    execute(command, LOGS / "prepare.log", environment())
    baseline = configured(base)
    configs = {}
    for path in sorted((ROOT / "instantiated_configs").glob(unique_base.stem + "____*.jsonnet")):
        actual = configured(path)
        k = actual["models"]["generate_titles"]["retrieval_count"]
        require(k in cfg["ks"] and str(k) not in configs, "Unexpected/duplicate instantiated k")
        expected = json.loads(json.dumps(baseline))
        expected["models"]["generate_titles"]["retrieval_count"] = k
        require(actual == expected, "Formal configuration semantics changed at k=" + str(k))
        snapshot = config_dir / path.name
        atomic_text(snapshot, path.read_text())
        configs[str(k)] = {"path": str(snapshot.relative_to(ROOT)), "sha256": sha256(snapshot)}
    require(set(configs) == set(map(str, cfg["ks"])), "Incomplete run.py write sweep")
    from commaqa.inference.prompt_reader import read_prompt
    qa = baseline["models"]["answer_main_question"]
    selected_prompt = read_prompt(qa["prompt_file"], **qa["prompt_reader_args"])
    require(len(selected_prompt.split("\n\n\n")) == 15, "Formal Few-shot initialization changed")
    atomic_text(RUNS / "selected_prompt.txt", selected_prompt)
    # Six coverage examples only; this is a smoke subset, not a train/val split.
    smoke = [next(row for row in rows if row["level"] == level and row["type"] == kind)
             for level in ("easy", "medium", "hard") for kind in ("bridge", "comparison")]
    SMOKE.mkdir(parents=True, exist_ok=True)
    atomic_text(SMOKE / "smoke.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in smoke))
    datasets = {}
    for mode, path, selected in (("full", DATA / "train_5000.jsonl", rows), ("smoke", SMOKE / "smoke.jsonl", smoke)):
        datasets[mode] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                          "question_ids": [row["question_id"] for row in selected], "count": len(selected),
                          "distributions": distributions(selected)}
    files = code_files() + [base, unique_base, ROOT / qa["prompt_file"], RUNS / "selected_prompt.txt",
                           DATA / "sample_manifest.json", ROOT / cfg["source"]]
    files += [ROOT / item["path"] for item in configs.values()]
    files += [ROOT / item["path"] for item in datasets.values()]
    manifest = {
        "schema_version": 1, "created_at_utc": now(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip(),
        "python": str(Path(sys.executable).resolve()),
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
        "configs": configs, "datasets": datasets, "ext_vars": {
            "retriever": configured(ROOT / ".retriever_address.jsonnet"),
            "llm_base_url": environment()["OPENAI_BASE_URL"]},
        "set_names": {"full": cfg["set_name"], "smoke": cfg["smoke_set_name"]},
        "file_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "semantics_validation": "all 15 evaluated configurations equal formal baseline except retrieval_count",
        "cache": {"directory": os.path.expanduser("~/.cache/gpt3calls"), "policy": "original decorated cached_openai_call unchanged"},
        "usage_policy": "actual API response only; legacy cached usage is null; saved usage is never new billed usage",
    }
    atomic_json(RUNS / "run_manifest.json", manifest)
    print("Prepared 15 configurations; sample and all formal semantics verified. No inference started.")


def validate_records(manifest, mode, k, predictions, records):
    qids = manifest["datasets"][mode]["question_ids"]
    expected = binding(manifest, mode, k)
    require(isinstance(predictions, dict) and set(predictions) == set(qids), "Prediction IDs must exactly equal input IDs")
    require(all(isinstance(value, str) for value in predictions.values()), "Every prediction must be a string")
    require([row["qid"] for row in records] == qids, "Context IDs/order must exactly equal input IDs, without duplicates")
    for row in records:
        require(row["binding"] == expected and row["k"] == k, "Context/config/data binding differs")
        require(row["prediction"] == predictions[row["qid"]], "Prediction/context pair differs")
        require(row["context_sha256"] == text_hash(row["effective_context_text"]), "Effective context hash differs")
        require(row["effective_context_text"] == "\n\n".join(row["reader_contexts"]), "Reader context text differs")
        require(0 < len(row["reader_contexts"]) == len(row["raw_contexts"]) <= k, "Context count differs")
        # Check the original paragraph formatter, without retrieving anything.
        from commaqa.inference.ircot import para_to_text
        rendered = [para_to_text(context["title"], context["paragraph_text"], 350) for context in row["raw_contexts"]]
        require(row["reader_contexts"] == rendered, "Saved raw/rendered contexts differ")
        require(row["chain_sha256"] == text_hash(row["chain"]), "Chain hash differs")
        for field in ("prompt_sha256", "call_sha256"):
            require(bool(re.fullmatch("[0-9a-f]{64}", row[field])), "Invalid call/prompt hash")
        api = row["api"]
        require(api["finish_reasons"] is None or api["finish_reasons"] == ["stop"], "Abnormal generation finish")
        if api["usage"] is not None:
            usage = api["usage"]
            require(api["usage_source"] in ("api_response", "saved_api_response"), "Untrusted API usage source")
            require(all(type(usage.get(key)) is int and usage[key] >= 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")), "Invalid API usage")
            require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "API usage totals differ")
        else:
            require(api["usage_source"].startswith("unavailable_"), "Missing API usage provenance")


def inspect_complete(manifest, mode, k):
    prediction = prediction_path(manifest, mode, k)
    marker = prediction.parent / "complete.json"
    try:
        completed = read_json(marker)
        require(completed["binding"] == binding(manifest, mode, k), "Completion binding differs")
        require(completed["status"] == "complete" and completed["predict_exit_code"] == 0, "Unsuccessful attempt")
        required = {prediction.name, "contexts.jsonl", prediction.stem + "_chains.txt",
                    prediction.stem + "_time_taken.txt", prediction.stem + "_full_eval_path.txt",
                    prediction.stem + "_variable_replacements.json",
                    "config__" + prediction.name[len("prediction__"):-5] + ".jsonnet"}
        require(set(completed["output_hashes"]) == required, "Completion output list differs")
        for name, expected in completed["output_hashes"].items():
            require(sha256(prediction.parent / name) == expected, "Output hash differs: " + name)
        records = read_rows(prediction.parent / "contexts.jsonl")
        validate_records(manifest, mode, k, read_json(prediction), records)
        require((prediction.parent / (prediction.stem + "_chains.txt")).read_text() ==
                "".join(row["chain"] + "\n" for row in records), "Chain file differs from recorded predictions")
        return True, "all {} predictions and contexts verified".format(len(records))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, str(error)


def run_k(mode, k, allow_full=False):
    require(mode != "full" or allow_full, "Full inference requires --allow-full-run; never started by prepare/smoke")
    manifest = load_manifest()
    require(k in settings()["ks"], "k must be 1..15")
    base = RUNS / "predictions" / manifest["set_names"][mode]
    identity = base / "run_identity.json"
    if base.exists() and not identity.exists():
        require(not any(base.iterdir()), "Existing output set has no identity; refusing to overwrite")
    base.mkdir(parents=True, exist_ok=True)
    # Original predict.py still writes to predictions/<set_name>.
    legacy_base = ROOT / "predictions" / manifest["set_names"][mode]
    if not legacy_base.is_symlink():
        require(not legacy_base.exists(), "Legacy prediction path already exists")
        legacy_base.parent.mkdir(parents=True, exist_ok=True)
        legacy_base.symlink_to(os.path.relpath(base, legacy_base.parent), target_is_directory=True)
    require(legacy_base.resolve() == base.resolve(), "Legacy prediction link differs")
    with (base / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected_identity = {"manifest_sha256": sha256(RUNS / "run_manifest.json"), "mode": mode}
        if identity.exists():
            require(read_json(identity) == expected_identity, "Output set belongs to another experiment")
        else:
            atomic_json(identity, expected_identity)
        complete, reason = inspect_complete(manifest, mode, k)
        if complete:
            print("SKIP k={}: {}".format(k, reason))
            return
        print("RUN k={}: {}".format(k, reason), flush=True)
        prediction = prediction_path(manifest, mode, k)
        directory = prediction.parent
        directory.mkdir(parents=True, exist_ok=True)
        # Also refuse to disturb a surviving inference grandchild after SIGKILL.
        with (directory / ".inference.lock").open("a") as inference_lock:
            fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt_id = now().replace(":", "-") + "_" + uuid.uuid4().hex[:8]
        attempt = directory / "attempts" / attempt_id
        attempt.mkdir(parents=True)
        # Archive invalid/stale final artifacts before retrying. Cache is untouched.
        old = [path for path in directory.iterdir() if path.is_file() and path.name != ".inference.lock"]
        if old:
            archive = attempt / "previous_outputs"
            archive.mkdir()
            for path in old:
                shutil.move(str(path), str(archive / path.name))
        metadata = {"binding": binding(manifest, mode, k), "started_at_utc": now(), "status": "running",
                    "answer_policy": "allow_empty_string_v1"}
        atomic_json(attempt / "attempt.json", metadata)
        cfg = artifact_path(manifest["configs"][str(k)]["path"])
        data = artifact_path(manifest["datasets"][mode]["path"])
        command = [sys.executable, "predict.py", str(cfg.relative_to(ROOT)), str(data.relative_to(ROOT)),
                   "--set_name", manifest["set_names"][mode], "--llm_port_num", settings()["llm_port"],
                   "--skip-evaluation", "1", "--force", "--silent"]
        env = environment()
        env.update(HOTPOT5000_ATTEMPT=str(attempt), HOTPOT5000_MODE=mode, HOTPOT5000_K=str(k),
                   PYTHONPATH=str(SCRIPTS) + os.pathsep + str(ROOT))
        try:
            execute(command, attempt / "predict.log", env, lock.fileno())
            qids = manifest["datasets"][mode]["question_ids"]
            record_dir = attempt / "records"
            require({path.stem for path in record_dir.glob("*.json")} == set(qids), "Missing/unexpected per-question observations")
            records = [read_json(record_dir / (qid + ".json")) for qid in qids]
            validate_records(manifest, mode, k, read_json(prediction), records)
            atomic_text(directory / "contexts.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            backup_config = directory / ("config__" + prediction.name[len("prediction__"):-5] + ".jsonnet")
            require(sha256(backup_config) == manifest["configs"][str(k)]["sha256"], "Predict config snapshot differs")
            require((directory / (prediction.stem + "_full_eval_path.txt")).read_text() == str(data.relative_to(ROOT)), "Predict input path differs")
            require((directory / (prediction.stem + "_variable_replacements.json")).read_text() == "", "Unexpected variable replacements")
            outputs = [prediction, directory / "contexts.jsonl", backup_config] + [
                directory / (prediction.stem + suffix) for suffix in
                ("_chains.txt", "_time_taken.txt", "_full_eval_path.txt", "_variable_replacements.json")]
            marker = {"status": "complete", "binding": metadata["binding"], "predict_exit_code": 0,
                      "completed_at_utc": now(), "attempt": str(attempt.relative_to(ROOT)),
                      "answer_policy": "allow_empty_string_v1",
                      "empty_prediction_count": sum(not value.strip() for value in read_json(prediction).values()),
                      "output_hashes": {path.name: sha256(path) for path in outputs}}
            # Validate before publishing the completion marker.
            require((directory / (prediction.stem + "_chains.txt")).read_text() ==
                    "".join(row["chain"] + "\n" for row in records), "Chain aggregation differs")
            atomic_json(directory / "complete.json", marker)
            metadata.update(status="complete", ended_at_utc=now(), predict_exit_code=0)
            atomic_json(attempt / "attempt.json", metadata)
            print("COMPLETE k={} count={}".format(k, len(records)))
        except BaseException as error:
            # A killed parent may leave status=running. Only complete.json is authoritative.
            (directory / "complete.json").unlink(missing_ok=True)
            metadata.update(status="failed", ended_at_utc=now(), error=repr(error))
            atomic_json(attempt / "attempt.json", metadata)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check-data", "status", "check", "run-k"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--k", type=int)
    parser.add_argument("--allow-full-run", action="store_true")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError("Received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    os.chdir(str(ROOT))
    os.environ.update(environment())
    if args.command == "prepare":
        RUNS.mkdir(parents=True, exist_ok=True)
        with (RUNS / ".prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare()
    elif args.command == "check-data":
        verify_sample()
        print("PASS: frozen sample and source consistency")
    elif args.command == "run-k":
        require(args.k is not None, "--k is required")
        run_k(args.mode, args.k, args.allow_full_run)
    else:
        manifest = load_manifest()
        ks = [args.k] if args.k is not None else settings()["smoke_ks" if args.mode == "smoke" else "ks"]
        states = []
        for k in ks:
            done, reason = inspect_complete(manifest, args.mode, k)
            states.append(done)
            prediction = prediction_path(manifest, args.mode, k)
            attempts = sorted((prediction.parent / "attempts").glob("*"))
            partial = len(list((attempts[-1] / "records").glob("*.json"))) if attempts else 0
            print("k={} {} {}/{} {}".format(k, "COMPLETE" if done else "INCOMPLETE", manifest["datasets"][args.mode]["count"] if done else partial,
                  manifest["datasets"][args.mode]["count"], reason))
        if args.command == "check":
            require(all(states), "Some requested k values are incomplete")


if __name__ == "__main__":
    main()
