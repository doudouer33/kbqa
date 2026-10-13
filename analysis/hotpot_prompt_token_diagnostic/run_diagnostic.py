"""Reuse the saved 40 samples; preflight full prompts, then run the original pipeline."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _jsonnet
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from commaqa.inference.prompt_reader import read_prompt, get_tokenizer

CONFIG = ROOT / "base_configs/oner_qa_qwen_hotpotqa_prompt_token_diagnostic.jsonnet"
BASELINE = ROOT / "predictions/hotpot_stop_40_20261013_1050"
SAFETY_MARGIN = 256
EXT = {"RETRIEVER_HOST": "http://localhost", "RETRIEVER_PORT": "8000"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def evaluated(path):
    return json.loads(_jsonnet.evaluate_file(str(path), ext_vars=EXT))


def check_config(config, old):
    """Only the prompt path and three budget fields may change."""
    comparable = json.loads(json.dumps(config))
    current = comparable["models"]["answer_main_question"]
    previous = old["models"]["answer_main_question"]
    for key in ("prompt_file", "max_tokens", "model_tokens_limit"):
        current[key] = previous[key]
    for key in ("model_length_limit", "estimated_generation_length"):
        current["prompt_reader_args"][key] = previous["prompt_reader_args"][key]
    assert comparable == old, "Change outside prompt/token budget scope"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-name", default="hotpot_prompt_token_40_20261013")
    args = parser.parse_args()
    assert Path(args.set_name).name == args.set_name
    os.chdir(ROOT)
    os.environ["HF_HUB_OFFLINE"] = "1"
    output = ROOT / "predictions" / args.set_name
    dataset = ROOT / "processed_data/hotpotqa" / (args.set_name + ".jsonl")
    assert not output.exists() and not dataset.exists(), "Use a new set-name; never overwrite results"
    output.mkdir(parents=True)

    sample_bytes = (BASELINE / "samples.jsonl").read_bytes()
    samples = [json.loads(line) for line in sample_bytes.decode().splitlines()]
    ids = [row["question_id"] for row in samples]
    assert ids == json.loads((BASELINE / "manifest.json").read_text())["question_ids"]
    counts = Counter(row["type"] for row in samples)
    assert counts == {"bridge": 20, "comparison": 20} and len(set(ids)) == 40
    dataset.write_bytes(sample_bytes)
    (output / "samples.jsonl").write_bytes(sample_bytes)

    protected = list(BASELINE.rglob("*")) + [ROOT / "processed_data/hotpotqa/dev_1000_subsampled.jsonl"]
    for folder in ("commaqa", "metrics", "official_evaluation", "analysis/hotpot_stop_diagnostic"):
        protected.extend((ROOT / folder).rglob("*.py"))
    protected.extend(ROOT / name for name in (
        "evaluate.py", "evaluate_per_question.py", "run.py", "predict.py",
        "prompts/hotpotqa/gold_with_1_distractors_context_cot_qa_codex.txt",
        "base_configs/oner_qa_qwen_hotpotqa_stop_diagnostic.jsonnet",
        "analysis/hotpot_stop_diagnostic/report_20261013.md",
    ))
    protected.extend((ROOT / "predictions/dev_1000").glob("oner_qa_qwen_hotpotqa____*retrieval_count__*/*"))
    fingerprints = {str(path): sha256(path) for path in protected if path.is_file()}

    session = requests.Session()
    session.trust_env = False
    response = session.get("http://127.0.0.1:8002/v1/models", timeout=20)
    response.raise_for_status()
    models = response.json()
    model = next(row for row in models["data"] if row["id"] == "Qwen2.5-7B-Instruct")
    server_limit = model["max_model_len"]
    assert server_limit == 32768, "Reassess configuration if the service context limit changed"
    version = session.get("http://127.0.0.1:8002/version", timeout=20).json()

    config = evaluated(CONFIG)
    qa = config["models"]["answer_main_question"]
    old_config_path = next(BASELINE.glob("*retrieval_count__1___*/config__*.jsonnet"))
    check_config(config, evaluated(old_config_path))
    assert qa["stop"] == [] and qa["max_tokens"] == 512 and qa["model_tokens_limit"] == 16384
    assert qa["prompt_reader_args"]["estimated_generation_length"] == 512
    assert qa["prompt_reader_args"]["model_length_limit"] == 16384
    new_prompt = read_prompt(file_path=qa["prompt_file"], **qa["prompt_reader_args"])
    old_qa = evaluated(old_config_path)["models"]["answer_main_question"]
    old_args = dict(old_qa["prompt_reader_args"], model_length_limit=32768)
    full_original = read_prompt(file_path=old_qa["prompt_file"], **old_args)
    instructions = new_prompt.split("\n\nWikipedia Title:", 1)[0]
    assert instructions.startswith("Answer the question using the provided context.")
    assert new_prompt == instructions + "\n\n" + full_original, "Original selected examples must be unchanged"
    fewshot_ids = qa["prompt_reader_args"]["filter_by_key_values"]["qid"]
    assert len(fewshot_ids) == len(full_original.split("\n\n\n")) == len(new_prompt.split("\n\n\n")) == 15
    tokenizer = get_tokenizer(qa["tokenizer_model_name"])
    init_lengths = [len(tokenizer.tokenize(text)) for text in new_prompt.split("\n\n\n")]
    assert sum(init_lengths) + max(init_lengths) + 512 <= 16384

    logs = [json.loads(line) for line in (BASELINE / "diagnostics.jsonl").read_text().splitlines()]
    assert [(row["k"], row["qid"]) for row in logs] == [(k, qid) for k in (1, 8, 15) for qid in ids]
    preflight_rows = []
    for row in logs:
        prompt = new_prompt + "\n\n\n" + row["prompt_fit"]["test_example_before"]
        local_count = len(tokenizer.tokenize(prompt))
        response = session.post("http://127.0.0.1:8002/tokenize", json={
            "model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True, "add_special_tokens": False,
        }, timeout=30)
        response.raise_for_status()
        count = response.json()["count"]
        assert local_count + 512 + SAFETY_MARGIN <= 16384, (row["k"], row["qid"], local_count)
        assert count + 512 + SAFETY_MARGIN <= server_limit, (row["k"], row["qid"], count)
        preflight_rows.append({"k": row["k"], "qid": row["qid"], "local_prompt_tokens": local_count,
                               "api_prompt_tokens": count, "chat_template_overhead": count - local_count,
                               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    preflight = {"baseline_dir": str(BASELINE), "server_max_model_len": server_limit, "safety_margin": SAFETY_MARGIN,
                 "fewshot_ids": fewshot_ids, "selected_prompt": new_prompt, "instructions": instructions,
                 "initialization_estimate_tokens": sum(init_lengths) + max(init_lengths) + 512,
                 "requests": preflight_rows}
    (output / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n")
    (output / "prompt_snapshot.txt").write_bytes((ROOT / qa["prompt_file"]).read_bytes())
    (output / "base_config_snapshot.jsonnet").write_bytes(CONFIG.read_bytes())

    env = os.environ.copy()
    env.update({"PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
                "OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": "http://127.0.0.1:8002/v1",
                "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1",
                "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                "KBQA_PROMPT_TOKEN_DIAGNOSTIC_DIR": str(output),
                "PYTHONPATH": str(Path(__file__).parent) + os.pathsep + str(ROOT)})
    env.pop("KBQA_STOP_DIAGNOSTIC_DIR", None)
    common = [str(CONFIG.relative_to(ROOT)), "--instantiation_scheme", "oner_qa", "--prompt_set", "1",
              "--set_name", args.set_name, "--topk_values", "1,8,15", "--llm_port_num", "8002"]
    commands = [[sys.executable, "run.py", "write"] + common + ["--no_diff"],
                [sys.executable, "run.py", "predict"] + common + ["--evaluation_path", str(dataset.relative_to(ROOT)), "--silent"]]
    manifest = {"started_at": now(), "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                "baseline_dir": str(BASELINE), "sample_selection": "Byte-for-byte copy of previous samples.jsonl; no resampling",
                "sample_sha256": sha256(dataset), "question_ids": ids, "counts": dict(counts), "models": models, "vllm_version": version,
                "python": sys.executable, "versions": {name: importlib.metadata.version(name) for name in ("openai", "httpx", "transformers", "torch", "diskcache")},
                "commands": commands, "protected_sha256": fingerprints, "config_sha256": sha256(CONFIG),
                "prompt_sha256": sha256(ROOT / qa["prompt_file"]), "generation_cache": "Use previous hooks: cached_openai_call.__wrapped__; no generation cache",
                "environment": {key: env[key] for key in ("OPENAI_BASE_URL", "NO_PROXY", "HF_HUB_OFFLINE", "PYTHONPATH")},
                "functional_changes": ["Strict single-line CoT output instructions", "model_tokens_limit=16384", "estimated_generation_length=max_tokens=512"],
                "preflight_max_api_prompt_tokens": max(row["api_prompt_tokens"] for row in preflight_rows)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "preflight_requests": len(preflight_rows), "maximum_api_prompt_tokens": manifest["preflight_max_api_prompt_tokens"],
                      "initialization_estimate_tokens": preflight["initialization_estimate_tokens"], "fewshot_count": 15, "server_max_model_len": server_limit}, indent=2), flush=True)
    try:
        with (output / "run.log").open("w") as log:
            for command in commands:
                print("Running: " + " ".join(command), flush=True)
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        records = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
        assert [(row["k"], row["qid"]) for row in records] == [(k, qid) for k in (1, 8, 15) for qid in ids]
        sample_map = {row["question_id"]: row for row in samples}
        for row in records:
            assert "error" not in row and len(row["api_requests"]) == len(row["api_responses"]) == 1
            assert row["api_responses"][0]["status"] == 200
            assert not row["prompt_fit"]["changed"] and row["prompt_fit"]["after_demonstrations"] == 15
            assert row["api_requests"][0]["body"]["max_tokens"] == 512
            sample = sample_map[row["qid"]]
            raw = row["api_responses"][0]["body"]["choices"][0]["message"]["content"]
            row.update({"type": sample["type"], "gold": [span for answer in sample["answers_objects"] for span in answer["spans"]],
                        "fewshot_ids": fewshot_ids, "initialization_cropped": False, "runtime_cropped": False,
                        "instructions_present": instructions in row["api_requests"][0]["body"]["messages"][0]["content"],
                        "raw_has_newline": "\n" in raw or "\r" in raw, "has_exact_answer_marker": "So the answer is:" in raw})
        # Enrich only this new experiment's logs, retaining original payloads.
        (output / "diagnostics.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        for k in (1, 8, 15):
            saved_config = evaluated(next(output.glob("*retrieval_count__{}___*/config__*.jsonnet".format(k))))
            previous = evaluated(next(BASELINE.glob("*retrieval_count__{}___*/config__*.jsonnet".format(k))))
            check_config(saved_config, previous)
        assert all(sha256(Path(path)) == digest for path, digest in fingerprints.items()), "Protected source or historical result changed"
        manifest["completed_at"] = now()
        manifest["validation"] = {"records": 120, "wire_requests": 120, "responses_http_200": 120, "all_fewshot_counts_15": True,
                                  "cropped_prompts": 0, "instructions_present": 120, "protected_files_unchanged": True,
                                  "saved_configs_only_prompt_and_budget_differ": True}
    except BaseException as error:
        manifest["failed_at"] = now()
        manifest["error"] = repr(error)
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest["validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
