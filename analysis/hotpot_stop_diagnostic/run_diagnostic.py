"""Prepare exactly 40 fixed samples, then invoke the repository's run.py."""
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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-name", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    output = ROOT / "predictions" / args.set_name
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "processed_data/hotpotqa/dev_1000_subsampled.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    counts = Counter()
    selected = []
    for row in rows:
        if row["type"] in ("bridge", "comparison") and counts[row["type"]] < 20:
            selected.append(row)
            counts[row["type"]] += 1
    assert counts == {"bridge": 20, "comparison": 20}
    dataset = ROOT / "processed_data/hotpotqa" / (args.set_name + ".jsonl")
    assert not dataset.exists()
    data_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    dataset.write_text(data_text)
    (output / "samples.jsonl").write_text(data_text)
    session = requests.Session()
    session.trust_env = False
    response = session.get("http://127.0.0.1:8002/v1/models", timeout=20)
    response.raise_for_status()
    models = response.json()
    assert any(model["id"] == "Qwen2.5-7B-Instruct" for model in models["data"])
    config = ROOT / "base_configs/oner_qa_qwen_hotpotqa_stop_diagnostic.jsonnet"
    ext = {"RETRIEVER_HOST": "http://localhost", "RETRIEVER_PORT": "8000"}
    new_config = json.loads(_jsonnet.evaluate_file(str(config), ext_vars=ext))
    history = []
    for k in (1, 8, 15):
        directory = ROOT / "predictions/dev_1000" / ("oner_qa_qwen_hotpotqa____prompt_set_1___bm25_retrieval_count__{}___distractor_count__1".format(k))
        historical_config = next(directory.glob("config__*.jsonnet"))
        old_config = json.loads(_jsonnet.evaluate_file(str(historical_config), ext_vars=ext))
        comparable = json.loads(json.dumps(new_config))
        assert comparable["models"]["answer_main_question"].pop("stop") == []
        comparable["models"]["generate_titles"]["retrieval_count"] = k
        assert comparable == old_config, "Only stop and k may differ from historical config"
        paths = [historical_config, next(directory.glob("prediction__*subsampled.json")), next(directory.glob("prediction__*chains.txt"))]
        history.append({"k": k, "files": [{"path": str(path), "sha256": sha256(path)} for path in paths]})
    env = os.environ.copy()
    env.update({"PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""), "OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": "http://127.0.0.1:8002/v1", "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "KBQA_STOP_DIAGNOSTIC_DIR": str(output), "PYTHONPATH": str(Path(__file__).parent) + os.pathsep + str(ROOT)})
    common = [str(config.relative_to(ROOT)), "--instantiation_scheme", "oner_qa", "--prompt_set", "1", "--set_name", args.set_name, "--topk_values", "1,8,15", "--llm_port_num", "8002"]
    # run.py verify currently references an undefined global `args`. Its
    # generation path is unaffected; validate Jsonnet above and keep it intact.
    commands = [[sys.executable, "run.py", "write"] + common + ["--no_diff"], [sys.executable, "run.py", "predict"] + common + ["--evaluation_path", str(dataset.relative_to(ROOT)), "--silent"]]
    manifest = {"started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(), "source": str(source), "source_sha256": sha256(source), "sample_selection": "First 20 bridge and first 20 comparison in source order; no outcome-based selection", "question_ids": [row["question_id"] for row in selected], "counts": dict(counts), "dataset": str(dataset), "dataset_sha256": sha256(dataset), "models": models, "python": sys.executable, "versions": {name: importlib.metadata.version(name) for name in ("openai", "httpx", "transformers", "torch", "diskcache")}, "historical_inputs": history, "generation_cache": "cached_openai_call.__wrapped__; bypass old and new generation cache", "environment_adaptations": {key: env[key] for key in ("PATH", "OPENAI_BASE_URL", "NO_PROXY", "HF_HUB_OFFLINE", "PYTHONPATH")}, "verification_adaptation": "run.py verify failed before inference: run.py:853 references undefined args; no repository fix, Jsonnet is validated in this driver instead", "commands": commands, "functional_change": "stop=[] only; historical max_tokens=300 and prompt budget=4000 retained"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": dict(counts), "commands": commands}, indent=2), flush=True)
    for command in commands:
        subprocess.run(command, env=env, check=True)
    records = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
    assert len(records) == 120
    assert Counter(row["k"] for row in records) == {1: 40, 8: 40, 15: 40}
    assert all(len(row["api_requests"]) == len(row["api_responses"]) == 1 for row in records)
    assert all(row["api_requests"][0]["body"]["stop"] == [] for row in records)
    assert all(row["api_responses"][0]["status"] == 200 for row in records)
    for group in history:
        for file in group["files"]:
            assert sha256(Path(file["path"])) == file["sha256"], "Historical artifacts changed"
    manifest["completed_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    manifest["validation"] = {"records": 120, "wire_requests": 120, "all_stop_empty": True, "historical_artifacts_unchanged": True}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
