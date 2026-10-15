"""Gate the sequential 1000x15 run on a real 4+4, k=1/15 smoke test.

This orchestrates run.py write/predict; it does not implement question answering
or a second analysis framework. Use the same command with --resume to recover.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
import statistics
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

import _jsonnet
import requests
from analysis.hotpot_formal_runner.storage import atomic_json, atomic_text, sha256, load_jsonl, text_hash
from analysis.analyze_hotpot_500_500 import load_samples, score_predictions, GROUPS
from commaqa.inference.prompt_reader import read_prompt, get_tokenizer
from run import instatiate_config
from lib import get_retriever_address

CONFIG = ROOT / "base_configs/oner_qa_qwen_hotpotqa_formal.jsonnet"
DATA = ROOT / "processed_data/hotpotqa/dev_1000_subsampled.jsonl"
PYTHON = Path("/home/dengxin/miniconda3/envs/adaptiverag/bin/python")
ARTIFACTS = ("fixed_k_results.csv", "oracle_distribution.csv", "per_question_curves.jsonl", "hotpot_basic_statistics.json", "fixed_k_f1.png", "token_curve.png", "bridge_oracle_distribution.png", "comparison_oracle_distribution.png", "combined_oracle_distribution.png", "hotpot_500_500_analysis.md")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def environment(manifest=None, input_path=None):
    env = dict(os.environ)
    env.update(PATH=str(PYTHON.parent) + os.pathsep + env.get("PATH", ""), OPENAI_API_KEY="EMPTY",
        OPENAI_BASE_URL="http://127.0.0.1:8002/v1", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONHASHSEED="0", TOKENIZERS_PARALLELISM="false")
    if manifest:
        env["KBQA_FORMAL_MANIFEST"] = str(manifest)
        env["KBQA_FORMAL_INPUT_SHA256"] = sha256(input_path)
        env["PYTHONPATH"] = str(ROOT / "analysis/hotpot_formal_runner") + os.pathsep + str(ROOT)
    return env


def execute(command, log_path, env):
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n{} COMMAND {}\n".format(now(), json.dumps(command)))
        stream.flush()
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_args(command, set_name, ks):
    return [str(PYTHON), "run.py", command, str(CONFIG), "--instantiation_scheme", "oner_qa",
        "--prompt_set", "1", "--topk_values", ks, "--llm_port_num", "8002", "--set_name", set_name]


def protected_hashes():
    files = []
    for folder in ("processed_data/hotpotqa", "predictions", "predictions_old", "official_evaluation", "prompts"):
        files.extend((ROOT / folder).rglob("*"))
    for folder in ("analysis/hotpot_500_500", "analysis/hotpot_stop_diagnostic", "analysis/hotpot_prompt_token_diagnostic", "analysis/hotpot_unicode_json_diagnostic"):
        files.extend((ROOT / folder).rglob("*"))
    files.extend((ROOT / "analysis/hotpot_multiline_answer_diagnostic").glob("report*"))
    files.extend((ROOT / "base_configs").glob("*.jsonnet"))
    files.extend([ROOT / "commaqa/inference/ircot.py", ROOT / "commaqa/models/gpt3generator.py", ROOT / "evaluate.py", ROOT / "evaluate_per_question.py"])
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(files))
        if path.is_file() and "__pycache__" not in path.parts and path != CONFIG}


def choose_smoke(rows):
    chosen = []
    for kind in ("bridge", "comparison"):
        candidates = [row for row in rows if row["type"] == kind]
        yn = [row for row in candidates if row["answers_objects"][0]["spans"][0].lower() in ("yes", "no")]
        entity = [row for row in candidates if row not in yn]
        group = []
        if yn:
            group.append(yn[0])
        for row in (min(entity, key=lambda r: len(r["question_text"])), max(entity, key=lambda r: len(r["question_text"])), entity[0]):
            if row not in group:
                group.append(row)
        for row in candidates:
            if len(group) == 4:
                break
            if row not in group:
                group.append(row)
        chosen.extend(group)
    return chosen


def prepare(args, report_dir, formal_dir, smoke_dir):
    os.environ.update(environment())
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError("Run in the requested adaptiverag environment")
    samples = load_samples(DATA)
    rows = list(samples.values())
    assert Counter(row["type"] for row in rows) == {"bridge": 500, "comparison": 500}
    dataset_manifest = json.loads(DATA.with_name("dev_1000_sample_manifest.json").read_text())
    assert dataset_manifest["question_ids"] == list(samples)
    raw = {row["_id"]: row for row in json.loads((ROOT / "raw_data/hotpotqa/hotpot_dev_distractor_v1.json").read_text())}
    for row in rows:
        assert raw[row["question_id"]]["question"] == row["question_text"]
        assert raw[row["question_id"]]["answer"] == row["answers_objects"][0]["spans"][0]
        assert raw[row["question_id"]]["type"] == row["type"]
    retriever = get_retriever_address()
    ext = {"RETRIEVER_HOST": str(retriever["host"]), "RETRIEVER_PORT": str(retriever["port"])}
    config = json.loads(_jsonnet.evaluate_file(str(CONFIG), ext_vars=ext))
    qa = config["models"]["answer_main_question"]
    fewshot_ids = qa["prompt_reader_args"]["filter_by_key_values"]["qid"]
    assert len(fewshot_ids) == len(set(fewshot_ids)) == 15
    assert not set(fewshot_ids) & set(samples)
    prompt = read_prompt(qa["prompt_file"], **qa["prompt_reader_args"])
    assert len(prompt.split("\n\n\n")) == 15
    tokenizer = get_tokenizer(qa["tokenizer_model_name"])
    session = requests.Session()
    session.trust_env = False
    response = session.get("http://127.0.0.1:8002/v1/models", timeout=15)
    response.raise_for_status()
    model = next(m for m in response.json()["data"] if m["id"] == "Qwen2.5-7B-Instruct")
    assert model["max_model_len"] >= 16384
    smoke_rows = choose_smoke(rows)
    smoke_data = smoke_dir / "hotpotqa_smoke.jsonl"
    smoke_dir.mkdir()
    smoke_data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in smoke_rows))
    endpoint = ext["RETRIEVER_HOST"].rstrip("/") + ":" + ext["RETRIEVER_PORT"] + "/retrieve"
    response = session.post(endpoint, json={"retrieval_method": "retrieve_from_elasticsearch", "query_text": rows[0]["question_text"], "max_hits_count": 1, "corpus_name": "hotpotqa", "document_type": "title_paragraph_text"}, timeout=30)
    response.raise_for_status()
    assert response.json()["retrieval"]
    old_content = subprocess.check_output(["git", "show", "13f9a52dd283c2346fa113419cf703c8706f53aa:base_configs/oner_qa_qwen_hotpotqa_multiline_candidate.jsonnet"], cwd=str(ROOT)).decode()
    assert CONFIG.read_text() == old_content, "Formal rename must preserve the entire verified config"
    old_dir = report_dir / "candidate_semantics"
    old_dir.mkdir()
    for k in range(1, 16):
        variables = {"bm25_retrieval_count": str(k)}
        old = json.loads(_jsonnet.evaluate_snippet("candidate", instatiate_config(old_content, variables), ext_vars=ext))
        new = json.loads(_jsonnet.evaluate_snippet("formal", instatiate_config(CONFIG.read_text(), variables), ext_vars=ext))
        assert old == new
        atomic_json(old_dir / ("k_{}.json".format(k)), new)
    execute(run_args("write", args.set_name, "1-15") + ["--no_diff"], report_dir / "write.log", environment())
    configs = {}
    for path in (ROOT / "instantiated_configs").glob(CONFIG.stem + "____*.jsonnet"):
        evaluated = json.loads(_jsonnet.evaluate_file(str(path), ext_vars=ext))
        k = evaluated["models"]["generate_titles"]["retrieval_count"]
        assert evaluated == json.loads((old_dir / ("k_{}.json".format(k))).read_text())
        assert k not in configs
        configs[k] = path
    assert set(configs) == set(range(1, 16))
    snapshots = report_dir / "configs"
    snapshots.mkdir()
    for k, path in configs.items():
        (snapshots / path.name).write_bytes(path.read_bytes())
    atomic_text(report_dir / "selected_prompt.txt", prompt)
    atomic_json(report_dir / "protected_hashes.json", protected_hashes())
    formal_dir.mkdir()
    manifest = {"created_at": now(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "python": sys.executable, "versions": {name: importlib.metadata.version(name) for name in ("openai", "httpx", "transformers", "torch", "diskcache", "jsonnet", "matplotlib")},
        "dataset": str(DATA), "dataset_sha256": sha256(DATA), "question_ids": list(samples), "types": dict(Counter(row["type"] for row in rows)),
        "dataset_manifest_order_match": True, "raw_gold_match": True, "fewshot_ids": fewshot_ids, "fewshot_overlap": [],
        "prompt_file": qa["prompt_file"], "prompt_sha256": sha256(ROOT / qa["prompt_file"]), "selected_prompt_sha256": text_hash(prompt),
        "fewshot_tokens": len(tokenizer.encode(prompt)), "model": model, "ext_vars": ext, "formal_config_sha256": sha256(CONFIG),
        "config_hashes": {str(k): sha256(path) for k, path in configs.items()}, "config_paths": {str(k): str(path) for k, path in configs.items()},
        "smoke_data": str(smoke_data), "smoke_data_sha256": sha256(smoke_data), "smoke_qids": [r["question_id"] for r in smoke_rows],
        "smoke_samples": [{"qid": r["question_id"], "type": r["type"], "question": r["question_text"], "answer": r["answers_objects"][0]["spans"][0]} for r in smoke_rows],
        "scoring": "official HotpotQA", "oracle_rule": "max official F1; smallest k on ties", "generation_cache": "disabled; matching durable API responses only",
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in [ROOT / "run.py", ROOT / "predict.py", ROOT / "lib.py", ROOT / "evaluate.py", ROOT / "analysis/analyze_hotpot_500_500.py"] + sorted((ROOT / "analysis/hotpot_formal_runner").glob("*.py")) + sorted((ROOT / "commaqa").rglob("*.py"))}}
    atomic_json(report_dir / "manifest.json", manifest)
    return manifest


def validate_k(manifest, input_path, set_name, k):
    config_path = Path(manifest["config_paths"][str(k)])
    assert sha256(config_path) == manifest["config_hashes"][str(k)]
    directory = ROOT / "predictions" / set_name / config_path.stem
    predictions = list(directory.glob("prediction__*.json"))
    predictions = [p for p in predictions if not p.stem.endswith("_variable_replacements")]
    assert len(predictions) == 1
    path = predictions[0]
    samples = load_samples(input_path)
    prediction = json.loads(path.read_text())
    usage = load_jsonl(directory / "token_usage.jsonl")
    assert list(prediction) == list(samples), "Final prediction order/content differs from dataset"
    assert [row["qid"] for row in usage] == list(samples), "Token journal order/content differs from dataset"
    assert len(usage) == len(samples)
    for row in usage:
        assert row["k"] == k and row["prediction"] == prediction[row["qid"]]
        assert row["config_sha256"] == manifest["config_hashes"][str(k)]
        assert row["prompt_cropped"] is False and row["fewshot_count"] == 15
        assert row["finish_reason"] == "stop"
        assert row["prompt_tokens"] + 512 <= 16384
        assert row["total_tokens"] == row["prompt_tokens"] + row["completion_tokens"]
        assert isinstance(row["completion_tokens"], int) and row["latency_seconds"] > 0
        checkpoint = json.loads((directory / "checkpoints" / (row["qid"] + ".json")).read_text())
        assert checkpoint["tokens"] == row and checkpoint["result"][1] == prediction[row["qid"]]
    scores = score_predictions(samples, {k: path})
    metric = list(directory.glob("official_evaluation_metrics__*.json"))
    assert len(metric) == 1
    official = json.loads(metric[0].read_text())
    f1 = statistics.mean(item["scores"][k]["f1"] for item in scores.values())
    em = statistics.mean(item["scores"][k]["em"] for item in scores.values())
    assert abs(official["f1"] - f1) <= 0.0000051 and abs(official["em"] - em) <= 0.0000051, "Official CLI/function scores differ beyond legacy five-decimal rounding"
    # Existing metrics filename is retained; this run stores official scores in it.
    metrics_path = directory / metric[0].name.replace("official_evaluation_metrics__", "evaluation_metrics__")
    if metrics_path.exists():
        assert json.loads(metrics_path.read_text()) == official
    else:
        atomic_json(metrics_path, official)
    return {"k": k, "count": len(samples), "f1": f1, "em": em, "prediction_path": str(path),
        "token_path": str(directory / "token_usage.jsonl"), "finish_reasons": dict(Counter(row["finish_reason"] for row in usage)),
        "cropped": sum(row["prompt_cropped"] for row in usage), "format_invalid": sum(not row["format_valid"] for row in usage),
        "average_prompt_tokens": statistics.mean(row["prompt_tokens"] for row in usage), "average_completion_tokens": statistics.mean(row["completion_tokens"] for row in usage),
        "average_total_tokens": statistics.mean(row["total_tokens"] for row in usage), "average_latency_seconds": statistics.mean(row["latency_seconds"] for row in usage)}


def predict_or_recover(manifest, report_dir, input_path, set_name, k, log_path):
    """A crash after final prediction/evaluation must not repeat inference."""
    directory = ROOT / "predictions" / set_name / Path(manifest["config_paths"][str(k)]).stem
    path = directory / ("prediction__hotpotqa_to_hotpotqa__" + input_path.stem + ".json")
    if path.exists():
        # The standard writer can be interrupted. Never treat partial JSON as
        # complete; replay validated checkpoints through the original writer.
        try:
            predictions = json.loads(path.read_text())
            complete_file = list(predictions) == list(load_samples(input_path)) and all(predictions.values())
        except (ValueError, TypeError):
            complete_file = False
        if complete_file:
            if not list(directory.glob("official_evaluation_metrics__*.json")):
                execute(run_args("evaluate", set_name, str(k)) + ["--evaluation_path", str(input_path), "--official"], log_path, environment())
            return validate_k(manifest, input_path, set_name, k)
        # Preserve the interrupted final file and let checkpoint replay rebuild
        # it. Historical or foreign answers are never imported into checkpoints.
        backup = path.with_name(path.name + ".interrupted_" + str(time.time_ns()))
        path.rename(backup)
    execute(run_args("predict", set_name, str(k)) + ["--evaluation_path", str(input_path), "--silent", "--official_only"], log_path, environment(report_dir / "manifest.json", input_path))
    return validate_k(manifest, input_path, set_name, k)


def rehearse_recovery(manifest, report_dir, smoke_name):
    """Recover three completed and five response-only questions without generation."""
    config = Path(manifest["config_paths"]["1"])
    source = ROOT / "predictions" / smoke_name / config.stem
    target = report_dir / "recovery_rehearsal"
    if target.exists():
        # Preserve any failed rehearsal and retry in a unique directory.
        target = report_dir / ("recovery_rehearsal_" + str(time.time_ns()))
    target.mkdir()
    (target / "checkpoints").mkdir()
    shutil.copytree(str(source / "api_responses"), str(target / "api_responses"))
    shutil.copyfile(str(source / "request_events.jsonl"), str(target / "request_events.jsonl"))
    for qid in manifest["smoke_qids"][:3]:
        shutil.copyfile(str(source / "checkpoints" / (qid + ".json")), str(target / "checkpoints" / (qid + ".json")))
    atomic_text(target / "token_usage.jsonl", '{"interrupted_append":')
    smoke_data = Path(manifest["smoke_data"])
    env = environment(report_dir / "manifest.json", smoke_data)
    env.update(RETRIEVER_HOST=manifest["ext_vars"]["RETRIEVER_HOST"], RETRIEVER_PORT=manifest["ext_vars"]["RETRIEVER_PORT"], LLM_SERVER_HOST="http://127.0.0.1", LLM_SERVER_PORT="8002")
    prediction = target / "prediction.json"
    execute([str(PYTHON), "-m", "commaqa.inference.configurable_inference", "--config", str(config), "--input", str(smoke_data), "--output", str(prediction), "--silent"], report_dir / "recovery_rehearsal.log", env)
    expected = json.loads(next(source.glob("prediction__*smoke.json")).read_text())
    assert json.loads(prediction.read_text()) == expected
    usage = load_jsonl(target / "token_usage.jsonl")
    assert len(usage) == 8 and [row["qid"] for row in usage] == manifest["smoke_qids"]
    assert sum(row["recovered_api_response"] for row in usage) == 5
    assert (target / "request_events.jsonl").read_bytes() == (source / "request_events.jsonl").read_bytes()
    atomic_json(report_dir / "recovery_validation.json", {"completed_reused": 3, "saved_responses_recovered": 5, "new_model_calls": 0, "token_journal_repaired": True, "prediction_equal": True, "unique_pairs": 8})


def report(report_dir, manifest, state, formal_dir, smoke_dir):
    lines = ["# HotpotQA 500+500 正式实验报告", "", "- 更新时间：" + now(), "- 状态：" + state["status"],
        "- Git 基线：`" + manifest["git_commit"] + "`；修改未 commit/push。", "- Python：`" + manifest["python"] + "`。",
        "- 依赖版本：`" + json.dumps(manifest["versions"]) + "`。", "- 模型：Qwen2.5-7B-Instruct；vLLM http://127.0.0.1:8002/v1；BM25。",
        "- 正式配置：`" + str(CONFIG) + "`；与基线 candidate 在 k=1～15 语义一致。",
        "- 数据：`" + str(DATA) + "`；500 bridge + 500 comparison；1000 唯一 qid；manifest 顺序及原始官方答案均匹配。",
        "- 数据 SHA-256：`" + manifest["dataset_sha256"] + "`。",
        "- Prompt SHA-256：`" + manifest["prompt_sha256"] + "`；固定 15 Few-shot，与评测无重叠。",
        "- 配置：model_tokens_limit=16384；max_tokens=estimated_generation_length=512；stop=[]；temperature=0；top_p=1。",
        "- 真实 API usage、结束原因、耗时、Prompt hash 和原始生成文本逐题落盘；上下文快照与实际 Prompt 的上下文 hash 一致。",
        "- 启动时恢复原子 checkpoint，重建 Token journal；已完成题不再调用模型；已保存响应在 Prompt/config 一致时恢复提取。",
        "- 代理异常：仅为本次实验进程设置 NO_PROXY，并为模型 HTTP 客户端关闭环境代理。未重启/修改服务。",
        "- 子进程：run.py/predict.py 传播失败退出状态，最终文件另做逐题完整性核验。", "",
        "## Smoke Test", "", "- 结果：" + ("通过，16/16 真实调用完成；无裁剪、异常缓存或缺失记录。" if state.get("smoke_passed") else "尚未通过；不得启动正式推理。"),
        "- 目录：`" + str(smoke_dir) + "`。", "- 回归测试：" + state.get("regression_tests", "待运行") + "。", "",
        "| qid | type | question | gold |", "|---|---|---|---|"]
    for row in manifest["smoke_samples"]:
        lines.append("| {} | {} | {} | {} |".format(row["qid"], row["type"], row["question"].replace("|", "\\|"), row["answer"]))
    lines.extend(["", "## 正式执行", "", "- 完成：{}/15 个 k；{}/15000 个有效 (qid,k)。".format(len(state.get("completed", [])), sum(row["count"] for row in state.get("completed", []))),
        "- 预测目录：`" + str(formal_dir) + "`。", "- 逐题和 Token：每个 k 的 prediction__*.json、token_usage.jsonl、checkpoints/*.json、api_responses/*.json。",
        "- 当前状态详见 `status.json` 与 `formal_k_*.log`；尚未通过验证的 k 不计为完成。", "",
        "| k | count | official F1 | EM | API prompt tokens | completion tokens | total tokens | latency s |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in state.get("completed", []):
        lines.append("| {} | {} | {:.6f} | {:.6f} | {:.1f} | {:.1f} | {:.1f} | {:.3f} |".format(row["k"], row["count"], row["f1"], row["em"], row["average_prompt_tokens"], row["average_completion_tokens"], row["average_total_tokens"], row["average_latency_seconds"]))
    lines.extend(["", "## 分析与研究用途", "", "- 评分全部使用 official_evaluation/hotpotqa/hotpot_evaluate_v1.py；Oracle 最大化官方 F1，并列取最小 k，保留全部并列 k 和完整 F1 曲线。",
        "- 分析复用 analysis/analyze_hotpot_500_500.py；context_tokens 保持原函数计算，API prompt/completion/total tokens 独立保存。",
        "- 历史 DROP F1 与新官方 F1 不能直接比较；历史预测和分析保持不变。",
        "- Dynamic Top-K 标签分析：" + ("可以使用；15000 条完整性验收及原有十项分析产物已通过。" if state["status"] == "complete" else "暂不可作为已验收完整数据使用；需全部 15 个 k 通过完整性验证并完成分析。")])
    if state.get("analysis_dir"):
        lines.append("- 原有分析产物与 Bridge/Comparison/Combined Fixed-k、Oracle 并列和差距统计：`" + state["analysis_dir"] + "`。")
    if state.get("error"):
        lines.extend(["", "## 异常", "", "```text", state["error"], "```", "", "未伪造或补齐结果；受影响运行已停止。"])
    if state.get("engineering_repairs"):
        lines.extend(["", "## Smoke 工程修复记录", "", *state["engineering_repairs"]])
    atomic_text(report_dir / "formal_experiment_report_zh.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-name", default="hotpot_formal_1000_20261014")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-smoke", action="store_true", help="Record reviewed engineering fixes before any formal inference; rerun the smoke gate")
    parser.add_argument("--stage", choices=("smoke", "all"), default="all")
    args = parser.parse_args()
    assert Path(args.set_name).name == args.set_name
    os.chdir(ROOT)
    report_dir = ROOT / "analysis" / (args.set_name + "_run")
    formal_dir = ROOT / "predictions" / args.set_name
    smoke_name = args.set_name + "_smoke"
    smoke_dir = ROOT / "predictions" / smoke_name
    if args.resume:
        manifest = json.loads((report_dir / "manifest.json").read_text())
        state = json.loads((report_dir / "status.json").read_text())
        if args.repair_smoke:
            assert not state["smoke_passed"] and not state["completed"]
            assert not list(formal_dir.rglob("checkpoints/*.json")), "Cannot change sources after formal inference begins"
            assert sha256(CONFIG) == manifest["formal_config_sha256"] and sha256(DATA) == manifest["dataset_sha256"]
            changed = [name for name, digest in manifest["source_hashes"].items() if sha256(ROOT / name) != digest]
            assert changed and all(name.startswith("analysis/hotpot_formal_runner/") for name in changed), "Only experiment-local smoke guard/recovery repairs are permitted"
            revision = "smoke_repair_" + str(time.time_ns())
            atomic_json(report_dir / (revision + "_previous_manifest.json"), manifest)
            atomic_json(report_dir / (revision + "_previous_status.json"), state)
            manifest["source_hashes"].update({name: sha256(ROOT / name) for name in changed})
            atomic_json(report_dir / "manifest.json", manifest)
            state.setdefault("engineering_repairs", []).append("- {}：修正检查以遵循原 CopyQuestionParticipant.strip()；增加完整预测已落盘后的安全恢复及流式文件哈希。变化文件：{}。配置、Prompt、数据、评分和模型生成参数保持一致；失败记录保留，重跑 Smoke gate。".format(now(), ", ".join(changed)))
            state.pop("error", None)
        for name, digest in manifest["source_hashes"].items():
            assert sha256(ROOT / name) == digest, "Code changed since experiment began: " + name
        assert sha256(DATA) == manifest["dataset_sha256"] and sha256(CONFIG) == manifest["formal_config_sha256"]
    else:
        assert not any(p.exists() for p in (report_dir, formal_dir, smoke_dir)), "Use a unique set-name; do not overwrite an experiment"
        report_dir.mkdir()
        manifest = prepare(args, report_dir, formal_dir, smoke_dir)
        state = {"status": "prepared", "started_at": now(), "completed": [], "smoke_passed": False}
    lock = (report_dir / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def save():
        state["updated_at"] = now()
        atomic_json(report_dir / "status.json", state)
        report(report_dir, manifest, state, formal_dir, smoke_dir)
    try:
        save()
        if not state["smoke_passed"]:
            state["status"] = "smoke_running"
            save()
            execute([str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-v"], report_dir / "regression_tests.log", environment())
            state["regression_tests"] = "全部 unittest 通过；包含 Unicode/JSON、多行 CoT、Yes/No、官方评分和断点恢复"
            smoke_data = Path(manifest["smoke_data"])
            summaries = [predict_or_recover(manifest, report_dir, smoke_data, smoke_name, k, report_dir / ("smoke_k_{}.log".format(k))) for k in (1, 15)]
            assert all(row["count"] == 8 and row["format_invalid"] == 0 for row in summaries)
            for k in (1, 15):
                directory = ROOT / "predictions" / smoke_name / Path(manifest["config_paths"][str(k)]).stem
                events = load_jsonl(directory / "request_events.jsonl")
                assert len(events) == 8, "Smoke must make exactly eight real calls per k"
            rehearse_recovery(manifest, report_dir, smoke_name)
            atomic_json(report_dir / "smoke_results.json", {"passed": True, "actual_calls": 16, "samples": manifest["smoke_samples"], "summaries": summaries})
            state.update(smoke_passed=True, status="smoke_passed")
            save()
            print("SMOKE PASSED: 16/16; entering formal sweep", flush=True)
        if args.stage == "smoke":
            return
        for k in range(1, 16):
            if k in [row["k"] for row in state["completed"]]:
                validate_k(manifest, DATA, args.set_name, k)
                continue
            state.update(status="formal_running", current_k=k)
            save()
            start = time.monotonic()
            summary = predict_or_recover(manifest, report_dir, DATA, args.set_name, k, report_dir / ("formal_k_{}.log".format(k)))
            summary["wall_seconds"] = time.monotonic() - start
            state["completed"].append(summary)
            save()
            print("FORMAL COMPLETE k={} count={} f1={:.6f}".format(k, summary["count"], summary["f1"]), flush=True)
        state["status"] = "analyzing"
        save()
        analysis_dir = ROOT / "analysis" / args.set_name
        execute([str(PYTHON), "analysis/analyze_hotpot_500_500.py", "--data", str(DATA), "--predictions-dir", str(formal_dir), "--output-dir", str(analysis_dir)], report_dir / "analysis.log", environment())
        assert all((analysis_dir / name).is_file() and (analysis_dir / name).stat().st_size for name in ARTIFACTS)
        assert len((analysis_dir / "per_question_curves.jsonl").read_text().splitlines()) == 1000
        for name, digest in json.loads((report_dir / "protected_hashes.json").read_text()).items():
            assert sha256(ROOT / name) == digest, "Protected historical file changed: " + name
        assert sum(row["count"] for row in state["completed"]) == 15000
        atomic_json(report_dir / "acceptance.json", {"valid_pairs": 15000, "all_k_validated": True, "historical_files_unchanged": True,
            "dataset_sha256": sha256(DATA), "analysis_artifacts": list(ARTIFACTS), "finished_at": now()})
        state.update(status="complete", analysis_dir=str(analysis_dir), finished_at=now())
        save()
    except BaseException as error:
        state.update(status="stopped_on_error", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
