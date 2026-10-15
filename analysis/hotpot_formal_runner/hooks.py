"""Observe the verified ONER pipeline and fail closed on experiment violations.

Same observation points as hotpot_stop_diagnostic/hooks.py. Only transport,
memoization, durable recording and failure handling are adapted here.
"""
import copy
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import httpx
import requests
from openai import OpenAI
from openai.types.chat import ChatCompletion

from analysis.hotpot_formal_runner.storage import atomic_json, atomic_text, append_json, sha256, text_hash


class ExperimentViolation(BaseException):
    """Bypass legacy exception handlers that retry or change generation budgets."""


def require(condition, message):
    if not condition:
        raise ExperimentViolation(message)


def install():
    import _jsonnet
    from commaqa.models import gpt3generator as generator
    from commaqa.execution import llm_qa_model
    from commaqa.inference import model_search, ircot, participant_qa

    manifest_path = Path(os.environ["KBQA_FORMAL_MANIFEST"])
    manifest = json.loads(manifest_path.read_text())
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    input_path = Path(sys.argv[sys.argv.index("--input") + 1])
    output = Path(sys.argv[sys.argv.index("--output") + 1]).parent
    config = json.loads(_jsonnet.evaluate_file(str(config_path), ext_vars=manifest["ext_vars"]))
    k = config["models"]["generate_titles"]["retrieval_count"]
    require(sha256(config_path) == manifest["config_hashes"][str(k)], "Instantiated config changed")
    require(sha256(input_path) == os.environ["KBQA_FORMAL_INPUT_SHA256"], "Dataset changed")
    require(sha256(config["models"]["answer_main_question"]["prompt_file"]) == manifest["prompt_sha256"], "Prompt changed")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".inference.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Keep the descriptor alive throughout this process.
    generator._formal_lock = lock
    checkpoints = output / "checkpoints"
    responses = output / "api_responses"
    checkpoints.mkdir(exist_ok=True)
    responses.mkdir(exist_ok=True)
    samples = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    qids = [row["question_id"] for row in samples]
    require(len(qids) == len(set(qids)), "Duplicate input qid")
    completed = {}
    for path in checkpoints.glob("*.json"):
        row = json.loads(path.read_text())
        require(row["qid"] in qids and row["k"] == k, "Foreign checkpoint")
        require(row["config_sha256"] == sha256(config_path), "Checkpoint config differs")
        require(row["input_sha256"] == sha256(input_path), "Checkpoint dataset differs")
        require(row["qid"] not in completed, "Duplicate checkpoint")
        completed[row["qid"]] = row
    # Recover an interrupted journal append from atomic completed checkpoints.
    atomic_text(output / "token_usage.jsonl", "".join(
        json.dumps(completed[qid]["tokens"], ensure_ascii=False) + "\n" for qid in qids if qid in completed
    ))
    current = {}
    expected_prompt = (manifest_path.parent / "selected_prompt.txt").read_text()
    tokenizer_session = requests.Session()
    tokenizer_session.trust_env = False
    request_events = output / "request_events.jsonl"
    previously_requested = set()
    if request_events.exists():
        for line in request_events.read_text().splitlines():
            previously_requested.add(json.loads(line)["qid"])

    original_read = llm_qa_model.read_prompt
    def checked_read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        require(result == expected_prompt, "Initialization dropped or changed Few-shot examples")
        require(len(result.split("\n\n\n")) == 15, "Expected 15 Few-shot examples")
        return result
    llm_qa_model.read_prompt = checked_read

    original_fit = generator.fit_prompt_into_given_limit
    def checked_fit(*args, **kwargs):
        before = kwargs["original_prompt"]
        current["prompt_sha256"] = text_hash(before)
        require(before.startswith(expected_prompt + "\n\n\n"), "Few-shot prefix changed")
        # CopyQuestionParticipant has always stripped outer question whitespace.
        require(before.endswith("\n\nQ: " + current["question"].strip() + "\nA:"), "Current question changed")
        require(kwargs["model_length_limit"] == 16384 and kwargs["estimated_generation_length"] == 512, "Token budgets changed")
        result = original_fit(*args, **kwargs)
        require(result == before, "Runtime prompt was cropped; stop before generation")
        try:
            response = tokenizer_session.post("http://127.0.0.1:8002/tokenize", json={"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": result}], "add_generation_prompt": True}, timeout=30)
            response.raise_for_status()
            current["api_prompt_tokens_checked"] = response.json()["count"]
        except Exception as error:
            raise ExperimentViolation("Runtime chat-template budget check failed: " + repr(error))
        require(current["api_prompt_tokens_checked"] + 512 <= 16384, "Runtime chat-template token budget exceeded; stop before generation")
        current["prompt"] = result
        return result
    generator.fit_prompt_into_given_limit = checked_fit

    original_retrieve = ircot.safe_post_request
    def checked_retrieve(url, params):
        try:
            response = original_retrieve(url, params)
            require(response.ok, "Retriever request failed")
            documents = response.json()["retrieval"]
            require(bool(documents), "Retriever returned no documents")
            current["retrieval"] = {"request": copy.deepcopy(params), "documents": documents}
            return response
        except Exception as error:
            raise ExperimentViolation("Retriever failed: " + repr(error))
    ircot.safe_post_request = checked_retrieve

    original_qa = participant_qa.LLMQAParticipantModel.query
    def observed_qa(self, state, *args, **kwargs):
        current["contexts"] = [{"title": title, "paragraph_text": para} for title, para in zip(state.data["titles"], state.data["paras"])]
        current["context_sha256"] = text_hash("\n\n".join(ircot.para_to_text(d["title"], d["paragraph_text"], self.max_para_num_words) for d in current["contexts"]))
        return original_qa(self, state, *args, **kwargs)
    participant_qa.LLMQAParticipantModel.query = observed_qa

    original_extract = ircot.AnswerExtractor.query
    def observed_extract(self, state, *args, **kwargs):
        query = state.data.get_last_answer()
        if query.startswith('"') and query.endswith('"'):
            query = json.loads(query)
        current["format_valid"] = bool(self.regex.match(query))
        return original_extract(self, state, *args, **kwargs)
    ircot.AnswerExtractor.query = observed_extract

    def on_request(request):
        body = json.loads(request.content)
        require(str(request.url) == "http://127.0.0.1:8002/v1/chat/completions", "Wrong generation endpoint")
        for key, value in {"model": "Qwen2.5-7B-Instruct", "stop": [], "max_tokens": 512, "temperature": 0, "top_p": 1, "n": 1, "frequency_penalty": 0, "presence_penalty": 0}.items():
            require(body[key] == value, "Wire parameter differs: " + key)
        require(body["messages"] == [{"role": "user", "content": current["prompt"]}], "Wire prompt changed")
        require(not current.get("request_started"), "Repeated request for one question")
        current["request_started"] = time.monotonic()
        append_json(output / "request_events.jsonl", {"qid": current["qid"], "k": k, "prompt_sha256": current["prompt_sha256"], "parameters": {key: value for key, value in body.items() if key != "messages"}})

    def on_response(response):
        response.read()
        require(response.status_code == 200, "vLLM request failed: " + str(response.status_code) + " " + response.text[:1000])
        body = response.json()
        usage = body.get("usage", {})
        require(all(isinstance(usage.get(key), int) and usage[key] >= 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")), "Missing actual API usage")
        require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Inconsistent API usage")
        require(usage["prompt_tokens"] == current["api_prompt_tokens_checked"], "Runtime chat-template count differs from API usage")
        current["api"] = {"response": body, "latency_seconds": time.monotonic() - current["request_started"], "prompt_sha256": current["prompt_sha256"], "config_sha256": sha256(config_path)}
        atomic_json(responses / (current["qid"] + ".json"), current["api"])
        require(usage["prompt_tokens"] + 512 <= 16384, "Actual chat prompt exceeds formal token budget")
        require(body["choices"][0]["finish_reason"] == "stop", "Generation did not finish with stop")

    generator.client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="EMPTY", max_retries=0, timeout=180,
        http_client=httpx.Client(trust_env=False, event_hooks={"request": [on_request], "response": [on_response]}))
    # The verified temperature=0 call body, bypassing the historical disk cache.
    original_call = generator.cached_openai_call.__wrapped__
    def guarded_call(**arguments):
        pending = responses / (current["qid"] + ".json")
        try:
            if pending.exists():
                api = json.loads(pending.read_text())
                require(api["prompt_sha256"] == current["prompt_sha256"] and api["config_sha256"] == sha256(config_path), "Saved API response does not match current prompt/config")
                require(api["response"]["choices"][0]["finish_reason"] == "stop", "Saved response has an abnormal finish reason")
                current["api"] = api
                current["recovered_api_response"] = True
                return generator._chat_response_to_legacy_dict(ChatCompletion(**api["response"]))
            require(current["qid"] not in previously_requested, "Prior request lacks a durable response; ambiguous completion, stop without duplicate inference")
            return original_call(**arguments)
        except Exception as error:
            raise ExperimentViolation("Model request failed; no parameter retry: " + repr(error))
    generator.openai_call = guarded_call

    original_prediction = model_search.QuestionSearchBase.return_qid_prediction
    def checked_prediction(self, example, *args, **kwargs):
        qid = example["qid"]
        if qid in completed:
            return tuple(completed[qid]["result"])
        current.clear()
        current.update(qid=qid, question=example["question"])
        start = time.monotonic()
        try:
            result = original_prediction(self, example, *args, **kwargs)
            require(result[0] == qid and bool(result[1]), "Missing or empty final prediction")
            require("api" in current and "contexts" in current, "Missing model/context record")
            body = current["api"]["response"]
            tokens = {"qid": qid, "k": k, **body["usage"], "latency_seconds": current["api"]["latency_seconds"],
                "finish_reason": body["choices"][0]["finish_reason"], "prompt_cropped": False, "format_valid": current["format_valid"],
                "prompt_sha256": current["prompt_sha256"], "context_sha256": current["context_sha256"], "fewshot_count": 15,
                "raw_generation": body["choices"][0]["message"]["content"], "prediction": result[1],
                "config_sha256": sha256(config_path), "recovered_api_response": current.get("recovered_api_response", False)}
            row = {"qid": qid, "k": k, "result": result, "tokens": tokens, "contexts": current["contexts"],
                "retrieval": current["retrieval"], "config_sha256": sha256(config_path), "input_sha256": sha256(input_path),
                "elapsed_seconds": time.monotonic() - start}
            atomic_json(checkpoints / (qid + ".json"), row)
            append_json(output / "token_usage.jsonl", tokens)
            completed[qid] = row
            print("FORMAL k={} completed={}/{} qid={} seconds={:.2f}".format(k, len(completed), len(qids), qid, row["elapsed_seconds"]), flush=True)
            return result
        except BaseException as error:
            append_json(output / "errors.jsonl", {key: value for key, value in dict(current, error=repr(error)).items() if key not in ("prompt", "api")})
            raise
    model_search.QuestionSearchBase.return_qid_prediction = checked_prediction
