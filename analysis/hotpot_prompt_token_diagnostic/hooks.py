"""Reuse stop diagnostics and reject any prompt loss before a generation call."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path


class DiagnosticViolation(BaseException):
    """Fatal guard failure; do not let the generator retry with changed budgets."""


def require(condition, message):
    if not condition:
        raise DiagnosticViolation(message)


def install():
    from analysis.hotpot_stop_diagnostic.hooks import install as install_stop_hooks

    output = Path(os.environ["KBQA_PROMPT_TOKEN_DIAGNOSTIC_DIR"])
    os.environ["KBQA_STOP_DIAGNOSTIC_DIR"] = str(output)
    install_stop_hooks()

    from commaqa.models import gpt3generator as generator
    from commaqa.execution import llm_qa_model
    from commaqa.inference import model_search, ircot

    preflight = json.loads((output / "preflight.json").read_text())
    baseline_dir = Path(preflight["baseline_dir"])
    baseline = {
        (row["k"], row["qid"]): row
        for row in map(json.loads, (baseline_dir / "diagnostics.jsonl").read_text().splitlines())
    }
    expected_tokens = {(row["k"], row["qid"]): row for row in preflight["requests"]}
    config = Path(sys.argv[sys.argv.index("--config") + 1])
    k = int(re.search(r"retrieval_count__(\d+)", config.name).group(1))
    current = {}

    def append(row):
        with (output / "guard_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    original_read = llm_qa_model.read_prompt

    def checked_read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        require(result == preflight["selected_prompt"], "Initialization changed or dropped a selected example")
        require(len(result.split("\n\n\n")) == 15, "Expected all 15 examples")
        append({"event": "initialization", "k": k, "fewshot_ids": preflight["fewshot_ids"], "cropped": False})
        return result

    llm_qa_model.read_prompt = checked_read
    original_fit = generator.fit_prompt_into_given_limit

    def checked_fit(*args, **kwargs):
        key = (k, current["qid"])
        expected = preflight["selected_prompt"] + "\n\n\n" + baseline[key]["prompt_fit"]["test_example_before"]
        require(kwargs["original_prompt"] == expected, "Question or retrieved context differs from the previous experiment")
        require(kwargs["estimated_generation_length"] == 512, "Generation reserve must be 512")
        require(kwargs["model_length_limit"] == 16384, "Unexpected local token budget")
        result = original_fit(*args, **kwargs)
        require(result == expected, "Runtime prompt fitting cropped or changed the prompt")
        require(len(result.split("\n\n\n")) - 1 == 15, "Runtime example count is not 15")
        current["prompt"] = result
        current["fewshot_ids"] = preflight["fewshot_ids"]
        current["prompt_sha256"] = hashlib.sha256(result.encode()).hexdigest()
        require(current["prompt_sha256"] == expected_tokens[key]["prompt_sha256"], "Preflight prompt differs")
        return result

    generator.fit_prompt_into_given_limit = checked_fit
    original_retrieve = ircot.safe_post_request

    def checked_retrieve(url, params):
        old = baseline[(k, current["qid"])]["retrieval_calls"][0]
        require(url == old["url"] and params == old["request"], "BM25 request changed")
        result = original_retrieve(url, params)
        require(result.json()["retrieval"] == old["response"]["retrieval"], "BM25 results changed; stop the paired experiment")
        current["retrieval_unchanged"] = True
        return result

    ircot.safe_post_request = checked_retrieve

    def checked_request(request):
        if not request.url.path.endswith("/chat/completions"):
            return
        body = json.loads(request.content)
        require(str(request.url) == "http://127.0.0.1:8002/v1/chat/completions", "Wrong API endpoint")
        for key, value in {"model": "Qwen2.5-7B-Instruct", "stop": [], "max_tokens": 512, "temperature": 0, "top_p": 1, "n": 1}.items():
            require(body[key] == value, "Unexpected wire parameter: " + key)
        require(body["messages"] == [{"role": "user", "content": current["prompt"]}], "Wire prompt changed")
        require(preflight["instructions"] in current["prompt"], "Instructions missing on the wire")
        count = expected_tokens[(k, current["qid"])]["api_prompt_tokens"]
        require(count + 512 + preflight["safety_margin"] <= preflight["server_max_model_len"], "Server context budget insufficient")
        current["wire_validated"] = True

    # Validate before the existing logging hook and before sending any request.
    generator.client._client.event_hooks["request"].insert(0, checked_request)

    def checked_response(response):
        if not response.request.url.path.endswith("/chat/completions"):
            return
        response.read()
        body = response.json()
        require(response.status_code == 200, "Generation request failed")
        count = expected_tokens[(k, current["qid"])]["api_prompt_tokens"]
        require(body["usage"]["prompt_tokens"] == count, "API usage differs from server chat-template preflight")
        current["api_prompt_tokens"] = count
        current["completion_tokens"] = body["usage"]["completion_tokens"]
        current["finish_reason"] = body["choices"][0]["finish_reason"]

    generator.client._client.event_hooks["response"].append(checked_response)
    original_prediction = model_search.QuestionSearchBase.return_qid_prediction

    def checked_prediction(self, example, *args, **kwargs):
        current.clear()
        current.update({"event": "question", "k": k, "qid": example["qid"], "cropped": False})
        try:
            return original_prediction(self, example, *args, **kwargs)
        except BaseException as error:
            current["error"] = repr(error)
            raise
        finally:
            append({key: value for key, value in current.items() if key != "prompt"})

    model_search.QuestionSearchBase.return_qid_prediction = checked_prediction
