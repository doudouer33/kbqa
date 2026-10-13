"""Observe the existing pipeline without changing prompts or answer extraction."""
import json
import os
import re
import time
from pathlib import Path

import httpx
from openai import OpenAI


def install():
    from commaqa.models import gpt3generator as generator
    from commaqa.inference import prompt_reader, ircot, model_search
    from commaqa.execution import llm_qa_model

    output = Path(os.environ["KBQA_STOP_DIAGNOSTIC_DIR"])
    config_path = Path(__import__("sys").argv[__import__("sys").argv.index("--config") + 1])
    k = int(re.search(r"retrieval_count__(\d+)", config_path.name).group(1))
    current = {}

    def append(name, row):
        with (output / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_request(request):
        if request.url.path.endswith("/chat/completions"):
            payload = json.loads(request.content)
            assert payload["stop"] == [], "The wire request must contain stop=[]"
            current.setdefault("api_requests", []).append({"url": str(request.url), "body": payload})
            append("http_events.jsonl", {"k": k, "qid": current.get("qid"), "event": "request", "url": str(request.url), "body": payload})

    def on_response(response):
        if response.request.url.path.endswith("/chat/completions"):
            response.read()
            body = response.json()
            current.setdefault("api_responses", []).append({"status": response.status_code, "body": body})
            append("http_events.jsonl", {"k": k, "qid": current.get("qid"), "event": "response", "status": response.status_code, "body": body})

    # Environment adaptation: the original module's OpenAI() uses these same
    # OPENAI_* variables; hooks only retain the response before legacy conversion.
    generator.client = OpenAI(http_client=httpx.Client(event_hooks={"request": [on_request], "response": [on_response]}))
    # Bypass memoization only in this diagnostic subprocess. The original call
    # body and temperature=0 dispatch are retained; old cache entries are unused.
    generator.cached_openai_call = generator.cached_openai_call.__wrapped__

    original_fit = generator.fit_prompt_into_given_limit

    def observe_fit(*args, **kwargs):
        before = kwargs["original_prompt"]
        result = original_fit(*args, **kwargs)
        tokenizer = prompt_reader.get_tokenizer(kwargs["tokenizer_model_name"])
        delimiter = kwargs["demonstration_delimiter"]
        current["prompt_fit"] = {
            "before": before, "after": result,
            "before_tokens": len(tokenizer.tokenize(before)),
            "after_tokens": len(tokenizer.tokenize(result)),
            "model_length_limit": kwargs["model_length_limit"],
            "estimated_generation_length": kwargs["estimated_generation_length"],
            "before_demonstrations": max(0, len(before.split(delimiter)) - 1),
            "after_demonstrations": max(0, len(result.split(delimiter)) - 1),
            "test_example_before": before.split(delimiter)[-1],
            "test_example_after": result.split(delimiter)[-1],
            "changed": result != before,
        }
        return result

    generator.fit_prompt_into_given_limit = observe_fit
    original_read = llm_qa_model.read_prompt

    def observe_read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        append("prompt_initialization.jsonl", {"k": k, "arguments": kwargs, "selected_prompt": result, "retained_demonstrations": len(result.split("\n\n\n"))})
        return result

    llm_qa_model.read_prompt = observe_read
    original_generate = generator.GPT3Generator.generate_text_sequence

    def observe_generate(self, prompt):
        current["generator_settings"] = {key: getattr(self, key) for key in ("engine", "temperature", "max_tokens", "top_p", "stop", "model_tokens_limit", "tokenizer_model_name", "remove_method", "n", "best_of", "logprobs")}
        result = original_generate(self, prompt)
        current["generator_outputs"] = result
        return result

    generator.GPT3Generator.generate_text_sequence = observe_generate
    original_extract = ircot.AnswerExtractor.query

    def observe_extract(self, state, debug=False):
        serialized = state.data.get_last_answer()
        query = serialized[1:-1] if serialized.startswith('"') and serialized.endswith('"') else serialized
        match = self.regex.match(query)
        result = original_extract(self, state, debug=debug)
        current["extractor"] = {"serialized_input": serialized, "regex_input": query, "regex": self.regex.pattern, "primary_match": bool(match), "matched_group": match.group(1) if match else None, "fallback": self.match_all_on_failure and not bool(match), "output": result.data.get_last_answer() if result else None}
        return result

    ircot.AnswerExtractor.query = observe_extract
    original_retrieve = ircot.safe_post_request

    def observe_retrieve(url, params):
        result = original_retrieve(url, params)
        current.setdefault("retrieval_calls", []).append({"url": url, "request": dict(params), "status": result.status_code, "response": result.json()})
        return result

    ircot.safe_post_request = observe_retrieve
    original_prediction = model_search.QuestionSearchBase.return_qid_prediction

    def observe_prediction(self, example, *args, **kwargs):
        current.clear()
        current.update({"k": k, "qid": example["qid"], "question": example["question"], "config_path": str(config_path)})
        start = time.monotonic()
        try:
            result = original_prediction(self, example, *args, **kwargs)
            current.update({"final_answer": result[1], "reasoning_chain": result[2]})
            return result
        except BaseException as error:
            current["error"] = repr(error)
            raise
        finally:
            current["elapsed_seconds"] = time.monotonic() - start
            append("diagnostics.jsonl", current)
            print("DIAGNOSTIC k={} qid={} api_calls={} seconds={:.2f}".format(k, example["qid"], len(current.get("api_requests", [])), current["elapsed_seconds"]), flush=True)

    model_search.QuestionSearchBase.return_qid_prediction = observe_prediction
