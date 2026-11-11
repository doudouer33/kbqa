"""Experiment-only observers, activated via sitecustomize in predict.py's child.

Never replace the OpenAI client, decorated cache function, call arguments, or
return values. The converter observation runs only for an actual API response;
old DiskCache entries can therefore truthfully have unknown API usage.
"""
import copy
import fcntl
import json
import os
import sys
from pathlib import Path

from common import RUNS, ROOT, atomic_json, binding, now, read_json, require, sha256, text_hash


class ObservationFailure(BaseException):
    """Do not let legacy Exception handlers retry instrumentation failures."""


def check(condition, message):
    if not condition:
        raise ObservationFailure(message)


def install():
    from commaqa.models import gpt3generator as generator
    from commaqa.execution import llm_qa_model
    from commaqa.inference import ircot, model_search, participant_qa

    manifest = read_json(RUNS / "run_manifest.json")
    mode, k = os.environ["HOTPOT5000_MODE"], int(os.environ["HOTPOT5000_K"])
    expected = binding(manifest, mode, k)
    config = Path(sys.argv[sys.argv.index("--config") + 1])
    data = Path(sys.argv[sys.argv.index("--input") + 1])
    attempt = Path(os.environ["HOTPOT5000_ATTEMPT"])
    require(sha256(config) == expected["config_sha256"] and sha256(data) == expected["input_sha256"], "Child input/config changed")
    metadata = read_json(attempt / "attempt.json")
    require(metadata["binding"] == expected, "Child attempt binding mismatch")
    records = attempt / "records"
    records.mkdir(exist_ok=True)
    output = Path(sys.argv[sys.argv.index("--output") + 1]).parent
    inference_lock = (output / ".inference.lock").open("a")
    fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    generator._hotpot5000_inference_lock = inference_lock
    current = {}
    completed_count = [0]

    original_retrieve = ircot.safe_post_request
    def observe_retrieve(*args, **kwargs):
        response = original_retrieve(*args, **kwargs)
        check(response.ok, "Retriever HTTP failure; cannot record a successful question")
        check(bool(response.json().get("retrieval")), "Retriever returned no documents")
        return response
    ircot.safe_post_request = observe_retrieve

    original_qa = participant_qa.LLMQAParticipantModel.query
    def observe_qa(self, state, *args, **kwargs):
        current["raw_contexts"] = [
            {"title": title, "paragraph_text": para}
            for title, para in zip(state.data["titles"], state.data["paras"])
        ]
        return original_qa(self, state, *args, **kwargs)
    participant_qa.LLMQAParticipantModel.query = observe_qa

    original_ask = llm_qa_model.LLMQAModel.ask_question
    def observe_ask(self, input_question, context, context_suffix=""):
        # This is the original participant's rendered list AFTER pin ordering,
        # filtering, title formatting and 350-word paragraph truncation.
        check(isinstance(context, list) and self.add_context, "Unexpected formal Reader context")
        check(not input_question.startswith("PARA_"), "Unexpected per-paragraph QA route")
        current["reader_contexts"] = copy.deepcopy(context)
        current["effective_context_text"] = "\n\n".join(context)
        current["context_suffix"] = context_suffix
        current["input_question"] = input_question
        current["expected_test_suffix"] = ("\n\n\n" + current["effective_context_text"] + context_suffix
                                             + "\n\nQ: " + input_question + "\nA:")
        return original_ask(self, input_question, context, context_suffix)
    llm_qa_model.LLMQAModel.ask_question = observe_ask

    original_fit = generator.fit_prompt_into_given_limit
    def observe_fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        current["prompt_cropped"] = result != kwargs["original_prompt"]
        return result
    generator.fit_prompt_into_given_limit = observe_fit

    original_convert = generator._chat_response_to_legacy_dict
    def observe_response(response):
        usage = getattr(response, "usage", None)
        values = ({key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
                  if usage is not None else None)
        valid = values is not None and all(type(value) is int and value >= 0 for value in values.values())
        valid = valid and values["total_tokens"] == values["prompt_tokens"] + values["completion_tokens"]
        current["api_response"] = {
            "usage": values if valid else None, "usage_source": "api_response" if valid else "unavailable_api_response",
            "response_id": getattr(response, "id", None), "response_model": getattr(response, "model", None),
            "finish_reasons": [choice.finish_reason for choice in response.choices],
        }
        return original_convert(response)
    generator._chat_response_to_legacy_dict = observe_response

    original_call = generator.openai_call
    def observe_call(**arguments):
        prompt = arguments["prompt"]
        check(prompt.endswith(current["expected_test_suffix"]), "Runtime truncation changed the test context; stopping before LLM call")
        # Hash the actual post-fit prompt AND every original cache argument.
        # No endpoint or instrumentation fields are added to the legacy cache key.
        current["prompt_sha256"] = text_hash(prompt)
        current["effective_context_verified_in_prompt"] = True
        current["call_sha256"] = text_hash(json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        current["parameters"] = {key: copy.deepcopy(value) for key, value in arguments.items() if key != "prompt"}
        current["fewshot_count_after_fit"] = len(prompt.split("\n\n\n")) - 1
        current.pop("api_response", None)
        result = original_call(**arguments)
        api = current.pop("api_response", None)
        current["cache_hit"] = arguments["temperature"] == 0 and api is None
        current["api"] = api or {
            "usage": None, "usage_source": "unavailable_legacy_diskcache",
            "response_id": None, "response_model": None, "finish_reasons": None,
        }
        current["raw_generation"] = result["choices"][0]["text"]
        return result
    generator.openai_call = observe_call

    original_find = model_search.BestFirstDecomposer.find_answer_decomp
    def observe_find(self, *args, **kwargs):
        result = original_find(self, *args, **kwargs)
        current["reached_final_state"] = result[0] is not None
        return result
    model_search.BestFirstDecomposer.find_answer_decomp = observe_find

    original_prediction = model_search.QuestionSearchBase.return_qid_prediction
    def observe_prediction(self, example, *args, **kwargs):
        current.clear()
        current.update(qid=example["qid"])
        result = original_prediction(self, example, *args, **kwargs)
        check(result[0] == current["qid"] and isinstance(result[1], str), "Invalid prediction")
        check(current.get("reached_final_state"), "Inference did not reach a final state")
        check("api" in current and "reader_contexts" in current, "Prediction lacks context/call observation")
        check(len(current["raw_contexts"]) == len(current["reader_contexts"]) > 0, "Context count mismatch")
        check(len(current["raw_contexts"]) <= k, "Context count exceeds requested k")
        reasons = current["api"]["finish_reasons"]
        check(reasons is None or reasons == ["stop"], "Abnormal API finish_reason: " + str(reasons))
        # A cached response has no new API usage. Preserve a prior observed usage
        # only for exactly the same prompt, parameters and raw generation.
        previous = output / "observations" / (current["qid"] + ".json")
        if current["cache_hit"] and previous.exists():
            old = read_json(previous)
            if (old["binding"] == expected and old["call_sha256"] == current["call_sha256"]
                    and old["raw_generation"] == current["raw_generation"] and old["api"]["usage"] is not None):
                current["api"] = dict(old["api"], usage_source="saved_api_response")
        row = {key: copy.deepcopy(value) for key, value in current.items() if key != "expected_test_suffix"}
        row.update(schema_version=1, k=k, binding=expected, recorded_at_utc=now(), prediction=result[1], chain=result[2],
                   context_sha256=text_hash(current["effective_context_text"]), chain_sha256=text_hash(result[2]))
        atomic_json(records / (current["qid"] + ".json"), row)
        atomic_json(previous, row)
        completed_count[0] += 1
        print("HOTPOT5000 k={} completed={} qid={} cache_hit={}".format(k, completed_count[0], current["qid"], current["cache_hit"]), flush=True)
        return result
    model_search.QuestionSearchBase.return_qid_prediction = observe_prediction
