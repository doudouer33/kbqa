#!/usr/bin/env python3
"""Reconstruct ONER model inputs from a prediction directory.

The script replays the retrieval stage, validates the replayed title sequence
against the saved reasoning chain, and renders the exact prompt string produced
by the current project code.  It does not call the language model and it does
not modify any original prediction artifact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dependencies():
    try:
        import _jsonnet
        import requests
        from rapidfuzz import fuzz
        from commaqa.inference.prompt_reader import fit_prompt_into_given_limit, read_prompt
    except ImportError as exc:
        raise SystemExit(
            "Missing a project dependency. Run this script in the same Python "
            "environment used by run.py (see requirements.txt). Original error: "
            f"{exc}"
        ) from exc
    return _jsonnet, requests, fuzz, fit_prompt_into_given_limit, read_prompt


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(directory: Path, pattern: str, required: bool = True) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one file matching {pattern!r} in {directory}, "
            f"found {len(matches)}: {[str(path) for path in matches]}"
        )
    return matches[0]


def read_variable_replacements(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"Variable replacements must be a JSON object: {path}")
    return value


def apply_variable_replacements(content: str, replacements: Dict[str, str]) -> str:
    """Apply the same local-variable replacement convention used by run.py."""

    replacements = copy.deepcopy(replacements)
    for variable_name, variable_value in replacements.items():
        if not isinstance(variable_value, str):
            raise ValueError(f"Replacement {variable_name!r} must be a string")

        for invoked_name in re.findall(r"\$[a-zA-Z0-9-_]+", variable_value):
            invoked_name = invoked_name.lstrip("$")
            if invoked_name not in replacements:
                raise ValueError(f"Replacement {variable_name!r} refers to missing {invoked_name!r}")
            variable_value = variable_value.replace("$" + invoked_name, replacements[invoked_name])

        if re.match(r"eval\(.+\)", variable_value):
            expression = re.sub(r"eval\((.+)\)", r"\1", variable_value)
            variable_value = str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - legacy config format

        pattern = re.compile(rf"(.*local {re.escape(variable_name)} =) (.+?)(;.*)", re.DOTALL)
        if not pattern.match(content):
            raise ValueError(f"Variable {variable_name!r} is not defined in the saved config")
        content = re.sub(pattern, r"\1 " + variable_value + r"\3", content)

    return content


def load_config(
    config_path: Path,
    replacements: Dict[str, str],
    retriever_host: str,
    retriever_port: int,
    jsonnet_module,
) -> Dict[str, Any]:
    content = config_path.read_text(encoding="utf-8")
    content = apply_variable_replacements(content, replacements)
    evaluated = jsonnet_module.evaluate_snippet(
        str(config_path),
        content,
        ext_vars={
            "RETRIEVER_HOST": retriever_host,
            "RETRIEVER_PORT": str(retriever_port),
            "LLM_SERVER_HOST": "http://127.0.0.1",
            "LLM_SERVER_PORT": "0",
        },
    )
    config = json.loads(evaluated)
    if config.get("start_state") != "generate_titles":
        raise ValueError(
            "This script currently supports ONER/ONER_QA configs whose start_state "
            f"is 'generate_titles'; found {config.get('start_state')!r}"
        )
    return config


def resolve_evaluation_path(experiment_dir: Path, full_eval_path_file: Path) -> Path:
    raw_path = full_eval_path_file.read_text(encoding="utf-8").strip()
    if not raw_path:
        raise ValueError(f"Empty evaluation path file: {full_eval_path_file}")
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([REPO_ROOT / path, experiment_dir / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Evaluation input not found; tried: {[str(path) for path in candidates]}")


def load_evaluation_questions(path: Path) -> Dict[str, str]:
    questions: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item["question_id"]
            question = item["question_text"]
            if qid in questions:
                raise ValueError(f"Duplicate qid {qid!r} in {path}:{line_number}")
            questions[qid] = question
    return questions


def parse_oner_chain(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    raw_blocks = path.read_text(encoding="utf-8").split("\n\n")
    for block_index, raw_block in enumerate(raw_blocks):
        block = raw_block.strip("\n")
        if not block.strip():
            continue
        lines = block.splitlines()
        if len(lines) < 4:
            raise ValueError(f"Malformed chain block {block_index} in {path}: fewer than four lines")

        qid = lines[0].strip()
        source_question = lines[1]
        retrieval_line_index = next(
            (index for index, line in enumerate(lines[2:], 2) if line.startswith("A: [")), None
        )
        if retrieval_line_index is None:
            raise ValueError(f"No retrieval-title line for qid {qid!r} in {path}")
        expected_titles = json.loads(lines[retrieval_line_index][3:])
        if not isinstance(expected_titles, list) or not all(isinstance(title, str) for title in expected_titles):
            raise ValueError(f"Invalid retrieval-title list for qid {qid!r}")

        question_line_index = next(
            (
                index
                for index, line in enumerate(lines[retrieval_line_index + 1 :], retrieval_line_index + 1)
                if line.startswith("Q: ") and line != "Q: [EOQ]"
            ),
            None,
        )
        if question_line_index is None:
            raise ValueError(f"No model-question line for qid {qid!r}")
        model_question = lines[question_line_index][3:]

        generated_answer = None
        final_answer = None
        answer_lines = [line[3:] for line in lines[question_line_index + 1 :] if line.startswith("A: ")]
        if answer_lines:
            try:
                generated_answer = json.loads(answer_lines[0])
            except json.JSONDecodeError:
                generated_answer = answer_lines[0]
        if len(answer_lines) >= 2:
            final_answer = answer_lines[1]

        entries.append(
            {
                "chain_block_index": block_index,
                "qid": qid,
                "source_question": source_question,
                "model_question": model_question,
                "chain_titles": expected_titles,
                "chain_generated_answer": generated_answer,
                "chain_final_answer": final_answer,
            }
        )
    return entries


def remove_wh_words(text: str) -> str:
    wh_words = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
    words = [word for word in text.split(" ") if word.strip().lower() not in wh_words]
    return " ".join(words)


def is_para_closely_matching(
    existing_titles: List[str],
    existing_paras: List[str],
    new_title: str,
    new_para: str,
    fuzz_module,
    match_threshold: float = 90,
) -> bool:
    if new_title in existing_titles and new_para in existing_paras:
        return True
    for existing_title, existing_para in zip(existing_titles, existing_paras):
        title_matches = fuzz_module.ratio(existing_title, new_title) >= match_threshold
        paragraph_matches = fuzz_module.ratio(existing_para, new_para) >= match_threshold
        if title_matches and paragraph_matches:
            return True
    return False


def replay_retrieval(
    entry: Dict[str, Any],
    retrieval_config: Dict[str, Any],
    retriever_url: str,
    timeout: float,
    retries: int,
    requests_module,
    fuzz_module,
) -> Dict[str, Any]:
    query_source = retrieval_config.get("query_source", "last_answer")
    if query_source != "original_question":
        raise ValueError(f"ONER replay only supports query_source='original_question'; found {query_source!r}")
    if retrieval_config.get("retrieval_type") != "bm25":
        raise ValueError("ONER replay currently supports only BM25 retrieval")

    retrieval_query = remove_wh_words(entry["source_question"])
    retrieval_count = int(retrieval_config["retrieval_count"])
    selected_titles: List[str] = []
    selected_paras: List[str] = []
    selected_documents: List[Dict[str, Any]] = []
    raw_hit_count = 0
    skipped_long = 0
    skipped_duplicate = 0

    if retrieval_count == 0 or not retrieval_query.strip():
        return {
            "retrieval_query": retrieval_query,
            "raw_hit_count": 0,
            "skipped_long": 0,
            "skipped_duplicate": 0,
            "documents": [],
        }

    allowed_types = retrieval_config.get("allowed_paragraph_types") or [None]
    global_max = int(retrieval_config.get("global_max_num_paras", 100))
    dont_skip_long = bool(retrieval_config.get("dont_skip_long_paras", False))
    corpus_name = retrieval_config.get("source_corpus_name")
    document_type = retrieval_config.get("document_type", "title")

    for allowed_type in allowed_types:
        payload: Dict[str, Any] = {
            "retrieval_method": "retrieve_from_elasticsearch",
            "query_text": retrieval_query,
            "max_hits_count": retrieval_count,
            "corpus_name": corpus_name,
            "document_type": document_type,
        }
        if allowed_type is not None:
            payload["allowed_paragraph_types"] = [allowed_type]

        response = None
        last_error: Optional[Exception] = None
        for _ in range(max(1, retries)):
            try:
                response = requests_module.post(retriever_url, json=payload, timeout=timeout)
                response.raise_for_status()
                break
            except requests_module.RequestException as exc:
                last_error = exc
                response = None
        if response is None:
            raise RuntimeError(f"Retriever request failed after {retries} attempt(s): {last_error}")

        result = response.json()
        retrieval = result.get("retrieval")
        if not isinstance(retrieval, list):
            raise ValueError(f"Unexpected retriever response: {result}")
        raw_hit_count += len(retrieval)

        for item in retrieval:
            if item.get("corpus_name") != corpus_name:
                raise ValueError(
                    f"Retriever returned corpus {item.get('corpus_name')!r}; expected {corpus_name!r}"
                )
            title = item["title"]
            paragraph = item["paragraph_text"]
            if len(paragraph.split(" ")) > 600 and not dont_skip_long:
                skipped_long += 1
                continue
            if is_para_closely_matching(
                selected_titles, selected_paras, title, paragraph, fuzz_module=fuzz_module
            ):
                skipped_duplicate += 1
                continue
            if len(selected_paras) >= global_max:
                continue

            selected_titles.append(title)
            selected_paras.append(paragraph)
            selected_documents.append(copy.deepcopy(item))

    return {
        "retrieval_query": retrieval_query,
        "raw_hit_count": raw_hit_count,
        "skipped_long": skipped_long,
        "skipped_duplicate": skipped_duplicate,
        "documents": selected_documents,
    }


def para_to_text(title: str, paragraph: str, max_num_words: int) -> str:
    paragraph = " ".join(paragraph.split(" ")[:max_num_words])
    if paragraph.strip().startswith("Wikipedia Title: "):
        return paragraph.strip()
    return "Wikipedia Title: " + title + "\n" + paragraph.strip()


def infer_generator_tokenizer_name(answer_config: Dict[str, Any]) -> str:
    explicit_name = answer_config.get("tokenizer_model_name")
    if explicit_name:
        return explicit_name
    engine = answer_config.get("engine", "")
    if "qwen" in engine.lower():
        return "Qwen/Qwen2.5-7B-Instruct"
    return "gpt2"


def load_shared_prompt(answer_config: Dict[str, Any], read_prompt) -> Tuple[str, Path]:
    prompt_path = Path(answer_config["prompt_file"])
    if not prompt_path.is_absolute():
        prompt_path = REPO_ROOT / prompt_path
    prompt_path = prompt_path.resolve()
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompt_args = copy.deepcopy(answer_config.get("prompt_reader_args") or {})
    prompt_args["file_path"] = str(prompt_path)
    prompt = read_prompt(**prompt_args)
    return prompt, prompt_path


def render_prompt(
    entry: Dict[str, Any],
    documents: List[Dict[str, Any]],
    answer_config: Dict[str, Any],
    shared_prompt: str,
    fit_prompt_into_given_limit,
) -> Dict[str, Any]:
    if answer_config.get("shuffle_paras", False):
        raise ValueError("shuffle_paras=true cannot be reconstructed reliably from an ONER chain")
    if answer_config.get("key_info_type") is not None:
        raise ValueError("key_info_type is not supported by the ONER reconstructor")

    max_para_words = int(answer_config.get("max_para_num_words", 350))
    context_parts = [
        para_to_text(document["title"], document["paragraph_text"], max_para_words)
        for document in documents
    ]
    context_text = "\n\n".join(context_parts)

    question = entry["model_question"]
    question_prefix = answer_config.get("question_prefix", "")
    if question_prefix:
        question = question_prefix + question

    prompt_before_fit = shared_prompt + "\n"
    if context_text and answer_config.get("add_context", True):
        prompt_before_fit += "\n\n" + context_text
    prompt_before_fit += "\n\nQ: " + question + "\nA:"
    prompt_before_fit = prompt_before_fit.rstrip()

    max_tokens = int(answer_config.get("max_tokens", 300))
    model_length_limit = int(answer_config.get("model_tokens_limit", 120000))
    tokenizer_name = infer_generator_tokenizer_name(answer_config)
    full_prompt = fit_prompt_into_given_limit(
        original_prompt=prompt_before_fit,
        model_length_limit=model_length_limit,
        estimated_generation_length=max_tokens,
        demonstration_delimiter="\n\n\n",
        shuffle=False,
        remove_method=answer_config.get("remove_method", "first"),
        tokenizer_model_name=tokenizer_name,
        last_is_test_example=True,
    )

    return {
        "context_parts": context_parts,
        "context_text": context_text,
        "model_question": question,
        "question_suffix": "Q: " + question + "\nA:",
        "prompt_before_fit": prompt_before_fit,
        "full_prompt": full_prompt,
        "full_prompt_sha256": sha256_text(full_prompt),
        "prompt_changed_by_fit": full_prompt != prompt_before_fit,
        "prompt_fit": {
            "tokenizer_model_name": tokenizer_name,
            "model_tokens_limit": model_length_limit,
            "estimated_generation_length": max_tokens,
            "remove_method": answer_config.get("remove_method", "first"),
        },
        "request_model": answer_config.get("engine") or answer_config.get("model_name"),
        "request_messages": [{"role": "user", "content": full_prompt}],
    }


def atomic_write_json(path: Path, payload: Dict[str, Any], compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent), prefix=path.name + ".", suffix=".tmp", delete=False
    ) as file_obj:
        temp_path = Path(file_obj.name)
        json.dump(payload, file_obj, ensure_ascii=False, indent=None if compact else 2)
        file_obj.write("\n")
    os.replace(temp_path, path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay retrieval and reconstruct full ONER prompts for one prediction directory."
    )
    parser.add_argument("experiment_dir", type=Path, help="An ONER prediction directory under predictions/")
    parser.add_argument("--retriever-host", default="http://127.0.0.1")
    parser.add_argument("--retriever-port", type=int, default=8000)
    parser.add_argument("--output", type=Path, help="Output JSON path (defaults inside experiment_dir)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent retriever requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Only reconstruct the first N questions (for testing)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing reconstruction file")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON instead of indented JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero after writing if any question fails validation",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    jsonnet_module, requests_module, fuzz_module, fit_prompt, read_prompt = _load_dependencies()

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise SystemExit(f"Not a directory: {experiment_dir}")
    if not experiment_dir.name.startswith(("oner_", "oner_qa_")):
        raise SystemExit(f"This script only supports ONER directories: {experiment_dir.name}")

    config_path = find_one(experiment_dir, "config__*.jsonnet")
    chain_path = find_one(experiment_dir, "prediction__*_chains.txt")
    prediction_path = Path(str(chain_path).replace("_chains.txt", ".json"))
    if not prediction_path.exists():
        raise SystemExit(f"Prediction JSON corresponding to chain not found: {prediction_path}")
    full_eval_path_file = find_one(experiment_dir, "prediction__*_full_eval_path.txt")
    replacements_path = find_one(experiment_dir, "prediction__*_variable_replacements.json", required=False)

    replacements = read_variable_replacements(replacements_path)
    retriever_host = args.retriever_host.rstrip("/")
    retriever_url = f"{retriever_host}:{args.retriever_port}/retrieve/"
    config = load_config(
        config_path,
        replacements,
        retriever_host=retriever_host,
        retriever_port=args.retriever_port,
        jsonnet_module=jsonnet_module,
    )

    retrieval_config = config["models"].get("generate_titles")
    answer_config = config["models"].get("answer_main_question")
    if retrieval_config is None or answer_config is None:
        raise SystemExit("Saved config does not contain generate_titles and answer_main_question models")

    reader_config = config.get("reader", {})
    if reader_config.get("add_pinned_paras", False):
        raise SystemExit("Pinned paragraphs are not yet supported by the ONER reconstructor")

    evaluation_path = resolve_evaluation_path(experiment_dir, full_eval_path_file)
    evaluation_questions = load_evaluation_questions(evaluation_path)
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    chain_entries = parse_oner_chain(chain_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        chain_entries = chain_entries[: args.limit]

    shared_prompt, prompt_path = load_shared_prompt(answer_config, read_prompt=read_prompt)

    base_name = chain_path.name
    if base_name.startswith("prediction__"):
        base_name = base_name[len("prediction__") :]
    if base_name.endswith("_chains.txt"):
        base_name = base_name[: -len("_chains.txt")]
    if args.output:
        output_path = args.output.resolve()
    else:
        limit_suffix = f"__limit_{args.limit}" if args.limit is not None else ""
        output_path = experiment_dir / f"reconstructed_inputs__{base_name}{limit_suffix}.json"
    if output_path.exists() and not args.force:
        raise SystemExit(f"Output already exists; pass --force to overwrite: {output_path}")

    replay_results: List[Optional[Dict[str, Any]]] = [None] * len(chain_entries)

    def replay_at(index: int) -> Tuple[int, Dict[str, Any]]:
        return (
            index,
            replay_retrieval(
                chain_entries[index],
                retrieval_config=retrieval_config,
                retriever_url=retriever_url,
                timeout=args.timeout,
                retries=args.retries,
                requests_module=requests_module,
                fuzz_module=fuzz_module,
            ),
        )

    print(f"Replaying retrieval for {len(chain_entries)} question(s) via {retriever_url}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(replay_at, index): index for index in range(len(chain_entries))}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                result_index, replay_result = future.result()
                replay_results[result_index] = replay_result
            except Exception as exc:  # keep a complete audit file even when individual requests fail
                replay_results[index] = {"error": f"{type(exc).__name__}: {exc}", "documents": []}
            completed += 1
            if completed % 50 == 0 or completed == len(chain_entries):
                print(f"  retrieval {completed}/{len(chain_entries)}", flush=True)

    reconstructed_items: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    for index, (entry, replay_result) in enumerate(zip(chain_entries, replay_results)):
        assert replay_result is not None
        qid = entry["qid"]
        input_question = evaluation_questions.get(qid)
        replay_titles = [document["title"] for document in replay_result.get("documents", [])]
        question_match = input_question is not None and input_question == entry["source_question"]
        question_match_stripped = input_question is not None and input_question.strip() == entry["source_question"].strip()
        title_match = replay_titles == entry["chain_titles"]
        predicted_answer = predictions.get(qid)
        prediction_match = predicted_answer == entry.get("chain_final_answer")
        retrieval_error = replay_result.get("error")

        try:
            rendered = render_prompt(
                entry,
                documents=replay_result.get("documents", []),
                answer_config=answer_config,
                shared_prompt=shared_prompt,
                fit_prompt_into_given_limit=fit_prompt,
            )
            render_error = None
        except Exception as exc:
            rendered = None
            render_error = f"{type(exc).__name__}: {exc}"

        if retrieval_error or render_error or not question_match_stripped or not title_match:
            status = "failed"
        else:
            # The old chain stores titles but not document IDs, so historical byte-for-byte identity
            # cannot be proven even when the current replay agrees perfectly.
            status = "replayed_match"
        status_counts[status] = status_counts.get(status, 0) + 1

        reconstructed_items.append(
            {
                "index": index,
                "qid": qid,
                "source_question": entry["source_question"],
                "input_question": input_question,
                "chain": {
                    "titles": entry["chain_titles"],
                    "generated_answer": entry.get("chain_generated_answer"),
                    "final_answer": entry.get("chain_final_answer"),
                },
                "retrieval": replay_result,
                "rendered_input": rendered,
                "validation": {
                    "status": status,
                    "valid_for_token_count": status == "replayed_match",
                    "question_match_exact": question_match,
                    "question_match_after_strip": question_match_stripped,
                    "chain_titles_match": title_match,
                    "replayed_titles": replay_titles,
                    "prediction_answer_match": prediction_match,
                    "retrieval_error": retrieval_error,
                    "render_error": render_error,
                },
            }
        )

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "reconstruction_type": "oner",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_directory": str(experiment_dir),
        "retriever": {"url": retriever_url, "port": args.retriever_port},
        "source_files": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "chain": {"path": str(chain_path), "sha256": sha256_file(chain_path)},
            "prediction": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
            "evaluation": {"path": str(evaluation_path), "sha256": sha256_file(evaluation_path)},
            "prompt": {"path": str(prompt_path), "sha256": sha256_file(prompt_path)},
        },
        "config_summary": {
            "start_state": config.get("start_state"),
            "retrieval": retrieval_config,
            "answer_model": answer_config,
            "reader": reader_config,
            "variable_replacements": replacements,
        },
        "shared_prompt": {
            "text": shared_prompt,
            "sha256": sha256_text(shared_prompt),
        },
        "summary": {
            "question_count": len(reconstructed_items),
            "status_counts": status_counts,
            "all_valid_for_token_count": status_counts.get("failed", 0) == 0,
            "limited_run": args.limit is not None,
        },
        "items": reconstructed_items,
    }

    atomic_write_json(output_path, payload, compact=args.compact)
    print(f"Wrote reconstruction: {output_path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)

    if args.strict and status_counts.get("failed", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
