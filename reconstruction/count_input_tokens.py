#!/usr/bin/env python3
"""Count retrieved-context tokens from one reconstruction file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Dict, List, Optional


SUPPORTED_MODEL_PREFIX = "gpt-4o-mini"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_reconstruction(experiment_dir: Path) -> Path:
    matches = sorted(
        path
        for path in experiment_dir.glob("reconstructed_inputs__*.json")
        if "__limit_" not in path.stem
    )
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one complete reconstructed_inputs JSON in "
            f"{experiment_dir}, found {len(matches)}: {[str(path) for path in matches]}"
        )
    return matches[0]


def encode_length(encoding, value: str) -> int:
    return len(encoding.encode(value, disallowed_special=()))


def summarize(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {
            "total": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "total": sum(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def atomic_write_json(path: Path, payload: Dict[str, Any], compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as file_obj:
        temp_path = Path(file_obj.name)
        json.dump(payload, file_obj, ensure_ascii=False, indent=None if compact else 2)
        file_obj.write("\n")
    os.replace(temp_path, path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count retrieved-context tokens for one reconstructed ONER prediction directory."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument(
        "--reconstruction",
        type=Path,
        help="Explicit reconstruction JSON; defaults to the complete file in experiment_dir",
    )
    parser.add_argument("--output", type=Path, help="Output JSON path inside experiment_dir by default")
    parser.add_argument("--model", help="Override the request model stored in the reconstruction")
    parser.add_argument(
        "--limit",
        type=int,
        help="Only count the first N reconstructed questions and write a separate limited-run output",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing token-count file")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit(
            "tiktoken is not installed in this Python environment. Install it in the environment used to run this script."
        ) from exc

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise SystemExit(f"Not a directory: {experiment_dir}")

    reconstruction_path = (
        args.reconstruction.resolve() if args.reconstruction else find_reconstruction(experiment_dir)
    )
    if not reconstruction_path.is_file():
        raise SystemExit(f"Reconstruction JSON not found: {reconstruction_path}")

    data = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    if data.get("reconstruction_type") != "oner" or not isinstance(data.get("items"), list):
        raise SystemExit(f"Not a supported ONER reconstruction file: {reconstruction_path}")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    selected_items = data["items"][: args.limit] if args.limit is not None else data["items"]

    stored_models = {
        item.get("rendered_input", {}).get("request_model")
        for item in data["items"]
        if item.get("rendered_input") is not None
        and item.get("rendered_input", {}).get("request_model")
    }
    if args.model:
        model = args.model
    elif len(stored_models) == 1:
        model = next(iter(stored_models))
    else:
        raise SystemExit(
            f"Could not infer one request model from reconstruction: {sorted(stored_models)}; pass --model"
        )

    if not model.startswith(SUPPORTED_MODEL_PREFIX):
        raise SystemExit(
            f"This counter currently supports only {SUPPORTED_MODEL_PREFIX}; found {model!r}"
        )
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError as exc:
        raise SystemExit(f"tiktoken does not recognize model {model!r}") from exc

    suffix = reconstruction_path.name[len("reconstructed_inputs__") :]
    if args.output:
        output_path = args.output.resolve()
    else:
        suffix_stem = suffix[: -len(".json")] if suffix.endswith(".json") else suffix
        limit_suffix = f"__limit_{args.limit}" if args.limit is not None else ""
        output_path = experiment_dir / (
            f"retrieval_context_token_counts__{suffix_stem}{limit_suffix}.json"
        )
    if output_path.exists() and not args.force:
        raise SystemExit(f"Output already exists; pass --force to overwrite: {output_path}")

    token_counts: Dict[str, int] = {}
    skipped: List[Dict[str, str]] = []

    for item in selected_items:
        qid = item.get("qid")
        if not isinstance(qid, str) or not qid:
            raise ValueError("Every reconstruction item must contain a non-empty string qid")
        if qid in token_counts or any(entry["qid"] == qid for entry in skipped):
            raise ValueError(f"Duplicate qid in reconstruction: {qid}")

        validation = item.get("validation") or {}
        if not validation.get("valid_for_token_count", False):
            skipped.append(
                {
                    "qid": qid,
                    "reason": "reconstruction_validation_failed",
                    "status": str(validation.get("status", "unknown")),
                }
            )
            continue

        rendered_input: Optional[Dict[str, Any]] = item.get("rendered_input")
        if rendered_input is None or not isinstance(rendered_input.get("context_text"), str):
            skipped.append(
                {
                    "qid": qid,
                    "reason": "context_text_missing",
                    "status": str(validation.get("status", "unknown")),
                }
            )
            continue

        token_counts[qid] = encode_length(encoding, rendered_input["context_text"])

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "count_type": "retrieved_context_tokens",
        "model": model,
        "encoding": encoding.name,
        "counting_rules": {
            "source_field": "rendered_input.context_text",
            "note": "Counts only the reconstructed retrieved-document context; excludes few-shot examples, the question, and chat-message framing.",
        },
        "source_reconstruction": {
            "path": str(reconstruction_path),
            "sha256": sha256_file(reconstruction_path),
        },
        "limited_run": args.limit is not None,
        "limit": args.limit,
        "summary": {
            "source_question_count": len(data["items"]),
            "selected_question_count": len(selected_items),
            "counted_question_count": len(token_counts),
            "skipped_question_count": len(skipped),
            "context_tokens": summarize(list(token_counts.values())),
        },
        "token_counts": token_counts,
        "skipped": skipped,
    }
    atomic_write_json(output_path, payload, compact=args.compact)

    print(f"Wrote token counts: {output_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
