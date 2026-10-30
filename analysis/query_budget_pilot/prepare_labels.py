#!/usr/bin/env python3
"""Reuse oracle functions; save three targets, offline curves and ID splits."""

import argparse
import math
from collections import Counter
from pathlib import Path

from common import (
    HERE,
    LABELS,
    by_id,
    load_config,
    min_feasible_k,
    read_json,
    read_rows,
    require,
    resolve,
    sha256,
    source_questions,
    split_counts,
    stratified_split,
    target_labels,
    validate_queries,
    validate_split,
    write_json,
    write_jsonl,
    build_label,
    Parameters,
)


def prepare(config, output):
    require(
        not any(
            (output / name).exists()
            for name in ("metadata.json", "labels.jsonl", "offline_curves.jsonl", "splits", "validation.json")
        ),
        "Prepared labels already exist",
    )
    questions, validation, source_metadata, old = source_questions(config)
    queries = validate_queries(read_rows(output / "queries.jsonl"), config["expected_count"])
    require(set(queries) == set(old), "Query / source ID mismatch")
    old_minimum = by_id(read_rows(resolve(config["source_minimum"])), "minimum reference")
    require(set(old_minimum) == set(queries), "Minimum reference ID mismatch")
    labels, curves = [], []
    for q in questions:
        require(queries[q.question_id]["question_text"] == q.question_text, "Query text differs from source")
        targets = target_labels(q, config)
        reference = old[q.question_id]
        target = targets["original_soft"]
        require(reference["label_status"] == target["status"], "Old label status mismatch")
        if target["probabilities"] is not None:
            require(
                all(
                    math.isclose(a, b, abs_tol=1e-12, rel_tol=0)
                    for a, b in zip(target["probabilities"], reference["soft_label"])
                ),
                "Old soft probability mismatch",
            )
        params = Parameters.from_dict(reference["parameters"])
        minimum = min_feasible_k(q, build_label(q, params))
        saved_minimum = old_minimum[q.question_id]["min_feasible_k"]
        require(saved_minimum == (str(minimum) if minimum is not None else ""), "Old minimum mismatch")
        labels.append(
            {
                "question_id": q.question_id,
                "type": q.type,
                "has_valid_label": target["status"] == "valid",
                "targets": targets,
            }
        )
        curves.append(
            {
                "question_id": q.question_id,
                "type": q.type,
                "k_values": list(q.ks),
                "f1": q.f1,
                "context_tokens": q.tokens["context_tokens"],
                "prompt_tokens": q.tokens["prompt_tokens"],
                "evidence_recall": [e["sentence_recall_lower"] for e in q.evidence],
            }
        )
    # Only IDs, types and validity enter splitting. Target vectors are joined by ID during training.
    identities = [(r["question_id"], r["type"], r["has_valid_label"]) for r in labels]
    label_map = by_id(labels, "prepared labels")
    splits = {seed: stratified_split(identities, seed) for seed in config["seeds"]}
    for split in splits.values():
        validate_split(split, queries)
        split["counts"] = split_counts(split, label_map)
    write_jsonl(output / "labels.jsonl", labels)
    write_jsonl(output / "offline_curves.jsonl", curves)
    (output / "splits").mkdir(exist_ok=False)
    for seed, split in splits.items():
        write_json(output / "splits" / ("seed_{}.json".format(seed)), split)
    status_counts = dict(Counter(r["targets"]["original_soft"]["status"] for r in labels))
    artifacts = [
        output / "queries.jsonl",
        output / "query_metadata.json",
        output / "labels.jsonl",
        output / "offline_curves.jsonl",
    ] + sorted((output / "splits").glob("*.json"))
    metadata = {
        "pilot_configuration": config,
        "query_count": len(labels),
        "status_counts": status_counts,
        "labels": config["labels"],
        "constraints": {
            "epsilon": 0.1,
            "tau": 0.5,
            "evidence": "sentence_recall_lower",
            "cost": "context_tokens",
            "k": list(range(1, 16)),
        },
        "oracle_source_metadata": source_metadata,
        "source_validation": validation,
        "reference_checks": {"old_soft_labels": len(labels), "old_minimum": len(labels)},
        "source_labels_sha256": sha256(resolve(config["source_labels"])),
        "source_minimum_sha256": sha256(resolve(config["source_minimum"])),
        "pilot_code_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        "artifacts_sha256": {str(p.relative_to(output)): sha256(p) for p in artifacts},
        "seeds": config["seeds"],
        "targets": list(LABELS),
        "study_scope": "Exploratory learnability on method-design HotpotQA Dev; not independent generalization evidence",
    }
    write_json(output / "metadata.json", metadata)
    write_json(
        output / "validation.json",
        {
            "passed": True,
            "query_count": len(labels),
            "status_counts": status_counts,
            "split_counts": {str(s): v["counts"] for s, v in splits.items()},
            "source_ids_equal": True,
            "all_targets_share_validity": True,
            "split_overlap": 0,
            "old_labels_and_minimum_match": True,
        },
    )
    # Re-read persisted artifacts, hashes, targets and every split.
    from common import load_prepared

    load_prepared(output, config["expected_count"])
    for seed in config["seeds"]:
        validate_split(read_json(output / "splits" / ("seed_{}.json".format(seed))), queries)
    return status_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    print(prepare(config, args.output or resolve(config["prepared"])))


if __name__ == "__main__":
    main()
