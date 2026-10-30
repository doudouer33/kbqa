#!/usr/bin/env python3
"""Export deterministic, label-free queries; preserve original question text."""

import argparse
from pathlib import Path

from common import (
    HERE,
    by_id,
    load_config,
    read_json,
    read_rows,
    require,
    resolve,
    sha256,
    validate_queries,
    write_json,
    write_jsonl,
)


def prepare(config, output):
    base = read_json(resolve(config["base_config"]))
    source = resolve(base["inputs"]["dataset"])
    original = by_id(read_rows(source), "dataset")
    oracle = by_id(read_rows(resolve(config["source_labels"])), "Soft Oracle")
    queries = [{"question_id": qid, "question_text": original[qid]["question_text"]} for qid in sorted(original)]
    validate_queries(queries, config["expected_count"])
    require(set(original) == set(oracle), "Query / Soft Oracle IDs differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists() and not output.with_name("query_metadata.json").exists(), "Query output already exists")
    write_jsonl(output, queries)
    metadata = {
        "query_count": len(queries),
        "unique_question_ids": len(original),
        "id_sets_match": True,
        "order": "question_id ascending; text preserved byte-for-byte after JSON decoding",
        "source_path": str(source),
        "source_sha256": sha256(source),
        "soft_oracle_sha256": sha256(resolve(config["source_labels"])),
        "queries_sha256": sha256(output),
    }
    write_json(output.with_name("query_metadata.json"), metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or resolve(config["prepared"]) / "queries.jsonl"
    print(prepare(config, output))


if __name__ == "__main__":
    main()
