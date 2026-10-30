#!/usr/bin/env python3
"""Build a minimal standalone upload directory without labels or model files."""

import argparse
import shutil
from pathlib import Path

from common import (
    HERE,
    fresh_directory,
    load_config,
    read_json,
    read_rows,
    require,
    resolve,
    sha256,
    validate_queries,
    write_json,
)


def pack(config, output):
    prepared = resolve(config["prepared"])
    queries = prepared / "queries.jsonl"
    validate_queries(read_rows(queries), config["expected_count"])
    query_metadata = read_json(prepared / "query_metadata.json")
    require(sha256(queries) == query_metadata["queries_sha256"], "Query checksum changed")
    output = fresh_directory(output)
    sources = {
        "extract_embeddings.py": HERE / "extract_embeddings.py",
        "queries.jsonl": queries,
        "requirements-embedding.txt": HERE / "requirements-embedding.txt",
        "README.md": HERE / "A800_README.md",
        "a800_env.sh": HERE / "a800_env.sh",
    }
    for name, source in sources.items():
        shutil.copyfile(source, output / name)
    write_json(
        output / "job_manifest.json",
        {
            "query_count": config["expected_count"],
            "embedding_dim": config["embedding_dim"],
            "source_dataset_sha256": query_metadata["source_sha256"],
            "queries_sha256": sha256(queries),
            "a800_root": "/nas/luoyichen",
            "files_sha256": {name: sha256(output / name) for name in sources},
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--output", type=Path, default=HERE / "staging" / "a800_embedding_job")
    args = parser.parse_args()
    print("Prepared standalone job: " + str(pack(load_config(args.config), args.output)))


if __name__ == "__main__":
    main()
