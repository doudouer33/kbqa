#!/usr/bin/env python3
"""Standalone BGE-M3 dense extraction / verification. No kbqa imports.

See https://huggingface.co/BAAI/bge-m3 and SentenceTransformer.encode docs.
Only the inference branch imports torch / sentence_transformers.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def read_queries(path, expected_count):
    rows = [
        json.loads(line, object_pairs_hook=unique_object)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    require(len(rows) == expected_count, "Wrong Query count: expected {}".format(expected_count))
    for row in rows:
        require(set(row) == {"question_id", "question_text"}, "Input must contain only question_id / question_text")
        require(all(isinstance(row[k], str) and row[k].strip() for k in row), "Empty ID or text")
    require(len({r["question_id"] for r in rows}) == len(rows), "Duplicate Question ID")
    return rows


def validate_arrays(question_ids, embeddings, expected_ids, expected_dim=1024, normalized=True):
    import numpy as np

    require(question_ids.ndim == 1 and question_ids.dtype.kind == "U", "Question IDs must be a 1-D Unicode array")
    ids = question_ids.tolist()
    require(len(ids) == len(set(ids)), "Duplicate embedding Question ID")
    require(set(ids) == set(expected_ids) and len(ids) == len(expected_ids), "Embedding / Query ID mismatch")
    require(embeddings.shape == (len(ids), expected_dim), "Wrong embedding shape: " + str(embeddings.shape))
    require(embeddings.dtype == np.dtype("float32"), "Embeddings must be float32")
    require(bool(np.isfinite(embeddings).all()), "Non-finite embedding values")
    if normalized:
        require(
            bool(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-4, atol=1e-5)), "Expected unit L2 norms"
        )


def load_verified(input_path, output_path, metadata_path=None, expected_count=1000, expected_dim=1024):
    """Validate provenance and align by ID; accepts arbitrary archive row order."""
    import numpy as np

    rows = read_queries(input_path, expected_count)
    metadata_path = Path(metadata_path or Path(output_path).with_name("embedding_metadata.json"))
    metadata = read_json(metadata_path)
    require(metadata["input_sha256"] == sha256(input_path), "Embedding input SHA256 mismatch")
    require(metadata["embeddings_sha256"] == sha256(output_path), "Embedding archive SHA256 mismatch")
    require(
        metadata["query_count"] == expected_count and metadata["embedding_dim"] == expected_dim,
        "Embedding metadata shape mismatch",
    )
    require(metadata["normalization"] in ("l2", "none"), "Unknown normalization policy")
    with np.load(output_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"question_ids", "embeddings"}, "Unexpected NPZ fields")
        ids, embeddings = archive["question_ids"], archive["embeddings"]
    validate_arrays(ids, embeddings, [r["question_id"] for r in rows], expected_dim, metadata["normalization"] == "l2")
    mapping = {qid: i for i, qid in enumerate(ids.tolist())}
    aligned = embeddings[[mapping[r["question_id"]] for r in rows]]
    return aligned, metadata


def extract(args):
    import numpy as np

    rows = read_queries(args.input, args.expected_count)
    require(args.batch_size > 0 and 0 < args.max_length <= 8192, "Invalid batch size / maximum length")
    require(
        args.device == "cpu" or (args.device.startswith("cuda:") and args.device[5:].isdigit()),
        "Use cpu or a single cuda:N device",
    )
    metadata_path = args.metadata or args.output.with_name("embedding_metadata.json")
    require(
        not args.output.exists() and not metadata_path.exists(),
        "Output already exists; choose a fresh output directory",
    )
    input_hash = sha256(args.input)
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        args.model,
        device=args.device,
        cache_folder=args.cache_folder,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    require(model.get_sentence_embedding_dimension() == 1024, "Expected 1024-D BGE-M3 model")
    require(
        getattr(model[1], "pooling_mode_cls_token", False),
        "BGE-M3 requires its official CLS pooling module; supply the complete model directory",
    )
    require(
        not any(
            getattr(model[1], name, False)
            for name in (
                "pooling_mode_mean_tokens",
                "pooling_mode_max_tokens",
                "pooling_mode_mean_sqrt_len_tokens",
                "pooling_mode_weightedmean_tokens",
                "pooling_mode_lasttoken",
            )
        ),
        "Unexpected BGE-M3 pooling configuration",
    )
    model.max_seq_length = args.max_length
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    texts = [r["question_text"] for r in rows]
    lengths = [len(ids) for ids in model.tokenizer(texts, truncation=False, padding=False)["input_ids"]]
    require(max(lengths) <= args.max_length, "Query would be truncated; increase --max-length")
    with torch.inference_mode():
        vectors = model.encode(
            texts,
            batch_size=args.batch_size,
            device=args.device,
            show_progress_bar=True,
            output_value="sentence_embedding",
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
            precision="float32",
            prompt="",
        )
    vectors = np.asarray(vectors, dtype=np.float32)
    ids = np.asarray([r["question_id"] for r in rows], dtype=np.str_)
    validate_arrays(ids, vectors, ids.tolist(), normalized=args.normalize)
    require(sha256(args.input) == input_hash, "Input changed during extraction")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, question_ids=ids, embeddings=vectors)
    transformer = model[0].auto_model
    model_config = transformer.config
    versions = {"python": platform.python_version()}
    for package in ("numpy", "torch", "sentence-transformers", "transformers", "huggingface-hub", "tokenizers"):
        versions[package] = importlib.metadata.version(package)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name_or_path": args.model,
        "model_local_path": str(Path(args.model).resolve()) if Path(args.model).is_dir() else None,
        "requested_revision": args.revision,
        "resolved_commit_hash": getattr(model_config, "_commit_hash", None),
        "model_config": model_config.to_dict(),
        "model_modules": str(model),
        "embedding_dim": 1024,
        "dtype": "float32",
        "batch_size": args.batch_size,
        "normalization": "l2" if args.normalize else "none",
        "pooling": "cls",
        "query_count": len(rows),
        "input_sha256": input_hash,
        "embeddings_sha256": sha256(args.output),
        "script_sha256": sha256(__file__),
        "max_length": args.max_length,
        "max_observed_query_tokens": max(lengths),
        "truncated_queries": 0,
        "prompt": "",
        "device": args.device,
        "inference_dtype": str(next(model.parameters()).dtype),
        "local_files_only": args.local_files_only,
        "cache_folder": args.cache_folder,
        "software_versions": versions,
        "command": [sys.executable] + sys.argv,
        "model_config_sha256": (
            {p.name: sha256(p) for p in sorted(Path(args.model).glob("*.json"))} if Path(args.model).is_dir() else {}
        ),
    }
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    load_verified(args.input, args.output, metadata_path, args.expected_count)
    print("Verified embeddings: shape={}, dtype=float32, metadata={}".format(vectors.shape, metadata_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="NPZ output path")
    parser.add_argument("--metadata", type=Path, help="Default: embedding_metadata.json beside NPZ")
    parser.add_argument("--model", default="BAAI/bge-m3", help="HF model ID or existing local model directory")
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0", help="cpu or a single cuda:N; no multi-GPU pool")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--cache-folder")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="Default: L2 normalization")
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument(
        "--verify-only", action="store_true", help="Validate existing outputs; never load/download a model"
    )
    args = parser.parse_args()
    if args.verify_only:
        vectors, metadata = load_verified(args.input, args.output, args.metadata, args.expected_count)
        print(
            json.dumps(
                {"shape": list(vectors.shape), "question_id_count": len(vectors), "metadata": metadata},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        extract(args)


if __name__ == "__main__":
    main()
