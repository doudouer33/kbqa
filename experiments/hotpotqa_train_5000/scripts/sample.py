"""Freeze an unstratified sample, preserving each source JSONL record byte for byte."""
import argparse
import hashlib
import json
import random
from collections import Counter

from common import DATA, ROOT, artifact_path, atomic_json, atomic_text, configured, now, read_json, read_rows, require, settings, sha256


def validate_row(row):
    required = {"dataset", "question_id", "question_text", "level", "type", "answers_objects", "contexts"}
    require(required <= row.keys(), "Missing preprocessed fields: " + str(required - row.keys()))
    require(row["dataset"] == "hotpotqa", "Unexpected dataset")
    require(isinstance(row["question_id"], str) and bool(row["question_id"].strip()), "Empty qid")
    require(isinstance(row["question_text"], str) and bool(row["question_text"].strip()), "Empty question")
    require(row["level"] in ("easy", "medium", "hard"), "Invalid level")
    require(row["type"] in ("bridge", "comparison"), "Invalid type")
    require(isinstance(row["answers_objects"], list) and bool(row["answers_objects"]), "Missing answers")
    for answer in row["answers_objects"]:
        require({"number", "date", "spans"} <= answer.keys(), "Incomplete answer")
        require({"day", "month", "year"} <= answer["date"].keys(), "Incomplete answer date")
        require(isinstance(answer["spans"], list), "Invalid answer spans")
        require(bool(answer["number"] or answer["spans"] or any(answer["date"].values())), "Empty answer")
    require(isinstance(row["contexts"], list) and bool(row["contexts"]), "Missing contexts")
    for context in row["contexts"]:
        require({"idx", "title", "paragraph_text", "is_supporting"} <= context.keys(), "Incomplete context")
        require(isinstance(context["idx"], int) and isinstance(context["is_supporting"], bool), "Invalid context metadata")
        require(isinstance(context["title"], str) and isinstance(context["paragraph_text"], str), "Invalid context text")


def distributions(rows):
    return {
        "level": dict(sorted(Counter(row["level"] for row in rows).items())),
        "type": dict(sorted(Counter(row["type"] for row in rows).items())),
        "level_type": dict(sorted(Counter(row["level"] + "/" + row["type"] for row in rows).items())),
    }


def fewshots():
    cfg = configured(ROOT / settings()["baseline"])
    qa = cfg["models"]["answer_main_question"]
    ids = qa["prompt_reader_args"]["filter_by_key_values"]["qid"]
    require(len(ids) == len(set(ids)) == 15, "Expected the formal 15 Few-shot IDs")
    prompt = ROOT / qa["prompt_file"]
    metadata = [json.loads(line[len("# METADATA: "):]) for line in prompt.read_text().splitlines()
                if line.startswith("# METADATA: ")]
    for qid in ids:
        require(sum(row.get("qid") == qid for row in metadata) == 1, "Few-shot ID missing or repeated in prompt: " + qid)
    return ids, {"path": str(prompt.relative_to(ROOT)), "sha256": sha256(prompt)}


def scan_source(path, excluded):
    seen, candidates, excluded_present, metadata = set(), [], [], []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            digest.update(line)
            require(bool(line.strip()), "Blank source record at byte " + str(offset))
            row = json.loads(line)
            validate_row(row)
            qid = row["question_id"]
            require(qid not in seen, "Duplicate source ID: " + qid)
            seen.add(qid)
            metadata.append({"level": row["level"], "type": row["type"]})
            if qid in excluded:
                excluded_present.append(qid)
            else:
                candidates.append((qid, offset))
    return candidates, excluded_present, metadata, digest.hexdigest()


def freeze():
    cfg = settings()
    data, manifest = DATA / "train_5000.jsonl", DATA / "sample_manifest.json"
    if data.exists() or manifest.exists():
        verify()
        print("Frozen sample verified; no resampling.")
        return
    ids, prompt = fewshots()
    source = ROOT / cfg["source"]
    candidates, excluded, source_meta, source_hash = scan_source(source, set(ids))
    chosen = random.Random(cfg["seed"]).sample(candidates, cfg["sample_size"])
    raw_lines, rows = [], []
    with source.open("rb") as stream:
        for qid, offset in chosen:
            stream.seek(offset)
            line = stream.readline()
            row = json.loads(line)
            require(row["question_id"] == qid, "Source changed while sampling")
            raw_lines.append(line.decode("utf-8"))
            rows.append(row)
    require(sha256(source) == source_hash, "Source changed while sampling")
    DATA.mkdir(parents=True, exist_ok=True)
    atomic_text(data, "".join(raw_lines))
    atomic_json(manifest, {
        "schema_version": 1, "created_at_utc": now(), "seed": cfg["seed"], "sample_size": len(rows),
        "method": "random.Random(seed).sample(source_order_after_fewshot_exclusion, size); no stratification",
        "source": {"path": cfg["source"], "sha256": source_hash, "count": len(source_meta)},
        "sample": {"path": str(data.relative_to(ROOT)), "sha256": sha256(data)},
        "baseline": {"path": cfg["baseline"], "sha256": sha256(ROOT / cfg["baseline"])},
        "fewshot_prompt": prompt, "fewshot_ids": ids, "excluded_source_ids": excluded,
        "eligible_count": len(candidates), "fewshot_overlap": [],
        "question_ids": [row["question_id"] for row in rows],
        "distributions": distributions(rows), "source_distributions": distributions(source_meta),
        "source_fields": sorted(rows[0]), "preservation": "byte-identical source JSONL lines in sampled order",
    })
    verify()
    print("Frozen 5000-question sample:", distributions(rows))


def verify():
    cfg = settings()
    manifest = read_json(DATA / "sample_manifest.json")
    require(manifest["seed"] == cfg["seed"] == 42 and manifest["sample_size"] == cfg["sample_size"] == 5000, "Sampling settings changed")
    for key in ("source", "sample", "baseline", "fewshot_prompt"):
        item = manifest[key]
        require(sha256(artifact_path(item["path"])) == item["sha256"], key + " SHA256 differs")
    rows = read_rows(artifact_path(manifest["sample"]["path"]))
    qids = [row["question_id"] for row in rows]
    require(len(rows) == len(set(qids)) == 5000, "Sample size or ID uniqueness failed")
    require(qids == manifest["question_ids"], "Manifest IDs/order differ")
    ids, prompt = fewshots()
    require(ids == manifest["fewshot_ids"] and prompt == manifest["fewshot_prompt"], "Few-shot selection changed")
    require(not set(qids) & set(ids), "Sample overlaps Few-shot IDs")
    for row in rows:
        validate_row(row)
    require(distributions(rows) == manifest["distributions"], "Distribution differs")
    # Audit raw-line equality and deterministically reproduce selection without rewriting it.
    candidates, excluded, source_meta, source_hash = scan_source(ROOT / cfg["source"], set(ids))
    expected = random.Random(42).sample(candidates, 5000)
    require([qid for qid, _ in expected] == qids, "Seed-42 source-order sampling differs")
    require(source_hash == manifest["source"]["sha256"], "Source changed during verification")
    require(excluded == manifest["excluded_source_ids"] and len(candidates) == manifest["eligible_count"], "Exclusion differs")
    require(distributions(source_meta) == manifest["source_distributions"], "Source distribution differs")
    with (ROOT / cfg["source"]).open("rb") as source, (DATA / "train_5000.jsonl").open("rb") as sample:
        for _, offset in expected:
            source.seek(offset)
            require(source.readline() == sample.readline(), "Sample record is not byte-identical to source")
        require(not sample.read(), "Unexpected sample suffix")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify()
        print("PASS: sample count, schema, IDs, distributions, exclusions, hashes, source bytes and seed.")
    else:
        freeze()
