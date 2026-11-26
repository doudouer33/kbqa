"""Freeze and verify a level-by-type stratified split of HotpotQA Train 5000."""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from tempfile import NamedTemporaryFile

from prepare_data import DEFAULT_SETTINGS, HERE, LEVELS, QUESTION_COUNT, TYPES, provenance, read_json, read_rows, require
from analysis.hotpot_formal_runner.storage import atomic_json, text_hash  # noqa: E402

SPLITS = ("train", "val", "test")
ANALYSIS_SPLITS = SPLITS[:2]
STRATA = tuple(product(LEVELS, TYPES))
ALGORITHM = "joint quota rounding by minimum squared deviation, seeded within-stratum shuffle, source order output v1"


def load_settings(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="Existing data preparation settings")
    parser.add_argument("--label-settings", type=Path, default=HERE / "configs/label_settings.json")
    args = parser.parse_args()
    base = read_json(args.settings)["outputs"]
    config = read_json(args.label_settings)
    metrics = (HERE / base["per_question_metrics"]).resolve()
    report = (HERE / base["validation_report"]).resolve()
    split_dir, analysis_dir = ((HERE / config["outputs"][name]).resolve() for name in ("splits", "analysis"))
    require(all(HERE in path.parents for path in (split_dir, analysis_dir)) and split_dir != analysis_dir
            and split_dir not in analysis_dir.parents and analysis_dir not in split_dir.parents,
            "Split and analysis directories must be separate and inside the experiment")
    protected = (metrics, report, args.settings.resolve(), args.label_settings.resolve())
    require(not any(directory in path.parents or directory == path for directory in (split_dir, analysis_dir)
                    for path in protected), "Label outputs overlap an input path")
    return metrics, report, config, split_dir, analysis_dir


def read_metadata(metrics, report):
    validation = read_json(report)
    require(validation["status"] == "passed" and validation["question_count"] == QUESTION_COUNT,
            f"Input validation failed in {report}")
    source = provenance(metrics)
    require(source["sha256"] == validation["output"]["sha256"], "Per-question metrics hash differs from validation report")
    metadata = {}
    for row in read_rows(metrics, source["sha256"]):
        qid, stratum = row["qid"], (row["level"], row["type"])
        require(isinstance(qid, str) and qid and qid not in metadata, f"Invalid or duplicate QID {qid!r}")
        require(stratum in STRATA, f"Invalid stratum for QID {qid}")
        metadata[qid] = stratum
    require(len(metadata) == QUESTION_COUNT, f"Expected {QUESTION_COUNT} questions")
    return metadata, source


def split_counts(metadata, qids):
    by_stratum = Counter(metadata[qid] for qid in qids)
    return {
        "count": len(qids),
        "level": {level: sum(by_stratum[level, kind] for kind in TYPES) for level in LEVELS},
        "type": {kind: sum(by_stratum[level, kind] for level in LEVELS) for kind in TYPES},
        "level_type": {f"{level}/{kind}": by_stratum[level, kind] for level, kind in STRATA},
    }


def allocate(stratum_sizes, sizes):
    """Round each ideal cell to floor or ceil while preserving both margins."""
    total = sum(stratum_sizes)
    ideals = [tuple(count * size for size in sizes) for count in stratum_sizes]
    options = []
    for count, cell_numerators in zip(stratum_sizes, ideals):
        floors = tuple(value // total for value in cell_numerators)
        fractional = [i for i, value in enumerate(cell_numerators) if value % total]
        options.append([tuple(value + (i in extra) for i, value in enumerate(floors))
                        for extra in combinations(fractional, count - sum(floors))])
    feasible = (matrix for matrix in product(*options) if tuple(map(sum, zip(*matrix))) == sizes)

    def error(matrix):
        return sum((cell * total - ideal) ** 2 for row, targets in zip(matrix, ideals)
                   for cell, ideal in zip(row, targets))

    result = min(feasible, key=lambda matrix: (error(matrix), matrix), default=None)
    require(result is not None, "No stratified quota allocation satisfies the requested split sizes")
    return result


def new_assignments(metadata, config):
    strata = defaultdict(list)
    for qid, stratum in metadata.items():
        strata[stratum].append(qid)
    sizes = tuple(config["sizes"][split] for split in SPLITS)
    quotas = allocate([len(strata[stratum]) for stratum in STRATA], sizes)
    randomizer = random.Random(config["seed"])
    assignments = {}
    for stratum, counts in zip(STRATA, quotas):
        ids = strata[stratum]
        randomizer.shuffle(ids)
        start = 0
        for split, count in zip(SPLITS, counts):
            assignments.update((qid, split) for qid in ids[start:start + count])
            start += count
    result = {split: [] for split in SPLITS}
    for qid in metadata:
        result[assignments[qid]].append(qid)
    return result


def manifest_content(metadata, source, config, assignments):
    seen = set()
    for split in SPLITS:
        qids = assignments[split]
        unique = set(qids)
        require(len(qids) == len(unique) == config["sizes"][split], f"Duplicate or wrong number of QIDs in {split}")
        require(not seen & unique and unique <= metadata.keys(), f"Overlapping or unknown QIDs in {split}")
        seen.update(unique)
    require(seen == metadata.keys(), "Split union differs from the input QID set")
    return {
        "schema_version": 1, "algorithm": ALGORITHM, "input": source, "configuration": config,
        "splits": {split: {"stats": split_counts(metadata, qids), "qid_jsonl_sha256": text_hash(qid_jsonl(qids)), "qids": qids}
                   for split, qids in assignments.items()},
    }


def qid_jsonl(qids):
    return "".join(json.dumps({"qid": qid}, ensure_ascii=False) + "\n" for qid in qids)


def install_frozen(path, content):
    """Publish a complete new file without replacing an existing frozen file."""
    if not path.exists():
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp") as temp:
            temp.write(content)
            temp.flush()
            try:
                os.link(temp.name, path)
            except FileExistsError:
                pass
    require(path.read_bytes() == content, f"Frozen file differs: {path}")


def get_manifest(metadata, source, config, directory, create=False):
    require(config["stratify_by"] == ["level", "type"] and type(config["seed"]) is int, "Invalid stratification or seed")
    sizes = config["sizes"]
    require(set(sizes) == set(SPLITS) and all(type(n) is int and n > 0 for n in sizes.values())
            and sum(sizes.values()) == len(metadata), "Invalid split sizes")
    path = directory / "split_manifest.json"
    if path.exists():
        manifest = read_json(path)
        assignments = {split: manifest["splits"][split]["qids"] for split in SPLITS}
        require(manifest == manifest_content(metadata, source, config, assignments), f"Frozen split manifest differs: {path}")
    else:
        require(create, f"Missing {path}. Run split_data.py first")
        assignments = new_assignments(metadata, config)
        manifest = manifest_content(metadata, source, config, assignments)
    qid_files = {directory / f"{split}.jsonl": qid_jsonl(qids).encode("utf-8") for split, qids in assignments.items()}
    for qid_path, content in qid_files.items():
        if qid_path.exists() or not create:
            require(qid_path.read_bytes() == content, f"Frozen QID file differs: {qid_path}")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            install_frozen(path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        for qid_path, content in qid_files.items():
            install_frozen(qid_path, content)
    return manifest


def main():
    metrics, report, config, split_dir, analysis_dir = load_settings(__doc__)
    metadata, source = read_metadata(metrics, report)
    manifest = get_manifest(metadata, source, config["split"], split_dir, create=True)
    statistics = {"input": source, "manifest": provenance(split_dir / "split_manifest.json"),
                  "configuration": manifest["configuration"], "algorithm": manifest["algorithm"],
                  "all": split_counts(metadata, list(metadata)),
                  "splits": {split: row["stats"] for split, row in manifest["splits"].items()},
                  "analysis_splits": list(ANALYSIS_SPLITS)}
    atomic_json(analysis_dir / "split_statistics.json", statistics)
    print("Frozen splits verified:", {split: val["count"] for split, val in statistics["splits"].items()})


if __name__ == "__main__":
    try:
        main()
    except (OSError, TypeError, ValueError, KeyError) as error:
        sys.exit(f"Split failed: {error}")
