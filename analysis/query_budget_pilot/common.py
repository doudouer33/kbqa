"""Read-only reuse of the established oracle; no RAG runtime imports."""

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "soft_oracle_1000"))
from build_soft_oracle import (  # noqa: E402
    Parameters,
    build_label,
    context,
    index_rows,
    read_json,
    read_rows,
    require,
    sha256,
    write_json as write_json,
)
from followup_diagnostics.analyze import min_feasible_k  # noqa: E402

LABELS = ("minimum_feasible", "original_soft", "sharper_soft")
MODELS = ("linear", "mlp")
PARTITIONS = ("train", "validation", "test")


def resolve(path):
    return (ROOT / path).resolve()


def load_config(path=None):
    path = Path(path or HERE / "config.json").resolve()
    config = read_json(path)
    require(set(config["labels"]) == set(LABELS), "Expected all three supervision targets")
    require(config["embedding_dim"] == 1024, "BGE-M3 dense dimension must be 1024")
    require(len(set(config["seeds"])) == len(config["seeds"]) >= 3, "Need at least three unique seeds")
    require(config["split_ratios"] == [0.7, 0.1, 0.2], "Expected 70/10/20 split")
    return config


def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def allocate_output(prefix):
    parent = HERE / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, 100000):
        path = parent / (prefix + "_{:03d}".format(i))
        if not path.exists():
            return path
    raise ValueError("No available output directory")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def by_id(rows, source):
    return {key[0]: row for key, row in index_rows(rows, ("question_id",), source).items()}


def validate_queries(rows, expected_count):
    indexed = by_id(rows, "queries")
    require(len(indexed) == expected_count, "Unexpected Query count")
    for row in rows:
        require(set(row) == {"question_id", "question_text"}, "Queries must contain only ID and text")
        require(isinstance(row["question_text"], str) and row["question_text"].strip(), "Empty Query text")
    return indexed


def source_questions(config):
    _, _, questions, validation, metadata = context(resolve(config["base_config"]))
    require(len(questions) == config["expected_count"], "Unexpected source count")
    old = by_id(read_rows(resolve(config["source_labels"])), "original labels")
    require(set(old) == {q.question_id for q in questions}, "Soft Oracle ID set mismatch")
    return questions, validation, metadata, old


def target_labels(question, config):
    results = {}
    reference = dict(
        epsilon=0.1, tau=0.5, scheme="D", evidence_metric="sentence_recall_lower", cost_metric="context_tokens"
    )
    for name in LABELS:
        params = Parameters.from_dict(dict(reference, **config["labels"][name]))
        result = build_label(question, params)
        target = result["soft_label"]
        if name == "minimum_feasible" and target is not None:
            minimum = min_feasible_k(question, result)
            target = tuple(float(k == minimum) for k in question.ks)
        results[name] = {"status": result["label_status"], "probabilities": target}
    require(len({r["status"] for r in results.values()}) == 1, "Target validity mismatch")
    return results


def stratified_split(identities, seed):
    """Allocate ID memberships before joining targets; type x valid strata only.

    Largest remainder allocation gives exact global sizes, with at most one
    sample of rounding per stratum in each sequential allocation.
    """
    buckets = defaultdict(list)
    for qid, kind, valid in sorted(identities):
        buckets[(kind, bool(valid))].append(qid)
    sizes = {key: len(value) for key, value in buckets.items()}
    n = sum(sizes.values())

    def apportion(fraction, target, capacities):
        ideal = {key: sizes[key] * fraction for key in sizes}
        counts = {key: min(int(ideal[key]), capacities[key]) for key in sizes}
        while sum(counts.values()) < target:
            available = [key for key in sorted(sizes) if counts[key] < capacities[key]]
            require(bool(available), "Split allocation has no capacity")
            key = max(available, key=lambda k: ideal[k] - counts[k])
            counts[key] += 1
        return counts

    train_counts = apportion(0.7, int(n * 0.7), sizes)
    val_counts = apportion(0.1, int(n * 0.1), {k: sizes[k] - train_counts[k] for k in sizes})
    rng = random.Random(seed)
    partitions = {name: [] for name in PARTITIONS}
    for key in sorted(buckets):
        values = buckets[key][:]
        rng.shuffle(values)
        a, b = train_counts[key], train_counts[key] + val_counts[key]
        for name, group in zip(PARTITIONS, (values[:a], values[a:b], values[b:])):
            partitions[name].extend(group)
    return {
        "seed": seed,
        "stratification": "type x has_valid_label",
        "partitions": {k: sorted(v) for k, v in partitions.items()},
    }


def validate_split(split, expected_ids):
    parts = split["partitions"]
    require(set(parts) == set(PARTITIONS), "Missing split partition")
    flat = [qid for name in PARTITIONS for qid in parts[name]]
    require(len(flat) == len(set(flat)), "Split Question ID overlap/duplicate")
    require(set(flat) == set(expected_ids), "Split does not cover exactly all Query IDs")
    n = len(flat)
    require(
        [len(parts[p]) for p in PARTITIONS] == [int(n * 0.7), int(n * 0.1), n - int(n * 0.7) - int(n * 0.1)],
        "Wrong split sizes",
    )


def load_prepared(path, expected_count=None):
    path = Path(path)
    manifest = read_json(path / "metadata.json")
    for relative, checksum in manifest["artifacts_sha256"].items():
        require(sha256(path / relative) == checksum, "Prepared artifact changed: " + relative)
    queries = read_rows(path / "queries.jsonl")
    qmap = validate_queries(queries, expected_count or manifest["query_count"])
    labels = by_id(read_rows(path / "labels.jsonl"), "pilot labels")
    curves = by_id(read_rows(path / "offline_curves.jsonl"), "offline curves")
    require(set(qmap) == set(labels) == set(curves), "Query/label/curve ID mismatch")
    for qid, row in labels.items():
        require(set(row["targets"]) == set(LABELS), "Missing target scheme")
        for target in row["targets"].values():
            probabilities = target["probabilities"]
            if row["has_valid_label"]:
                require(target["status"] == "valid" and probabilities is not None, "Missing valid target")
                require(
                    len(probabilities) == 15
                    and all(isinstance(p, (float, int)) and 0 <= p <= 1 for p in probabilities),
                    "Bad target vector",
                )
                require(abs(sum(probabilities) - 1) < 1e-6, "Target must sum to one")
            else:
                require(
                    target["status"] in ("empty_candidates", "all_zero_f1") and probabilities is None,
                    "Invalid target must be null",
                )
        require(row["type"] == curves[qid]["type"] in ("bridge", "comparison"), "Type mismatch")
        curve = curves[qid]
        require(curve["k_values"] == list(range(1, 16)), "Incomplete offline k curve")
        for field in ("f1", "context_tokens", "prompt_tokens", "evidence_recall"):
            values = curve[field]
            require(
                len(values) == 15
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values),
                "Invalid offline curve: " + field,
            )
            if field in ("f1", "evidence_recall"):
                require(all(0 <= v <= 1 for v in values), "Offline metric outside [0,1]")
            else:
                require(
                    all(v == int(v) and v >= (1 if field == "context_tokens" else 0) for v in values),
                    "Invalid token cost",
                )
    return queries, labels, curves, manifest


def fingerprint(prepared, embedding_path, metadata_path, seed):
    return {
        str(Path(p).name): sha256(p)
        for p in (
            Path(prepared) / "metadata.json",
            Path(prepared) / "queries.jsonl",
            Path(prepared) / "labels.jsonl",
            Path(prepared) / "offline_curves.jsonl",
            Path(prepared) / "splits" / ("seed_{}.json".format(seed)),
            embedding_path,
            metadata_path,
        )
    }


def split_counts(split, labels):
    return {
        name: dict(
            Counter(
                labels[qid]["type"] + "/" + ("valid" if labels[qid]["has_valid_label"] else "unlabeled") for qid in ids
            )
        )
        for name, ids in split["partitions"].items()
    }
