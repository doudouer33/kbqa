#!/usr/bin/env python3
"""Independent, offline Soft Oracle builder. Core uses only the standard library."""

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VERSION = "soft-oracle-1.0.0"
BOUNDARY_ATOL = 1e-12
SCHEMES = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}
COST_METRICS = ("context_tokens", "prompt_tokens", "completion_tokens", "total_tokens")
EVIDENCE_METRICS = tuple(u + "_recall_" + b for u in ("sentence", "paragraph") for b in ("lower", "upper"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, object_pairs_hook=unique_object)


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            require(bool(line.strip()), "Blank JSONL record {}:{}".format(path, line_no))
            rows.append(parse_json(line))
        return rows
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
            "Bad/duplicate CSV header: " + str(path),
        )
        rows = list(reader)
        require(all(None not in r and None not in r.values() for r in rows), "Malformed CSV row: " + str(path))
        return rows


def index_rows(rows, fields, source):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        require(all(isinstance(v, str) and v.strip() for v in key), "Missing key in " + source)
        require(key not in result, "Duplicate key {} in {}".format(key, source))
        result[key] = row
    return result


def number(value, name, low=None, high=None, integer=False):
    require(not isinstance(value, bool), "Boolean instead of number: " + name)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid number: " + name)
    require(math.isfinite(result), "Non-finite number: " + name)
    require(low is None or result >= low, "Below range: " + name)
    require(high is None or result <= high, "Above range: " + name)
    if integer:
        require(result.is_integer(), "Non-integer: " + name)
        return int(result)
    return result


def close(a, b, name):
    require(math.isclose(a, b, rel_tol=0, abs_tol=BOUNDARY_ATOL), "Mismatch: " + name)


@dataclass(frozen=True)
class Parameters:
    epsilon: float
    tau: float
    lambda_: float
    temperature: float
    evidence_metric: str
    cost_metric: str
    scheme: str

    def __post_init__(self):
        for field in ("epsilon", "tau", "lambda_", "temperature"):
            require(
                isinstance(getattr(self, field), (int, float)) and not isinstance(getattr(self, field), bool),
                "Parameter must be a JSON number: " + field,
            )
        number(self.epsilon, "epsilon", 0, 1)
        number(self.tau, "tau", 0, 1)
        number(self.lambda_, "lambda", 0)
        number(self.temperature, "temperature", 0)
        require(self.temperature > 0, "temperature must be positive")
        require(self.scheme in SCHEMES, "Unknown scheme")
        require(self.evidence_metric in EVIDENCE_METRICS, "Unsupported evidence_metric")
        require(self.cost_metric in COST_METRICS, "Unsupported cost_metric")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["lambda_"] = data.pop("lambda")
        return cls(**data)

    def to_dict(self):
        data = asdict(self)
        data["lambda"] = data.pop("lambda_")
        return data


@dataclass(frozen=True)
class Question:
    question_id: str
    type: str
    question_text: str
    gold_answer: str
    ks: tuple
    f1: tuple
    em: tuple
    tokens: dict
    evidence: tuple
    predictions: tuple
    original_hard_oracle_k: int


def check_evidence(raw, name):
    """Check existing counts/bounds; do not perform evidence matching."""
    result = dict(raw)
    for unit in ("sentence", "paragraph"):
        for suffix in ("hit", "unknown", "total"):
            field = unit + "_" + suffix
            result[field] = number(raw[field], name + "/" + field, 0, integer=True)
        hit, unknown, total = (result[unit + "_" + f] for f in ("hit", "unknown", "total"))
        require(total > 0 and hit + unknown <= total, "Bad evidence counts: " + name)
        for bound, expected in (("lower", hit / total), ("upper", (hit + unknown) / total)):
            field = unit + "_recall_" + bound
            result[field] = number(raw[field], name + "/" + field, 0, 1)
            close(result[field], expected, name + "/" + field)
        expected_status = "complete" if hit == total else "unknown" if unknown else "partial" if hit else "zero"
        require(raw[unit + "_status"] == expected_status, "Inconsistent evidence status: " + name)
    for field in ("unknown_gold_facts", "unknown_gold_titles"):
        result[field] = parse_json(raw[field])
    require(len(result["unknown_gold_facts"]) == result["sentence_unknown"], "Unknown fact count mismatch: " + name)
    require(len(result["unknown_gold_titles"]) == result["paragraph_unknown"], "Unknown title count mismatch: " + name)
    return result


def load_data(paths, ks, expected_type_counts):
    """Validate all four sources before exposing one joined representation."""
    ks = tuple(ks)
    require(ks and ks == tuple(range(1, len(ks) + 1)), "k_values must be consecutive from 1")
    for path in paths.values():
        require(Path(path).is_file(), "Missing offline input: " + str(path))
    sources = {name: read_rows(path) for name, path in paths.items()}
    curves = index_rows(sources["curves"], ("question_id",), "curves")
    samples = index_rows(sources["dataset"], ("question_id",), "dataset")
    questions = index_rows(sources["questions"], ("question_id",), "per_question")
    evidence = {}
    for row in sources["evidence"]:
        require(row["qid"] == row["question_id"], "Evidence qid aliases differ")
        k = number(row["k"], "evidence k", 1, integer=True)
        key = (row["question_id"], k)
        require(key not in evidence, "Duplicate (qid,k): " + str(key))
        evidence[key] = row
    require(set(curves) == set(samples) == set(questions), "Question ID sets differ across curves/dataset/per_question")
    expected_pairs = {(qid[0], k) for qid in curves for k in ks}
    require(
        set(evidence) == expected_pairs,
        "Evidence (qid,k) alignment failure: missing={}, extra={}".format(
            sorted(expected_pairs - set(evidence))[:5], sorted(set(evidence) - expected_pairs)[:5]
        ),
    )
    actual_types = Counter(r["type"] for r in samples.values())
    require(actual_types == expected_type_counts, "Unexpected type counts: " + str(actual_types))
    records = []
    for key in sorted(curves):
        curve, sample, diagnostic = curves[key], samples[key], questions[key]
        (qid,) = key
        name = qid
        require(diagnostic["qid"] == qid, "per_question qid alias mismatch: " + qid)
        require(curve["type"] == sample["type"] == diagnostic["type"], "Type mismatch: " + qid)
        require(isinstance(sample["question_text"], str) and sample["question_text"].strip(), "Missing question text")
        require(sample["question_text"].strip() == diagnostic["question"].strip(), "Question text mismatch: " + qid)
        for field in ("scores", "context_tokens", "api_token_usage"):
            require(set(curve[field]) == {str(k) for k in ks}, "Incomplete {} curve: {}".format(field, qid))
        f1 = tuple(number(curve["scores"][str(k)]["f1"], name + "/F1", 0, 1) for k in ks)
        em = tuple(number(curve["scores"][str(k)]["em"], name + "/EM", 0, 1, True) for k in ks)
        require(all(e != 1 or f == 1 for f, e in zip(f1, em)), "EM=1 but F1!=1: " + qid)
        best = max(f1)
        ties = [k for k, score in zip(ks, f1) if score == best]
        require(curve["best_k"] == ties[0] and curve["tied_best_ks"] == ties, "Original Hard Oracle mismatch: " + qid)
        require(
            number(diagnostic["oracle_k"], name + "/oracle", integer=True) == ties[0],
            "Diagnostic oracle mismatch: " + qid,
        )
        require(parse_json(diagnostic["tied_best_ks"]) == ties, "Diagnostic ties mismatch: " + qid)
        for field, values in (("f1_curve", f1), ("em_curve", em)):
            existing = parse_json(diagnostic[field])
            require(len(existing) == len(ks), "Diagnostic curve length mismatch: " + qid)
            for a, b in zip(existing, values):
                close(number(a, field, 0, 1), b, name + "/" + field)
        for field, expected in (("k1_f1", f1[0]), ("k1_em", em[0]), ("max_f1", best), ("max_em", max(em))):
            close(number(diagnostic[field], field, 0, 1), expected, name + "/" + field)
        require(diagnostic["all_zero_f1"] == str(best == 0), "Diagnostic zero flag mismatch: " + qid)
        tokens = {metric: [] for metric in COST_METRICS}
        evs, predictions = [], []
        for i, k in enumerate(ks):
            name = "{}/{}".format(qid, k)
            raw_ev = evidence[(qid, k)]
            require(raw_ev["type"] == sample["type"], "Evidence type mismatch: " + name)
            require(number(raw_ev["oracle_k"], name, integer=True) == ties[0], "Evidence oracle mismatch: " + name)
            close(number(raw_ev["f1"], name + "/F1", 0, 1), f1[i], name + "/F1")
            close(number(raw_ev["em"], name + "/EM", 0, 1, True), em[i], name + "/EM")
            evs.append(check_evidence(raw_ev, name))
            usage = curve["api_token_usage"][str(k)]
            require(usage["qid"] == qid and usage["k"] == k, "Token usage (qid,k) mismatch: " + name)
            tokens["context_tokens"].append(
                number(curve["context_tokens"][str(k)], name + "/context_tokens", 1, integer=True)
            )
            for metric in COST_METRICS[1:]:
                tokens[metric].append(number(usage[metric], name + "/" + metric, 0, integer=True))
            require(
                tokens["total_tokens"][-1] == tokens["prompt_tokens"][-1] + tokens["completion_tokens"][-1],
                "Token total mismatch: " + name,
            )
            require(tokens["prompt_tokens"][-1] >= tokens["context_tokens"][-1], "Context exceeds prompt: " + name)
            prediction = curve["scores"][str(k)]["prediction"]
            require(isinstance(prediction, str) and usage["prediction"] == prediction, "Prediction mismatch: " + name)
            predictions.append(prediction)
        for unit in ("sentence", "paragraph"):
            require(len({e[unit + "_total"] for e in evs}) == 1, "Evidence total varies by k: " + qid)
        records.append(
            Question(
                qid,
                sample["type"],
                sample["question_text"],
                diagnostic["gold_answer"],
                ks,
                f1,
                em,
                {m: tuple(v) for m, v in tokens.items()},
                tuple(evs),
                tuple(predictions),
                ties[0],
            )
        )
    validation = {
        "passed": True,
        "query_count": len(records),
        "pair_count": len(evidence),
        "type_counts": dict(actual_types),
        "k_values": list(ks),
        "duplicate_keys": 0,
        "missing_pairs": 0,
        "f1_em_mismatches": 0,
        "token_anomalies": 0,
        "all_zero_f1_count": sum(max(q.f1) == 0 for q in records),
        "unknown_sentence_pairs": sum(e["sentence_unknown"] > 0 for q in records for e in q.evidence),
        "unknown_sentence_queries": sum(any(e["sentence_unknown"] for e in q.evidence) for q in records),
        "context_tokens_min": min(min(q.tokens["context_tokens"]) for q in records),
        "context_token_decrease_pairs": sum(
            b < a for q in records for a, b in zip(q.tokens["context_tokens"], q.tokens["context_tokens"][1:])
        ),
        "checks": [
            "all four qid sets; full unique (qid,k) join",
            "type and text alignment",
            "F1/EM across three diagnostic views",
            "Hard Oracle/ties from all k",
            "evidence count/status/bound consistency; unknown retained",
            "token usage qid/k; integer nonnegative API tokens; positive context; total=prompt+completion",
        ],
    }
    return records, validation


def normalize_cost(tokens):
    values = tuple(number(v, "cost tokens", 0) for v in tokens)
    require(values and max(values) > 0, "Cost normalization requires a positive all-k maximum")
    maximum = max(values)
    return tuple(v / maximum for v in values)


def stable_softmax(utilities, mask, temperature):
    require(len(utilities) == len(mask) and any(mask), "Softmax needs a nonempty aligned candidate set")
    number(temperature, "temperature", 0)
    require(temperature > 0, "temperature must be positive")
    for u in utilities:
        number(u, "utility")
    maximum = max(u for u, valid in zip(utilities, mask) if valid)
    weights = [math.exp((u - maximum) / temperature) if valid else 0.0 for u, valid in zip(utilities, mask)]
    total = math.fsum(weights)
    return tuple(w / total for w in weights)


def build_label(question, params):
    f1, ks = question.f1, question.ks
    best = max(f1)  # Always all k, before any evidence restriction.
    ev = tuple(row[params.evidence_metric] for row in question.evidence)
    costs = normalize_cost(question.tokens[params.cost_metric])  # Always all k.
    f1_pass = tuple(v + BOUNDARY_ATOL >= best - params.epsilon for v in f1)
    evidence_pass = tuple(v + BOUNDARY_ATOL >= params.tau for v in ev)
    use_f1, use_ev = SCHEMES[params.scheme]
    mask = tuple((not use_f1 or a) and (not use_ev or b) for a, b in zip(f1_pass, evidence_pass))
    utility = tuple(f - params.lambda_ * c for f, c in zip(f1, costs))
    count = sum(mask)
    status = "all_zero_f1" if best == 0 else "empty_candidates" if count == 0 else "valid"
    probabilities, argmax_k, entropy, normalized_candidate = None, None, None, None
    if status == "valid":
        probabilities = stable_softmax(utility, mask, params.temperature)
        # Utility ordering avoids temperature/underflow-induced argmax changes. Ties choose smallest k.
        index = max((i for i, yes in enumerate(mask) if yes), key=lambda i: utility[i])
        argmax_k = ks[index]
        entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
        normalized_candidate = entropy / math.log(count) if count > 1 else None
    return {
        "question_id": question.question_id,
        "type": question.type,
        "parameters": params.to_dict(),
        "max_f1": best,
        "f1_pass": f1_pass,
        "evidence_pass": evidence_pass,
        "candidate_mask": mask,
        "valid_candidates": tuple(k for k, yes in zip(ks, mask) if yes),
        "valid_candidate_count": count,
        "normalized_cost_curve": costs,
        "utility_curve": utility,
        "soft_label": probabilities,
        "label_status": status,
        "argmax_soft_label_k": argmax_k,
        "original_hard_oracle_k": question.original_hard_oracle_k,
        "entropy": entropy,
        "entropy_normalized_all_k": entropy / math.log(len(ks)) if entropy is not None and len(ks) > 1 else None,
        "entropy_normalized_candidates": normalized_candidate,
        "has_unknown_sentence_evidence": any(r["sentence_unknown"] for r in question.evidence),
    }


def label_record(question, label):
    return {
        **label,
        "question_text": question.question_text,
        "gold_answer": question.gold_answer,
        "k_values": question.ks,
        "f1_curve": question.f1,
        "em_curve": question.em,
        "evidence_recall_curve": tuple(r[label["parameters"]["evidence_metric"]] for r in question.evidence),
        "token_cost_curve": question.tokens[label["parameters"]["cost_metric"]],
        "token_curves": question.tokens,
        "evidence_diagnostic": [
            {
                key: row[key]
                for key in (
                    "sentence_hit",
                    "sentence_total",
                    "sentence_unknown",
                    "sentence_status",
                    "sentence_recall_lower",
                    "sentence_recall_upper",
                    "paragraph_hit",
                    "paragraph_total",
                    "paragraph_unknown",
                    "paragraph_status",
                    "paragraph_recall_lower",
                    "paragraph_recall_upper",
                    "unknown_gold_facts",
                    "unknown_gold_titles",
                )
            }
            for row in question.evidence
        ],
    }


def write_csv(path, rows):
    require(bool(rows), "Cannot write an empty table: " + str(path))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list, tuple)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def save_labels(output, questions, labels):
    with (output / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for q, label in zip(questions, labels):
            handle.write(json.dumps(label_record(q, label), ensure_ascii=False, allow_nan=False) + "\n")
    rows = []
    for q, label in zip(questions, labels):
        for i, k in enumerate(q.ks):
            rows.append(
                {
                    "question_id": q.question_id,
                    "type": q.type,
                    "k": k,
                    "f1": q.f1[i],
                    "em": q.em[i],
                    "evidence_recall": q.evidence[i][label["parameters"]["evidence_metric"]],
                    **{m: q.tokens[m][i] for m in COST_METRICS},
                    "normalized_cost": label["normalized_cost_curve"][i],
                    "f1_pass": label["f1_pass"][i],
                    "evidence_pass": label["evidence_pass"][i],
                    "is_candidate": label["candidate_mask"][i],
                    "utility": label["utility_curve"][i],
                    "soft_label_probability": label["soft_label"][i] if label["soft_label"] is not None else None,
                    "label_status": label["label_status"],
                    "parameters": label["parameters"],
                    **{
                        key: q.evidence[i][key]
                        for key in (
                            "sentence_hit",
                            "sentence_total",
                            "sentence_unknown",
                            "sentence_status",
                            "sentence_recall_lower",
                            "sentence_recall_upper",
                            "paragraph_recall_lower",
                            "paragraph_recall_upper",
                            "unknown_gold_facts",
                        )
                    },
                }
            )
    write_csv(output / "per_k_details.csv", rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    config = read_json(path)
    reference = Parameters.from_dict(config["reference"])
    require(
        set(config["inputs"]) == {"curves", "evidence", "questions", "dataset"}, "Exactly four input roles required"
    )
    require(config["expected_type_counts"] == {"bridge": 500, "comparison": 500}, "This study requires 500+500 queries")
    require(config["k_values"] == list(range(1, 16)), "This study requires k=1..15")
    for field in ("epsilon", "tau", "lambda", "temperature"):
        values = config["sweep"][field]
        require(values and len(set(values)) == len(values), "Empty/duplicate sweep values: " + field)
        for value in values:
            trial = dict(config["reference"], **{field: value})
            Parameters.from_dict(trial)
    return config, reference


def context(config_path):
    config_path = Path(config_path).resolve()
    config, reference = load_config(config_path)
    paths = {name: (ROOT / value).resolve() for name, value in config["inputs"].items()}
    # Read-only inputs are hashed before loading, then checked again at completion.
    hashes = {name: sha256(path) if path.is_file() else None for name, path in paths.items()}
    questions, validation = load_data(paths, config["k_values"], config["expected_type_counts"])
    metadata = {
        "label_construction_version": VERSION,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "timezone": "Asia/Shanghai",
        "analysis_kind": "Offline Oracle / Label Analysis",
        "command": [sys.executable] + sys.argv,
        "python_version": platform.python_version(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "configuration": config,
        "runtime_parameters": reference.to_dict(),
        "inputs": {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()},
        "cost_normalization": "per-query tokens(k) / max(tokens(j) for all 15 j); independent of candidate set",
        "evidence_definition": "Reused diagnostic: confirmed Gold sentence/paragraph hits in actual model-visible filtered, 350-word-rendered contexts; lower=hit/total, upper=(hit+unknown)/total. Unknown is uncertainty, not confirmed absence.",
        "numeric_boundary_atol": BOUNDARY_ATOL,
        "entropy_log_base": "e",
        "single_candidate_normalized_entropy": None,
        "all_zero_policy": "retain counts/masks/curves, soft_label=null; same in A/B/C/D",
        "empty_policy": "soft_label=null; no fallback",
        "argmax_tie_policy": "maximum utility then smallest k",
        "scope": "Only local saved statistics; no retrieval, LLM, inference, or training imports/calls",
        "evidence_definition_sources": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                ROOT / "analysis/hotpot_formal_1000_20261014/evidence_diagnostic/analyze_evidence.py",
                ROOT / "analysis/hotpot_formal_1000_20261014/evidence_diagnostic/summary.md",
            )
            if p.is_file()
        },
        "sample_count": len(questions),
        "k_values": config["k_values"],
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(HERE.rglob("*.py"))},
    }
    try:
        metadata["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
        metadata["git_status_before_run"] = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-uall"], cwd=str(ROOT)
        ).decode()
    except (OSError, subprocess.CalledProcessError):
        metadata["git_head"] = None
    return config, reference, questions, validation, metadata


def allocate_output(path):
    if path is None:
        parent = HERE / "outputs"
        parent.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            candidate = parent / "experiment_{:03d}".format(index)
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                index += 1
    result = Path(path).resolve()
    # mkdir(exist_ok=False) also refuses an existing empty directory or a symlink.
    result.mkdir(parents=True, exist_ok=False)
    return result


def assert_inputs_unchanged(metadata):
    for source in metadata["inputs"].values():
        require(sha256(source["path"]) == source["sha256"], "Input changed during analysis: " + source["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--output", help="New directory only; default allocates experiment_NNN")
    args = parser.parse_args()
    _, reference, questions, validation, metadata = context(args.config)
    labels = [build_label(q, reference) for q in questions]
    output = allocate_output(args.output)
    save_labels(output, questions, labels)
    assert_inputs_unchanged(metadata)
    write_json(output / "validation.json", validation)
    write_json(output / "metadata.json", metadata)
    print(output)


if __name__ == "__main__":
    main()
