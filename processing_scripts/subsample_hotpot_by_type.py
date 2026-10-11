"""Sample processed HotpotQA dev data by question type."""

import argparse
import json
import os
import random
from collections import Counter


REQUIRED_TYPES = ("bridge", "comparison")


def convert_raw_instance(raw_instance):
    """Convert one official HotpotQA distractor record to the project schema."""
    question_id = raw_instance.get("id", raw_instance.get("_id"))
    if not question_id:
        raise ValueError("HotpotQA record is missing id/_id.")

    supporting_facts = raw_instance["supporting_facts"]
    if isinstance(supporting_facts, dict):
        supporting_titles = set(supporting_facts["title"])
    else:
        supporting_titles = {fact[0] for fact in supporting_facts}

    contexts = []
    raw_context = raw_instance["context"]
    if isinstance(raw_context, dict):
        context_pairs = zip(raw_context["title"], raw_context["sentences"])
    else:
        context_pairs = raw_context
    for index, (title, sentences) in enumerate(context_pairs):
        contexts.append(
            {
                "idx": index,
                "title": title.strip(),
                "paragraph_text": "".join(sentences).strip(),
                "is_supporting": title in supporting_titles,
            }
        )

    return {
        "dataset": "hotpotqa",
        "question_id": question_id,
        "question_text": raw_instance["question"],
        "level": raw_instance["level"],
        "type": raw_instance["type"],
        "answers_objects": [
            {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }
        ],
        "contexts": contexts,
    }


def stratified_sample(instances, per_type, seed):
    """Return a deterministic, shuffled equal-sized bridge/comparison sample."""
    grouped = {question_type: [] for question_type in REQUIRED_TYPES}
    for instance in instances:
        question_type = instance.get("type")
        if question_type in grouped:
            grouped[question_type].append(instance)

    missing = [name for name, rows in grouped.items() if len(rows) < per_type]
    if missing:
        raise ValueError("Insufficient rows for: " + ", ".join(missing))

    rng = random.Random(seed)
    sampled = []
    for question_type in REQUIRED_TYPES:
        sampled.extend(rng.sample(grouped[question_type], per_type))
    rng.shuffle(sampled)
    return sampled


def write_jsonl(instances, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as output_file:
        for instance in instances:
            output_file.write(json.dumps(instance, ensure_ascii=False) + "\n")


def read_jsonl(filepath):
    with open(filepath, encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_manifest(instances, filepath, seed, per_type):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    manifest = {
        "seed": seed,
        "per_type": per_type,
        "total": len(instances),
        "question_ids": [instance["question_id"] for instance in instances],
    }
    with open(filepath, "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def validate_instances(instances, per_type=None):
    if len({instance["question_id"] for instance in instances}) != len(instances):
        raise ValueError("question_id values are not unique.")
    for instance in instances:
        if instance.get("type") not in REQUIRED_TYPES:
            raise ValueError("Unknown or missing question type.")
        if not instance.get("contexts") or not instance.get("answers_objects"):
            raise ValueError("Missing contexts or answers_objects.")
    if per_type is not None:
        counts = Counter(instance["type"] for instance in instances)
        if any(counts[question_type] != per_type for question_type in REQUIRED_TYPES):
            raise ValueError("Sample does not have the requested per-type counts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed_data/hotpotqa/dev.jsonl")
    parser.add_argument("--sample-output", default="processed_data/hotpotqa/dev_1000_subsampled.jsonl")
    parser.add_argument("--ids-output", default="processed_data/hotpotqa/dev_1000_sample_manifest.json")
    parser.add_argument("--per-type", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13370)
    args = parser.parse_args()

    processed_instances = read_jsonl(args.input)
    validate_instances(processed_instances)
    sampled_instances = stratified_sample(processed_instances, args.per_type, args.seed)
    validate_instances(sampled_instances, per_type=args.per_type)

    write_jsonl(sampled_instances, args.sample_output)
    write_manifest(sampled_instances, args.ids_output, args.seed, args.per_type)
    print("processed_count={}".format(len(processed_instances)))
    print("sampled_type_counts={}".format(dict(Counter(row["type"] for row in sampled_instances))))
    print("seed={}".format(args.seed))


if __name__ == "__main__":
    main()
