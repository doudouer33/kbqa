#!/usr/bin/env python
"""Train and evaluate the stage-2 bucket classifier for retrieval intensity."""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from kbqa_classifier.train.train_stage1 import (
    WeightedLossTrainer,
    build_eval_artifact_paths,
    compute_classification_metrics,
    extract_optional_scalar,
    extract_question_text,
    infer_dataset_name,
    infer_model_tag,
    infer_split_name,
    load_config_label_mapping,
    load_json_if_exists,
    load_labeled_records,
    normalize_id2label_mapping,
    normalize_label2id_mapping,
    parse_raw_label,
    predict_logits,
    resolve_input_path,
    resolve_output_path,
    save_json,
    set_reproducible_seed,
    str2bool,
    write_jsonl,
)


DEFAULT_TRAIN_FILE = "kbqa_classifier/data/merged/dev_500_topk_train.json"
DEFAULT_EVAL_OUTPUT_DIR = "kbqa_classifier/output/classifier_eval"


class Stage2ClassificationDataset(Dataset):
    """Tokenized dataset for stage-2 bucket classification."""

    def __init__(self, examples: Sequence[Dict[str, Any]], tokenizer, max_length: int):
        self.examples = list(examples)
        self.encodings = tokenizer(
            [example["question"] for example in self.examples],
            truncation=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.examples[index]["bucket_label"], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the stage-2 retrieval bucket classifier.")
    parser.add_argument(
        "--train_file",
        type=str,
        default=DEFAULT_TRAIN_FILE,
        help="Path to the JSON or JSONL training file.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to the JSON or JSONL evaluation file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="BERT checkpoint name, saved model directory, or checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/stage2",
        help="Directory used to save the fine-tuned HuggingFace model.",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default=DEFAULT_EVAL_OUTPUT_DIR,
        help="Root directory used to save classifier evaluation artifacts.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional dataset tag used in evaluation artifact names.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help="Optional split tag used in evaluation artifact names.",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Optional model tag used in evaluation artifact names.",
    )
    parser.add_argument("--max_length", type=int, default=64, help="Maximum tokenized question length.")
    parser.add_argument(
        "--per_device_train_batch_size",
        "--train_batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=16,
        help="Per-device batch size used for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        "--eval_batch_size",
        dest="per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size used for classifier evaluation.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=float, default=5.0, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging interval in steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Checkpoint save strategy passed to TrainingArguments.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Used when --save_strategy=steps.",
    )
    parser.add_argument(
        "--do_train",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to run training. Supports --do_train, --do_train True, or --do_train False.",
    )
    parser.add_argument(
        "--do_eval",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to run classifier-level evaluation on --eval_file.",
    )
    return parser.parse_args()


def topk_label_to_bucket(topk_label: int) -> int:
    """Map the original positive top-k label into one of the 4 stage-2 buckets."""

    if 1 <= topk_label <= 2:
        return 0
    if 3 <= topk_label <= 5:
        return 1
    if 6 <= topk_label <= 9:
        return 2
    if 10 <= topk_label <= 15:
        return 3
    raise ValueError(
        f"Stage-2 bucket mapping only supports positive top-k labels in [1, 15], but got {topk_label}."
    )


def load_stage2_training_examples(
    train_file: Path,
) -> Tuple[List[Dict[str, Any]], int, Counter, Counter, str, str]:
    records, question_field, label_field, _, _ = load_labeled_records(train_file)

    original_label_distribution: Counter = Counter()
    bucket_distribution: Counter = Counter()
    examples: List[Dict[str, Any]] = []

    for index, record in enumerate(records):
        question = extract_question_text(record, question_field, index)
        if label_field not in record:
            raise KeyError(f"Missing label field {label_field!r} at record index {index}.")

        topk_label = parse_raw_label(record[label_field])
        original_label_distribution[topk_label] += 1

        # Stage 2 only trains on samples that already require retrieval,
        # so we drop label == 0 before mapping to bucket labels.
        if topk_label == 0:
            continue

        bucket_label = topk_label_to_bucket(topk_label)
        bucket_distribution[bucket_label] += 1
        examples.append(
            {
                "question": question,
                "topk_label": topk_label,
                "bucket_label": bucket_label,
            }
        )

    if not examples:
        raise ValueError(
            f"No positive-retrieval training examples found in {train_file}. Stage 2 requires label > 0 samples."
        )

    return examples, len(records), original_label_distribution, bucket_distribution, question_field, label_field


def load_stage2_eval_examples(
    eval_file: Path,
) -> Tuple[List[Dict[str, Any]], int, int, Counter, Counter, str, str, Optional[str], Optional[str]]:
    records, question_field, label_field, id_field, dataset_field = load_labeled_records(eval_file)

    examples: List[Dict[str, Any]] = []
    raw_label_distribution: Counter = Counter()
    bucket_distribution: Counter = Counter()
    skipped_sample_count = 0

    for index, record in enumerate(records):
        question = extract_question_text(record, question_field, index)
        if label_field not in record:
            raise KeyError(f"Missing label field {label_field!r} at record index {index}.")

        raw_topk_label = parse_raw_label(record[label_field])
        raw_label_distribution[raw_topk_label] += 1

        if raw_topk_label == 0:
            skipped_sample_count += 1
            continue

        bucket_label = topk_label_to_bucket(raw_topk_label)
        bucket_distribution[bucket_label] += 1

        sample_id = extract_optional_scalar(record, id_field)
        dataset_name = extract_optional_scalar(record, dataset_field)

        examples.append(
            {
                "id": sample_id if sample_id is not None else f"sample_{index}",
                "question": question,
                "dataset_name": dataset_name,
                "raw_topk_label": raw_topk_label,
                "gold_label": bucket_label,
            }
        )

    if not examples:
        raise ValueError(
            f"No stage-2 evaluable samples remain after filtering out original top-k label == 0 from {eval_file}."
        )

    return (
        examples,
        len(records),
        skipped_sample_count,
        raw_label_distribution,
        bucket_distribution,
        question_field,
        label_field,
        id_field,
        dataset_field,
    )


def compute_bucket_class_weights(labels: Sequence[int], num_labels: int = 4) -> torch.Tensor:
    label_distribution = Counter(labels)
    class_counts = [label_distribution.get(label_id, 0) for label_id in range(num_labels)]
    present_class_count = sum(class_count > 0 for class_count in class_counts)
    if present_class_count == 0:
        raise ValueError("No bucket labels were found for stage-2 training.")

    total_count = float(sum(class_counts))
    # Inverse-frequency class weights reduce the dominance of frequent bucket labels.
    weights = [
        total_count / (present_class_count * class_count) if class_count > 0 else 0.0
        for class_count in class_counts
    ]
    return torch.tensor(weights, dtype=torch.float)


def load_stage2_label_metadata(model_name_or_path: str) -> Tuple[Dict[int, str], Dict[str, int], Dict[str, Any]]:
    from kbqa_classifier.model.stage2_bert_classifier import (
        STAGE2_BUCKET_DEFINITIONS,
        STAGE2_ID2LABEL,
        STAGE2_LABEL2ID,
    )

    resolved_path = resolve_input_path(model_name_or_path)
    mapping_payload: Optional[Dict[str, Any]] = None
    if resolved_path.exists() and resolved_path.is_dir():
        mapping_payload = load_json_if_exists(resolved_path / "stage2_label_mapping.json")

    if mapping_payload is not None:
        id2label = normalize_id2label_mapping(mapping_payload.get("id2label"), STAGE2_ID2LABEL)
        label2id = normalize_label2id_mapping(mapping_payload.get("label2id"), STAGE2_LABEL2ID)
        bucket_definitions = mapping_payload.get("bucket_definitions")
        if not isinstance(bucket_definitions, dict):
            bucket_definitions = {str(key): value for key, value in STAGE2_BUCKET_DEFINITIONS.items()}
        return id2label, label2id, bucket_definitions

    id2label, label2id = load_config_label_mapping(model_name_or_path, STAGE2_ID2LABEL, STAGE2_LABEL2ID)
    bucket_definitions = {str(key): value for key, value in STAGE2_BUCKET_DEFINITIONS.items()}
    return id2label, label2id, bucket_definitions


def train_stage2_classifier(args: argparse.Namespace) -> Path:
    train_file = resolve_input_path(args.train_file)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples, original_sample_count, original_label_distribution, bucket_distribution, question_field, label_field = (
        load_stage2_training_examples(train_file)
    )
    bucket_labels = [example["bucket_label"] for example in examples]
    class_weights = compute_bucket_class_weights(bucket_labels, num_labels=4)

    from transformers import DataCollatorWithPadding, TrainingArguments

    from kbqa_classifier.model.stage2_bert_classifier import (
        STAGE2_BUCKET_DEFINITIONS,
        STAGE2_ID2LABEL,
        STAGE2_LABEL2ID,
        build_stage2_model,
        load_stage2_tokenizer,
    )

    tokenizer = load_stage2_tokenizer(args.model_name_or_path)
    model = build_stage2_model(args.model_name_or_path)
    train_dataset = Stage2ClassificationDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=False,
        evaluation_strategy="no",
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        remove_unused_columns=True,
    )

    print(f"Original training samples: {original_sample_count}")
    print(f"Filtered stage-2 training samples (label > 0): {len(examples)}")
    print(f"Bucket distribution: {dict(sorted(bucket_distribution.items()))}")
    print(f"Training output directory: {output_dir}")

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        class_weights=class_weights,
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()

    metrics = dict(train_result.metrics)
    metrics["original_samples"] = original_sample_count
    metrics["filtered_stage2_samples"] = len(examples)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    save_json(
        output_dir / "stage2_label_mapping.json",
        {
            "num_labels": 4,
            "id2label": {str(key): value for key, value in STAGE2_ID2LABEL.items()},
            "label2id": dict(STAGE2_LABEL2ID),
            "bucket_definitions": {
                str(bucket_id): bucket_info for bucket_id, bucket_info in STAGE2_BUCKET_DEFINITIONS.items()
            },
            "topk_to_bucket_rule": {
                "bucket_0": "top-k 1~2",
                "bucket_1": "top-k 3~5",
                "bucket_2": "top-k 6~9",
                "bucket_3": "top-k 10~15",
            },
        },
    )
    save_json(
        output_dir / "stage2_training_config.json",
        {
            "train_file": str(train_file),
            "eval_file": args.eval_file,
            "model_name_or_path": args.model_name_or_path,
            "question_field": question_field,
            "raw_label_field": label_field,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "logging_steps": args.logging_steps,
            "seed": args.seed,
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "do_train": args.do_train,
            "do_eval": args.do_eval,
            "dataset_name": args.dataset_name,
            "split_name": args.split_name,
            "model_tag": args.model_tag,
            "original_sample_count": original_sample_count,
            "filtered_stage2_sample_count": len(examples),
            "original_label_distribution": {
                str(key): value for key, value in sorted(original_label_distribution.items())
            },
            "bucket_distribution": {str(key): value for key, value in sorted(bucket_distribution.items())},
            "class_weights": class_weights.tolist(),
        },
    )
    save_json(output_dir / "training_args.json", training_args.to_dict())

    print("Stage-2 training finished.")
    print(f"Saved stage-2 model to: {output_dir}")
    return output_dir


def evaluate_stage2_classifier(args: argparse.Namespace, model_source: str) -> Tuple[Path, Path, Dict[str, Any]]:
    eval_file = resolve_input_path(args.eval_file)
    (
        examples,
        original_sample_count,
        skipped_sample_count,
        raw_label_distribution,
        bucket_distribution,
        question_field,
        label_field,
        id_field,
        dataset_field,
    ) = load_stage2_eval_examples(eval_file)

    from kbqa_classifier.model.stage2_bert_classifier import build_stage2_model, load_stage2_tokenizer

    id2label, label2id, bucket_definitions = load_stage2_label_metadata(model_source)
    tokenizer = load_stage2_tokenizer(model_source)
    model = build_stage2_model(model_source)

    logits, probabilities, pred_labels = predict_logits(
        texts=[example["question"] for example in examples],
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.per_device_eval_batch_size,
    )

    gold_labels = [int(example["gold_label"]) for example in examples]
    metrics = compute_classification_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        label_ids=sorted(id2label.keys()),
        id2label=id2label,
    )

    dataset_name = infer_dataset_name(args.dataset_name, examples, eval_file)
    split_name = infer_split_name(args.split_name, eval_file)
    model_tag = infer_model_tag(args.model_tag, model_source)
    _, metrics_path, predictions_path = build_eval_artifact_paths(
        stage_name="stage2",
        eval_output_dir=args.eval_output_dir,
        dataset_name=dataset_name,
        split_name=split_name,
        model_tag=model_tag,
    )

    prediction_rows: List[Dict[str, Any]] = []
    for example, logit_row, probability_row, pred_label in zip(examples, logits, probabilities, pred_labels):
        gold_label = int(example["gold_label"])
        pred_label = int(pred_label)
        resolved_dataset_name = example["dataset_name"] if example["dataset_name"] is not None else dataset_name

        prediction_rows.append(
            {
                "id": example["id"],
                "question": example.get("question"),
                "dataset_name": resolved_dataset_name,
                "raw_topk_label": int(example["raw_topk_label"]),
                "gold_label": gold_label,
                "gold_label_name": id2label.get(gold_label, str(gold_label)),
                "pred_label": pred_label,
                "pred_label_name": id2label.get(pred_label, str(pred_label)),
                "gold_bucket_definition": bucket_definitions.get(str(gold_label)),
                "pred_bucket_definition": bucket_definitions.get(str(pred_label)),
                "logits": [float(value) for value in logit_row.tolist()],
                "probabilities": [float(value) for value in probability_row.tolist()],
            }
        )

    metrics.update(
        {
            "stage": "stage2",
            "task": "classifier_eval",
            "dataset_name": dataset_name,
            "split_name": split_name,
            "model_tag": model_tag,
            "eval_file": str(eval_file),
            "model_name_or_path": str(model_source),
            "question_field": question_field,
            "raw_label_field": label_field,
            "id_field": id_field,
            "dataset_name_field": dataset_field,
            "original_sample_count": original_sample_count,
            "evaluated_sample_count": len(examples),
            "skipped_sample_count": skipped_sample_count,
            "raw_label_distribution": {str(key): value for key, value in sorted(raw_label_distribution.items())},
            "bucket_distribution": {str(key): value for key, value in sorted(bucket_distribution.items())},
            "label_mapping": {
                "id2label": {str(key): value for key, value in sorted(id2label.items())},
                "label2id": label2id,
            },
            "bucket_definitions": bucket_definitions,
            "prediction_file": str(predictions_path),
        }
    )

    save_json(metrics_path, metrics)
    write_jsonl(predictions_path, prediction_rows)

    print("Stage-2 classifier evaluation finished.")
    print(
        f"Original samples: {original_sample_count} | Evaluated samples (top-k > 0): {len(examples)} | "
        f"Skipped samples (top-k == 0): {skipped_sample_count}"
    )
    print(
        f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | "
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )
    print(f"Confusion matrix (gold rows, pred cols): {metrics['confusion_matrix']['matrix']}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")

    return metrics_path, predictions_path, metrics


def main() -> None:
    args = parse_args()

    if not args.do_train and not args.do_eval:
        raise ValueError("Nothing to do. At least one of --do_train or --do_eval must be True.")
    if args.do_eval and not args.eval_file:
        raise ValueError("--eval_file must be provided when --do_eval is True.")

    set_reproducible_seed(args.seed)

    eval_model_source = args.model_name_or_path

    if args.do_train:
        eval_model_source = str(train_stage2_classifier(args))
    else:
        print("Skipping stage-2 training because --do_train is False.")

    if args.do_eval:
        evaluate_stage2_classifier(args, eval_model_source)
    else:
        print("Skipping stage-2 evaluation because --do_eval is False.")


if __name__ == "__main__":
    main()
