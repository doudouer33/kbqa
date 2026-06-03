#!/usr/bin/env python
"""Train and validate the stage-1 classifier: top-k label 0 vs non-zero."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kbqa_classifier_new.model.stage1_classifier import (
    STAGE1_ID2LABEL,
    STAGE1_LABEL2ID,
    load_stage1_model,
    load_stage1_tokenizer,
)
from kbqa_classifier_new.train.common import (
    DEFAULT_TRAIN_FILE,
    DEFAULT_VALID_FILE,
    QuestionClassificationDataset,
    WeightedLossTrainer,
    build_prediction_rows,
    compute_class_weights,
    compute_classification_metrics,
    ensure_output_dir,
    load_stage_examples,
    make_trainer_metrics_fn,
    resolve_path,
    save_json,
    set_seed,
    str2bool,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 training: label 0 vs label > 0.")
    parser.add_argument("--train_file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval_file", default=DEFAULT_VALID_FILE)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased")
    parser.add_argument("--output_dir", default="kbqa_classifier_new/outputs/stage1")
    parser.add_argument("--question_field", default=None)
    parser.add_argument("--label_field", default=None)
    parser.add_argument("--id_field", default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", "--train_batch_size", dest="train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", "--eval_batch_size", dest="eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", choices=("no", "steps", "epoch"), default="epoch")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_class_weights", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--do_train", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--do_eval", type=str2bool, nargs="?", const=True, default=False)
    return parser.parse_args()


def build_trainer(
    args: argparse.Namespace,
    output_dir: Path,
    train_dataset: Optional[QuestionClassificationDataset],
    eval_dataset: Optional[QuestionClassificationDataset],
    class_weights,
):
    from transformers import DataCollatorWithPadding, TrainingArguments

    tokenizer = load_stage1_tokenizer(args.model_name_or_path)
    model = load_stage1_model(args.model_name_or_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluation_strategy="no",
        save_strategy=args.save_strategy if args.do_train else "no",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        data_seed=args.seed,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=True,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_trainer_metrics_fn(sorted(STAGE1_ID2LABEL), STAGE1_ID2LABEL),
        class_weights=class_weights,
    )
    return trainer, tokenizer, training_args


def train(args: argparse.Namespace, output_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    train_file = resolve_path(args.train_file)
    train_examples, train_metadata = load_stage_examples(
        train_file,
        stage="stage1",
        question_field=args.question_field,
        label_field=args.label_field,
        id_field=args.id_field,
    )

    tokenizer = load_stage1_tokenizer(args.model_name_or_path)
    train_dataset = QuestionClassificationDataset(train_examples, tokenizer, max_length=args.max_length)
    class_weights = (
        compute_class_weights([example["label"] for example in train_examples], num_labels=len(STAGE1_ID2LABEL))
        if args.use_class_weights
        else None
    )

    trainer, tokenizer, training_args = build_trainer(
        args,
        output_dir=output_dir,
        train_dataset=train_dataset,
        eval_dataset=None,
        class_weights=class_weights,
    )

    print(f"Stage 1 train examples: {len(train_examples)}")
    print(f"Stage 1 label distribution: {train_metadata['task_label_distribution']}")
    print(f"Training output directory: {output_dir}")

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    save_json(output_dir / "stage1_label_mapping.json", {
        "num_labels": len(STAGE1_ID2LABEL),
        "id2label": {str(key): value for key, value in STAGE1_ID2LABEL.items()},
        "label2id": STAGE1_LABEL2ID,
    })
    save_json(output_dir / "stage1_training_config.json", {
        "model_name_or_path": args.model_name_or_path,
        "max_length": args.max_length,
        "train_metadata": train_metadata,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "training_args": training_args.to_dict(),
    })

    print("Stage 1 training finished.")
    return trainer, train_metadata


def evaluate(args: argparse.Namespace, output_dir: Path, trainer=None) -> Dict[str, Any]:
    eval_file = resolve_path(args.eval_file)
    eval_examples, eval_metadata = load_stage_examples(
        eval_file,
        stage="stage1",
        question_field=args.question_field,
        label_field=args.label_field,
        id_field=args.id_field,
    )

    tokenizer = load_stage1_tokenizer(args.model_name_or_path)
    eval_dataset = QuestionClassificationDataset(eval_examples, tokenizer, max_length=args.max_length)

    if trainer is None:
        trainer, _, _ = build_trainer(
            args,
            output_dir=output_dir,
            train_dataset=None,
            eval_dataset=eval_dataset,
            class_weights=None,
        )
    else:
        trainer.eval_dataset = eval_dataset

    prediction_output = trainer.predict(eval_dataset)
    prediction_rows, gold_labels, pred_labels = build_prediction_rows(
        eval_examples,
        prediction_output.predictions,
        STAGE1_ID2LABEL,
    )
    metrics = compute_classification_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        label_ids=sorted(STAGE1_ID2LABEL),
        id2label=STAGE1_ID2LABEL,
    )
    metrics.update({
        "stage": "stage1",
        "task": "validation",
        "model_name_or_path": args.model_name_or_path,
        "eval_metadata": eval_metadata,
        "prediction_file": str(output_dir / "stage1_valid_predictions.jsonl"),
    })

    save_json(output_dir / "stage1_valid_metrics.json", metrics)
    write_jsonl(output_dir / "stage1_valid_predictions.jsonl", prediction_rows)
    print("Stage 1 validation finished.")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Validation artifacts saved to: {output_dir}")
    return metrics


def main() -> None:
    args = parse_args()
    if not args.do_train and not args.do_eval:
        raise ValueError("Nothing to do. Pass --do_train, --do_eval, or both.")

    set_seed(args.seed)
    output_dir = ensure_output_dir(args.output_dir)

    trainer = None
    if args.do_train:
        trainer, _ = train(args, output_dir)
        args.model_name_or_path = str(output_dir)

    if args.do_eval:
        evaluate(args, output_dir, trainer=trainer)


if __name__ == "__main__":
    main()
