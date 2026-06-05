#!/usr/bin/env python
"""Build top-k labels for 2wikimultihopqa, hotpotqa, and musique.

Use either --dev or --test. The script uses:
- predictions/<split>/ircot_qa_qwen_<dataset>____prompt_set_1___bm25_retrieval_count__<k>___distractor_count__1
  as k=1..8

The output structure mirrors kbqa_classifier/topk_data_builder.py:
- detailed samples contain best_k, best_em, best_f1, scores, and predictions
- simple samples contain id, question, and label
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

DATASET_NAMES: Tuple[str, ...] = ("2wikimultihopqa", "hotpotqa", "musique")
K_MIN = 1
K_MAX = 8

MODEL_NAME = "qwen"
QA_METHOD = "ircot"

PROCESSED_DATA_ROOT = PROJECT_ROOT / "processed_data"
OUTPUT_ROOT = PROJECT_ROOT / "kbqa_classifier_mutihop" / "data"


class SplitConfig(NamedTuple):
    cli_name: str
    predictions_dir_name: str
    set_name: str
    output_dir_name: str
    output_prefix: str


SPLIT_CONFIGS: Dict[str, SplitConfig] = {
    "dev": SplitConfig(
        cli_name="dev",
        predictions_dir_name="dev_3000",
        set_name="dev_3000_subsampled",
        output_dir_name="dev",
        output_prefix="dev_3000",
    ),
    "test": SplitConfig(
        cli_name="test",
        predictions_dir_name="test",
        set_name="test_subsampled",
        output_dir_name="test",
        output_prefix="test",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build top-k classifier data for 2wikimultihopqa, hotpotqa, and musique."
    )
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--dev",
        dest="split_name",
        action="store_const",
        const="dev",
        help="Build from predictions/dev_3000 and processed_data/*/dev_3000_subsampled.jsonl.",
    )
    split_group.add_argument(
        "--test",
        dest="split_name",
        action="store_const",
        const="test",
        help="Build from predictions/test and processed_data/*/test_subsampled.jsonl.",
    )
    return parser.parse_args()


def write_json(data: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_question_texts(processed_data_path: Path) -> Dict[str, str]:
    if not processed_data_path.is_file():
        raise FileNotFoundError(f"Processed data file not found: {processed_data_path}")

    id_to_question: Dict[str, str] = {}
    with processed_data_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                instance = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of processed data file: {processed_data_path}"
                ) from exc

            if "question_id" not in instance:
                raise KeyError(f"Missing 'question_id' on line {line_number} of {processed_data_path}")
            if "question_text" not in instance:
                raise KeyError(f"Missing 'question_text' on line {line_number} of {processed_data_path}")

            question_id = str(instance["question_id"])
            question_text = instance["question_text"]
            if question_text is None:
                raise ValueError(f"'question_text' is null for id '{question_id}' in {processed_data_path}")

            if question_id in id_to_question:
                raise ValueError(f"Duplicate question_id '{question_id}' found in {processed_data_path}")

            id_to_question[question_id] = str(question_text)

    return id_to_question


def get_predictions_root(split_config: SplitConfig) -> Path:
    return PROJECT_ROOT / "predictions" / split_config.predictions_dir_name


def get_evaluation_name(dataset_name: str, split_config: SplitConfig) -> str:
    return f"{dataset_name}_to_{dataset_name}__{split_config.set_name}"


def get_ircot_qa_pattern(dataset_name: str, split_config: SplitConfig) -> Path:
    return (
        get_predictions_root(split_config)
        / f"{QA_METHOD}_qa_{MODEL_NAME}_{dataset_name}____prompt_set_1___bm25_retrieval_count__{{k}}___distractor_count__1"
    )


def get_per_question_eval_path(experiment_dir: Path, dataset_name: str, split_config: SplitConfig) -> Path:
    return experiment_dir / f"per_question_eval__{get_evaluation_name(dataset_name, split_config)}.json"


def validate_split_inputs(split_config: SplitConfig) -> None:
    missing_paths: List[Path] = []
    predictions_root = get_predictions_root(split_config)

    if not predictions_root.is_dir():
        missing_paths.append(predictions_root)

    for dataset_name in DATASET_NAMES:
        processed_data_path = PROCESSED_DATA_ROOT / dataset_name / f"{split_config.set_name}.jsonl"
        if not processed_data_path.is_file():
            missing_paths.append(processed_data_path)

        ircot_qa_pattern = get_ircot_qa_pattern(dataset_name, split_config)
        for k in range(K_MIN, K_MAX + 1):
            experiment_dir = Path(str(ircot_qa_pattern).format(k=k))
            if not experiment_dir.is_dir():
                missing_paths.append(experiment_dir)
                continue

            per_question_eval_path = get_per_question_eval_path(
                experiment_dir,
                dataset_name,
                split_config,
            )
            if not per_question_eval_path.is_file():
                missing_paths.append(per_question_eval_path)

    if missing_paths:
        preview_count = 30
        preview = "\n".join(f"- {path}" for path in missing_paths[:preview_count])
        remaining = len(missing_paths) - preview_count
        if remaining > 0:
            preview = f"{preview}\n- ... and {remaining} more"
        raise FileNotFoundError(f"Missing inputs for --{split_config.cli_name}:\n{preview}")


def find_per_question_eval_file(
    experiment_dir: Path,
    set_name: str,
    evaluation_name: Optional[str] = None,
) -> Path:
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    candidates = sorted(experiment_dir.glob("per_question_eval__*.json"))
    if not candidates:
        raise FileNotFoundError(f"No per_question_eval__*.json file found in {experiment_dir}")

    if evaluation_name:
        target_name = f"per_question_eval__{evaluation_name}.json"
        matches = [candidate for candidate in candidates if candidate.name == target_name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"Expected {target_name} in {experiment_dir}, found {[candidate.name for candidate in candidates]}"
            )
        raise ValueError(f"Multiple files matched {target_name} in {experiment_dir}")

    if len(candidates) == 1:
        return candidates[0]

    set_name_matches = [candidate for candidate in candidates if candidate.stem.endswith(f"__{set_name}")]
    if len(set_name_matches) == 1:
        return set_name_matches[0]
    if len(set_name_matches) > 1:
        raise ValueError(
            f"Multiple per_question_eval files matched set_name='{set_name}' in {experiment_dir}: "
            f"{[candidate.name for candidate in set_name_matches]}"
        )

    raise ValueError(
        f"Multiple per_question_eval files found in {experiment_dir}: {[candidate.name for candidate in candidates]}. "
        "Pass evaluation_name to disambiguate."
    )


def load_per_question_eval(per_question_eval_path: Path) -> Dict[str, Dict[str, object]]:
    if not per_question_eval_path.is_file():
        raise FileNotFoundError(f"Per-question eval file not found: {per_question_eval_path}")

    with per_question_eval_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {per_question_eval_path}, got {type(data).__name__}")

    results_by_id: Dict[str, Dict[str, object]] = {}
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} in {per_question_eval_path} is not a JSON object")

        if "id" not in item:
            raise KeyError(f"Missing 'id' for item {index} in {per_question_eval_path}")
        if "em" not in item:
            raise KeyError(f"Missing 'em' for id '{item['id']}' in {per_question_eval_path}")
        if "f1" not in item:
            raise KeyError(f"Missing 'f1' for id '{item['id']}' in {per_question_eval_path}")

        question_id = str(item["id"])
        if question_id in results_by_id:
            raise ValueError(f"Duplicate id '{question_id}' found in {per_question_eval_path}")

        prediction = item.get("predicted_answer")
        if prediction is None:
            prediction = item.get("prediction")
        if prediction is None:
            raise KeyError(
                f"Neither 'predicted_answer' nor 'prediction' exists for id '{question_id}' in {per_question_eval_path}"
            )

        em = item["em"]
        if not isinstance(em, (int, float)) or em not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Expected EM to be 0/1 for id '{question_id}' in {per_question_eval_path}, got {em!r}")

        f1 = item["f1"]
        if not isinstance(f1, (int, float)):
            raise ValueError(f"Expected numeric F1 for id '{question_id}' in {per_question_eval_path}, got {f1!r}")

        results_by_id[question_id] = {
            "em": int(em),
            "f1": float(f1),
            "prediction": str(prediction),
        }

    return results_by_id


def resolve_k_experiment_dirs(
    ircot_qa_pattern: Path,
    k_min: int,
    k_max: int,
) -> Dict[int, Path]:
    if k_min < 1:
        raise ValueError(f"k_min must be >= 1, got {k_min}")
    if k_max < k_min:
        raise ValueError(f"k_max must be >= k_min, got k_min={k_min}, k_max={k_max}")

    pattern = str(ircot_qa_pattern)
    if "{k}" not in pattern:
        raise ValueError("ircot_qa_pattern must contain the '{k}' placeholder.")

    k_to_dir: Dict[int, Path] = {}
    for k in range(k_min, k_max + 1):
        experiment_dir = Path(pattern.format(k=k))
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"ircot_qa directory for k={k} not found: {experiment_dir}")
        k_to_dir[k] = experiment_dir

    return k_to_dir


def select_best_k(k_to_result: Dict[int, Dict[str, object]]) -> Tuple[int, Dict[str, object]]:
    return max(
        k_to_result.items(),
        key=lambda item: (item[1]["em"], item[1]["f1"], -item[0]),
    )


def build_topk_dataset(
    dataset_name: str,
    set_name: str,
    processed_data_path: Path,
    ircot_qa_pattern: Path,
    k_min: int,
    k_max: int,
    evaluation_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    logger = logger or logging.getLogger(__name__)

    question_texts = load_question_texts(processed_data_path)
    logger.info("Loaded %d questions from %s", len(question_texts), processed_data_path)

    k_to_experiment_dir = resolve_k_experiment_dirs(
        ircot_qa_pattern=ircot_qa_pattern,
        k_min=k_min,
        k_max=k_max,
    )
    logger.info("Using %d experiment directories for k values: %s", len(k_to_experiment_dir), sorted(k_to_experiment_dir))

    aggregated_results: Dict[str, Dict[int, Dict[str, object]]] = defaultdict(dict)
    coverage_by_k: Dict[int, int] = {}

    for k in sorted(k_to_experiment_dir):
        experiment_dir = k_to_experiment_dir[k]
        per_question_eval_file = find_per_question_eval_file(
            experiment_dir=experiment_dir,
            set_name=set_name,
            evaluation_name=evaluation_name,
        )
        results_by_id = load_per_question_eval(per_question_eval_file)
        coverage_by_k[k] = len(results_by_id)

        logger.info(
            "Loaded k=%d from %s using %s (%d questions)",
            k,
            experiment_dir,
            per_question_eval_file.name,
            len(results_by_id),
        )

        for question_id, result in results_by_id.items():
            aggregated_results[question_id][k] = result

    if not aggregated_results:
        raise ValueError("No per-question results were loaded. Nothing to write.")

    aggregated_question_ids = set(aggregated_results)
    missing_question_ids = sorted(aggregated_question_ids - set(question_texts))
    if missing_question_ids:
        preview = missing_question_ids[:10]
        raise ValueError(
            f"{len(missing_question_ids)} ids appear in per_question_eval files but not in {processed_data_path}. "
            f"Examples: {preview}"
        )

    missing_results_examples: List[str] = []
    missing_results_count = 0
    all_loaded_ks = sorted(k_to_experiment_dir)
    best_k_distribution: Counter = Counter()
    detailed_samples: List[Dict[str, object]] = []
    simple_samples: List[Dict[str, object]] = []

    for question_id in sorted(aggregated_results):
        k_to_result = aggregated_results[question_id]
        available_ks = sorted(k_to_result)

        missing_ks = [k for k in all_loaded_ks if k not in k_to_result]
        if missing_ks:
            missing_results_count += 1
            if len(missing_results_examples) < 5:
                missing_results_examples.append(f"{question_id}: missing k={missing_ks}")

        best_k, best_result = select_best_k(k_to_result)
        best_k_distribution[best_k] += 1

        scores = {
            str(k): {"em": k_to_result[k]["em"], "f1": k_to_result[k]["f1"]}
            for k in available_ks
        }
        predictions = {
            str(k): k_to_result[k]["prediction"]
            for k in available_ks
        }

        detailed_sample = {
            "id": question_id,
            "question": question_texts[question_id],
            "dataset_name": dataset_name,
            "best_k": best_k,
            "best_em": best_result["em"],
            "best_f1": best_result["f1"],
            "scores": scores,
            "predictions": predictions,
        }
        detailed_samples.append(detailed_sample)
        simple_samples.append(
            {
                "id": question_id,
                "question": question_texts[question_id],
                "label": best_k,
            }
        )

    if missing_results_count:
        logger.warning(
            "%d/%d questions are missing results for at least one loaded k. Examples: %s",
            missing_results_count,
            len(detailed_samples),
            "; ".join(missing_results_examples),
        )

    unused_processed_data_ids = len(set(question_texts) - aggregated_question_ids)
    if unused_processed_data_ids:
        logger.info(
            "%d questions from processed data have no aggregated per-question results and were not included.",
            unused_processed_data_ids,
        )

    logger.info(
        "Coverage by k: %s",
        ", ".join(f"{k}:{coverage_by_k[k]}" for k in sorted(coverage_by_k)),
    )
    logger.info("Aggregated %d questions", len(detailed_samples))
    logger.info(
        "best_k distribution: %s",
        json.dumps({str(k): best_k_distribution[k] for k in sorted(best_k_distribution)}, ensure_ascii=False),
    )

    summary = {
        "dataset_name": dataset_name,
        "set_name": set_name,
        "question_count": len(detailed_samples),
        "loaded_k_values": all_loaded_ks,
        "coverage_by_k": {str(k): coverage_by_k[k] for k in sorted(coverage_by_k)},
        "best_k_distribution": {str(k): best_k_distribution[k] for k in sorted(best_k_distribution)},
        "missing_results_count": missing_results_count,
    }
    return detailed_samples, simple_samples, summary


def build_dataset(
    dataset_name: str,
    split_config: SplitConfig,
    logger: logging.Logger,
) -> Dict[str, object]:
    processed_data_path = PROCESSED_DATA_ROOT / dataset_name / f"{split_config.set_name}.jsonl"
    ircot_qa_pattern = get_ircot_qa_pattern(dataset_name, split_config)
    evaluation_name = get_evaluation_name(dataset_name, split_config)

    detailed_samples, simple_samples, summary = build_topk_dataset(
        dataset_name=dataset_name,
        set_name=split_config.set_name,
        processed_data_path=processed_data_path,
        ircot_qa_pattern=ircot_qa_pattern,
        k_min=K_MIN,
        k_max=K_MAX,
        evaluation_name=evaluation_name,
        logger=logger,
    )

    dataset_output_dir = OUTPUT_ROOT / split_config.output_dir_name / dataset_name
    detailed_output_path = dataset_output_dir / f"{split_config.output_prefix}_topk_labels.json"
    simple_output_path = dataset_output_dir / f"{split_config.output_prefix}_topk_train.json"
    summary_output_path = dataset_output_dir / f"{split_config.output_prefix}_topk_summary.json"

    write_json(detailed_samples, detailed_output_path)
    write_json(simple_samples, simple_output_path)
    write_json(summary, summary_output_path)

    logger.info("Wrote detailed top-k labels to %s", detailed_output_path)
    logger.info("Wrote simple top-k train data to %s", simple_output_path)
    logger.info("Wrote top-k summary to %s", summary_output_path)

    return {
        "split": split_config.cli_name,
        "dataset_name": dataset_name,
        "detailed_output_path": str(detailed_output_path),
        "simple_output_path": str(simple_output_path),
        "summary_output_path": str(summary_output_path),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    split_config = SPLIT_CONFIGS[args.split_name]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("build_topk")

    validate_split_inputs(split_config)

    results: List[Dict[str, object]] = []
    for dataset_name in DATASET_NAMES:
        results.append(build_dataset(dataset_name, split_config, logger))

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
