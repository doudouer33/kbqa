#!/usr/bin/env python
"""Build top-k labels for nq, trivia, and squad.

Use either --dev or --test. The script uses:
- predictions/<split>/nor_qa_gpt_<dataset>____prompt_set_1 as k=0
- predictions/<split>/oner_qa_gpt_<dataset>____prompt_set_1___bm25_retrieval_count__<k>___distractor_count__1
  as k=1..15

The output structure mirrors kbqa_classifier/topk_data_builder.py:
- detailed samples contain best_k, best_em, best_f1, scores, and predictions
- simple samples contain id, question, and label
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kbqa_classifier.topk_data_builder import build_topk_dataset  # noqa: E402


DATASET_NAMES: Tuple[str, ...] = ("nq", "trivia", "squad")
K_MIN = 0
K_MAX = 15

PROCESSED_DATA_ROOT = PROJECT_ROOT / "processed_data"
OUTPUT_ROOT = PROJECT_ROOT / "kbqa_classifier_new" / "data"


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
        description="Build top-k classifier data for nq, trivia, and squad."
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


def get_predictions_root(split_config: SplitConfig) -> Path:
    return PROJECT_ROOT / "predictions" / split_config.predictions_dir_name


def get_evaluation_name(dataset_name: str, split_config: SplitConfig) -> str:
    return f"{dataset_name}_to_{dataset_name}__{split_config.set_name}"


def get_nor_qa_dir(dataset_name: str, split_config: SplitConfig) -> Path:
    return get_predictions_root(split_config) / f"nor_qa_gpt_{dataset_name}____prompt_set_1"


def get_oner_qa_pattern(dataset_name: str, split_config: SplitConfig) -> Path:
    return (
        get_predictions_root(split_config)
        / f"oner_qa_gpt_{dataset_name}____prompt_set_1___bm25_retrieval_count__{{k}}___distractor_count__1"
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

        nor_qa_dir = get_nor_qa_dir(dataset_name, split_config)
        if not nor_qa_dir.is_dir():
            missing_paths.append(nor_qa_dir)
        else:
            nor_eval_path = get_per_question_eval_path(nor_qa_dir, dataset_name, split_config)
            if not nor_eval_path.is_file():
                missing_paths.append(nor_eval_path)

        oner_qa_pattern = get_oner_qa_pattern(dataset_name, split_config)
        for k in range(1, K_MAX + 1):
            experiment_dir = Path(str(oner_qa_pattern).format(k=k))
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


def build_dataset(
    dataset_name: str,
    split_config: SplitConfig,
    logger: logging.Logger,
) -> Dict[str, object]:
    processed_data_path = PROCESSED_DATA_ROOT / dataset_name / f"{split_config.set_name}.jsonl"
    nor_qa_dir = get_nor_qa_dir(dataset_name, split_config)
    oner_qa_pattern = (
        get_predictions_root(split_config)
        / f"oner_qa_gpt_{dataset_name}____prompt_set_1___bm25_retrieval_count__{{k}}___distractor_count__1"
    )
    evaluation_name = get_evaluation_name(dataset_name, split_config)

    detailed_samples, simple_samples, summary = build_topk_dataset(
        dataset_name=dataset_name,
        set_name=split_config.set_name,
        processed_data_path=str(processed_data_path),
        nor_qa_dir=str(nor_qa_dir),
        oner_qa_pattern=str(oner_qa_pattern),
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
