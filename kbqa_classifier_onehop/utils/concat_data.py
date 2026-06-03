#!/usr/bin/env python
"""Merge train/valid top-k data and build unlabeled prediction data.

This script merges the three dataset outputs under:
- kbqa_classifier_new/data/dev/{nq,trivia,squad}/dev_3000_topk_train.json -> train_data/train/train.json
- kbqa_classifier_new/data/test/{nq,trivia,squad}/test_topk_train.json -> train_data/valid/valid.json

It also writes kbqa_classifier_new/data/prediction.json, which follows the
same format as training data but sets each `label` field to null.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from kbqa_classifier.summarize_topk import build_distribution, load_json_array
except ImportError:
    from summarize_topk import build_distribution, load_json_array  # type: ignore


DATASET_NAMES = ("nq", "trivia", "squad")
SOURCE_DATA_ROOT = PROJECT_ROOT / "kbqa_classifier_new" / "data"
TRAIN_DATA_ROOT = PROJECT_ROOT / "kbqa_classifier_new" / "train_data"
PREDICTION_PATH = SOURCE_DATA_ROOT / "prediction.json"


class SplitConfig(NamedTuple):
    source_split_dir: str
    source_file_name: str
    output_split_dir: str
    output_file_name: str


SPLITS: Dict[str, SplitConfig] = {
    "train": SplitConfig(
        source_split_dir="dev",
        source_file_name="dev_3000_topk_train.json",
        output_split_dir="train",
        output_file_name="train.json",
    ),
    "valid": SplitConfig(
        source_split_dir="test",
        source_file_name="test_topk_train.json",
        output_split_dir="valid",
        output_file_name="valid.json",
    ),
}


def write_json(data: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def assert_unique_ids(samples: Sequence[Dict[str, object]], path: Path) -> None:
    seen_ids = set()
    duplicate_ids = []

    for index, sample in enumerate(samples):
        if "id" not in sample:
            raise KeyError(f"Missing 'id' field at item {index} in {path}")

        sample_id = sample["id"]
        if not isinstance(sample_id, str):
            raise TypeError(f"Expected string id at item {index} in {path}, got {type(sample_id).__name__}")

        if sample_id in seen_ids:
            duplicate_ids.append(sample_id)
        else:
            seen_ids.add(sample_id)

    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"Duplicate ids found in {path}: {preview}")


def load_dataset_file(dataset_dir: Path, filename: str) -> List[Dict[str, object]]:
    path = dataset_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    samples = load_json_array(path)
    assert_unique_ids(samples, path)
    return samples


def load_merged_split(split_name: str, split_config: SplitConfig, logger: logging.Logger) -> List[Dict[str, object]]:
    merged_samples: List[Dict[str, object]] = []
    split_root = SOURCE_DATA_ROOT / split_config.source_split_dir
    if not split_root.is_dir():
        raise FileNotFoundError(f"Source split directory not found: {split_root}")

    for dataset_name in DATASET_NAMES:
        dataset_dir = split_root / dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        samples = load_dataset_file(dataset_dir, split_config.source_file_name)
        merged_samples.extend(samples)
        logger.info("Loaded %s %s: %d samples", split_name, dataset_name, len(samples))

    return merged_samples


def write_merged_split(
    split_name: str,
    split_config: SplitConfig,
    merged_samples: Sequence[Dict[str, object]],
    logger: logging.Logger,
) -> None:
    output_dir = TRAIN_DATA_ROOT / split_config.output_split_dir
    output_path = output_dir / split_config.output_file_name
    distribution_path = output_dir / "topk_distribution.json"

    assert_unique_ids(merged_samples, output_path)
    write_json(list(merged_samples), output_path)

    distribution = build_distribution(
        dataset_name=split_name,
        input_path=output_path,
        samples=merged_samples,
    )
    write_json(distribution, distribution_path)

    logger.info("Wrote %s split to %s", split_name, output_path)
    logger.info("Wrote %s distribution to %s", split_name, distribution_path)


def build_prediction_samples(valid_samples: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    prediction_samples: List[Dict[str, object]] = []
    for index, sample in enumerate(valid_samples):
        if "label" not in sample:
            raise KeyError(f"Missing 'label' field at validation item {index}")

        prediction_sample = dict(sample)
        prediction_sample["label"] = None
        prediction_samples.append(prediction_sample)

    return prediction_samples


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("concat_data")

    merged_train_samples = load_merged_split("train", SPLITS["train"], logger)
    merged_valid_samples = load_merged_split("valid", SPLITS["valid"], logger)

    write_merged_split("train", SPLITS["train"], merged_train_samples, logger)
    write_merged_split("valid", SPLITS["valid"], merged_valid_samples, logger)

    prediction_samples = build_prediction_samples(merged_valid_samples)
    prediction_path = PREDICTION_PATH
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(prediction_samples, prediction_path)
    logger.info("Wrote prediction data to %s", prediction_path)

    print(
        json.dumps(
            {
                "train_output_dir": str(TRAIN_DATA_ROOT / "train"),
                "valid_output_dir": str(TRAIN_DATA_ROOT / "valid"),
                "prediction_path": str(prediction_path),
                "train_samples": len(merged_train_samples),
                "valid_samples": len(merged_valid_samples),
                "prediction_samples": len(prediction_samples),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
