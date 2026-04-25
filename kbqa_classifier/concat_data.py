import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

try:
    from kbqa_classifier.summarize_topk import (
        DEFAULT_LABELS_FILE,
        DEFAULT_OUTPUT_FILE,
        DEFAULT_TRAIN_FILE,
        build_distribution,
        load_json_array,
    )
except ImportError:
    from summarize_topk import (  # type: ignore
        DEFAULT_LABELS_FILE,
        DEFAULT_OUTPUT_FILE,
        DEFAULT_TRAIN_FILE,
        build_distribution,
        load_json_array,
    )


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).resolve().parent / "data"

    parser = argparse.ArgumentParser(
        description="Concatenate top-k classifier datasets into one dataset directory."
    )
    parser.add_argument(
        "--datasets",
        default="nq,trivia,squad",
        help="Comma-separated dataset names or 'all'. Default: nq,trivia,squad",
    )
    parser.add_argument(
        "--data_dir",
        default=str(default_data_dir),
        help="Root directory that contains dataset subdirectories.",
    )
    parser.add_argument(
        "--output_dataset",
        default="merged",
        help="Name of the merged dataset directory created under data_dir.",
    )
    parser.add_argument(
        "--train_file",
        default=DEFAULT_TRAIN_FILE,
        help=f"Training file name under each dataset directory. Default: {DEFAULT_TRAIN_FILE}",
    )
    parser.add_argument(
        "--labels_file",
        default=DEFAULT_LABELS_FILE,
        help=f"Detailed labels file name under each dataset directory. Default: {DEFAULT_LABELS_FILE}",
    )
    parser.add_argument(
        "--distribution_file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Distribution file name written under the merged dataset directory. Default: {DEFAULT_OUTPUT_FILE}",
    )
    return parser.parse_args()


def write_json(data: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def resolve_datasets(dataset_arg: str, data_dir: Path, output_dataset: str) -> List[str]:
    normalized = dataset_arg.strip()
    if not normalized:
        raise ValueError("--datasets cannot be empty")

    if normalized.lower() == "all":
        datasets = sorted(
            path.name
            for path in data_dir.iterdir()
            if path.is_dir() and path.name != output_dataset
        )
        if not datasets:
            raise FileNotFoundError(f"No dataset directories found under {data_dir}")
        return datasets

    datasets: List[str] = []
    for part in normalized.split(","):
        dataset_name = part.strip()
        if not dataset_name:
            continue
        if dataset_name == output_dataset:
            raise ValueError(
                f"Output dataset {output_dataset!r} cannot also be a source dataset"
            )
        if dataset_name not in datasets:
            datasets.append(dataset_name)

    if not datasets:
        raise ValueError(f"Invalid --datasets value: {dataset_arg!r}")

    return datasets


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


def assert_matching_ids(
    dataset_name: str,
    train_samples: Sequence[Dict[str, object]],
    label_samples: Sequence[Dict[str, object]],
    train_path: Path,
    labels_path: Path,
) -> None:
    train_ids = [sample["id"] for sample in train_samples]
    label_ids = [sample["id"] for sample in label_samples]

    if set(train_ids) != set(label_ids):
        raise ValueError(
            f"Mismatched ids between {train_path} and {labels_path} for dataset {dataset_name}"
        )

    if len(train_ids) != len(label_ids):
        raise ValueError(
            f"Mismatched sample counts between {train_path} and {labels_path} for dataset {dataset_name}"
        )


def load_dataset_file(dataset_dir: Path, filename: str) -> List[Dict[str, object]]:
    path = dataset_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    samples = load_json_array(path)
    assert_unique_ids(samples, path)
    return samples


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("concat_data")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    dataset_names = resolve_datasets(args.datasets, data_dir, args.output_dataset)
    output_dir = data_dir / args.output_dataset

    merged_train_samples: List[Dict[str, object]] = []
    merged_label_samples: List[Dict[str, object]] = []
    source_summaries = []

    for dataset_name in dataset_names:
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        train_path = dataset_dir / args.train_file
        labels_path = dataset_dir / args.labels_file

        train_samples = load_dataset_file(dataset_dir, args.train_file)
        label_samples = load_dataset_file(dataset_dir, args.labels_file)
        assert_matching_ids(dataset_name, train_samples, label_samples, train_path, labels_path)

        merged_train_samples.extend(train_samples)
        merged_label_samples.extend(label_samples)
        source_summaries.append(
            {
                "dataset_name": dataset_name,
                "train_samples": len(train_samples),
                "label_samples": len(label_samples),
            }
        )

    assert_unique_ids(merged_train_samples, output_dir / args.train_file)
    assert_unique_ids(merged_label_samples, output_dir / args.labels_file)

    output_train_path = output_dir / args.train_file
    output_labels_path = output_dir / args.labels_file
    output_distribution_path = output_dir / args.distribution_file

    write_json(merged_train_samples, output_train_path)
    write_json(merged_label_samples, output_labels_path)

    distribution = build_distribution(
        dataset_name=args.output_dataset,
        input_path=output_train_path,
        samples=merged_train_samples,
    )
    write_json(distribution, output_distribution_path)

    logger.info("Wrote merged train data to %s", output_train_path)
    logger.info("Wrote merged label data to %s", output_labels_path)
    logger.info("Wrote merged distribution to %s", output_distribution_path)

    print(
        json.dumps(
            {
                "output_dataset": args.output_dataset,
                "source_datasets": dataset_names,
                "source_summaries": source_summaries,
                "merged_train_samples": len(merged_train_samples),
                "merged_label_samples": len(merged_label_samples),
                "label_distribution": distribution["label_distribution"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
