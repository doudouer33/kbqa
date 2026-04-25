import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_TRAIN_FILE = "dev_500_topk_train.json"
DEFAULT_LABELS_FILE = "dev_500_topk_labels.json"
DEFAULT_OUTPUT_FILE = "topk_distribution.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize top-k label distributions for one or more datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, comma-separated dataset names, or 'all'.",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Root directory that contains dataset subdirectories.",
    )
    parser.add_argument(
        "--input_file",
        default="",
        help=(
            "Optional input filename under each dataset directory. "
            f"Defaults to {DEFAULT_TRAIN_FILE}, then falls back to {DEFAULT_LABELS_FILE}."
        ),
    )
    parser.add_argument(
        "--output_name",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output filename written under each dataset directory. Default: {DEFAULT_OUTPUT_FILE}",
    )
    return parser.parse_args()


def resolve_datasets(dataset_arg: str, data_dir: Path) -> List[str]:
    normalized = dataset_arg.strip()
    if not normalized:
        raise ValueError("--dataset cannot be empty")

    if normalized.lower() == "all":
        datasets = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
        if not datasets:
            raise FileNotFoundError(f"No dataset directories found under {data_dir}")
        return datasets

    datasets = []
    for part in normalized.split(","):
        dataset_name = part.strip()
        if not dataset_name:
            continue
        if dataset_name not in datasets:
            datasets.append(dataset_name)

    if not datasets:
        raise ValueError(f"Invalid --dataset value: {dataset_arg!r}")

    return datasets


def resolve_input_path(dataset_dir: Path, input_file: str) -> Path:
    if input_file:
        candidate = dataset_dir / input_file
        if not candidate.is_file():
            raise FileNotFoundError(f"Input file not found: {candidate}")
        return candidate

    candidates = [dataset_dir / DEFAULT_TRAIN_FILE, dataset_dir / DEFAULT_LABELS_FILE]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No supported top-k input file found in {dataset_dir}. "
        f"Expected one of {[path.name for path in candidates]}"
    )


def load_json_array(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} in {path} is not a JSON object")

    return data


def detect_label_field(samples: Sequence[Dict[str, object]], path: Path) -> str:
    if not samples:
        raise ValueError(f"Input file is empty: {path}")

    supported_fields = ("label", "best_k")
    detected_fields = [field for field in supported_fields if any(field in sample for sample in samples)]
    if not detected_fields:
        raise KeyError(f"Neither 'label' nor 'best_k' exists in {path}")

    label_field = detected_fields[0]
    missing_indices = [index for index, sample in enumerate(samples) if label_field not in sample]
    if missing_indices:
        preview = missing_indices[:5]
        raise KeyError(f"Field '{label_field}' is missing for items {preview} in {path}")

    return label_field


def normalize_label(value: object, path: Path, index: int, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a valid label at item {index} in {path} field '{field_name}'")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("-"):
            sign = -1
            stripped = stripped[1:]
        else:
            sign = 1
        if stripped.isdigit():
            return sign * int(stripped)

    raise ValueError(f"Invalid label value at item {index} in {path} field '{field_name}': {value!r}")


def build_distribution(
    dataset_name: str,
    input_path: Path,
    samples: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    label_field = detect_label_field(samples, input_path)
    label_counter: Counter[int] = Counter()

    for index, sample in enumerate(samples):
        label = normalize_label(sample[label_field], input_path, index, label_field)
        label_counter[label] += 1

    total_samples = sum(label_counter.values())
    if total_samples == 0:
        raise ValueError(f"No labels counted from {input_path}")

    sorted_labels = sorted(label_counter)
    distribution = {str(label): label_counter[label] for label in sorted_labels}
    proportions = {
        str(label): round(label_counter[label] / total_samples, 6)
        for label in sorted_labels
    }

    return {
        "dataset_name": dataset_name,
        "source_file": input_path.name,
        "label_field": label_field,
        "sample_count": total_samples,
        "label_distribution": distribution,
        "label_proportion": proportions,
    }


def write_json(data: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def summarize_dataset(
    dataset_name: str,
    data_dir: Path,
    input_file: str,
    output_name: str,
    logger: logging.Logger,
) -> Tuple[Path, Dict[str, object]]:
    dataset_dir = data_dir / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    input_path = resolve_input_path(dataset_dir, input_file)
    samples = load_json_array(input_path)
    summary = build_distribution(dataset_name, input_path, samples)

    output_path = dataset_dir / output_name
    write_json(summary, output_path)
    logger.info("Wrote %s", output_path)
    return output_path, summary


def format_brief_distribution(distribution: Dict[str, int]) -> str:
    return ", ".join(f"{label}:{count}" for label, count in distribution.items())


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("summarize_topk")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    datasets = resolve_datasets(args.dataset, data_dir)
    all_summaries = []

    for dataset_name in datasets:
        output_path, summary = summarize_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            input_file=args.input_file,
            output_name=args.output_name,
            logger=logger,
        )
        all_summaries.append(
            {
                "dataset_name": dataset_name,
                "output_path": str(output_path),
                "sample_count": summary["sample_count"],
                "label_distribution": summary["label_distribution"],
            }
        )

    for item in all_summaries:
        print(
            f"{item['dataset_name']}: {item['sample_count']} samples, "
            f"{format_brief_distribution(item['label_distribution'])} -> {item['output_path']}"
        )


if __name__ == "__main__":
    main()
