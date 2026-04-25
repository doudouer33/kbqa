import argparse
import json
import logging
from pathlib import Path

try:
    from kbqa_classifier.topk_data_builder import build_topk_dataset
except ImportError:
    from topk_data_builder import build_topk_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build top-k classifier training data from per-question evaluation files."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. nq, trivia, squad")
    parser.add_argument("--set_name", required=True, help="Set name used in processed_data and per_question_eval")
    parser.add_argument("--processed_data_path", required=True, help="Path to processed_data/<dataset>/<set_name>.jsonl")
    parser.add_argument("--output_path", required=True, help="Path for the detailed JSON array output")
    parser.add_argument(
        "--simple_output_path",
        default="",
        help="Optional path for simplified JSON array output with {id, question, label}",
    )
    parser.add_argument(
        "--nor_qa_dir",
        default="",
        help="Directory for the nor_qa experiment. This is used only for k=0.",
    )
    parser.add_argument(
        "--oner_qa_pattern",
        default="",
        help="Directory pattern for oner_qa experiments. Must contain '{k}'. Only k=1..k_max are used.",
    )
    parser.add_argument("--k_min", type=int, default=0, help="Minimum k to include")
    parser.add_argument("--k_max", type=int, default=15, help="Maximum k to include")
    parser.add_argument(
        "--evaluation_name",
        default="",
        help="Optional exact evaluation name used to match per_question_eval__<evaluation_name>.json",
    )
    return parser.parse_args()


def write_json(data, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("build_topk_dataset")

    detailed_samples, simple_samples, summary = build_topk_dataset(
        dataset_name=args.dataset,
        set_name=args.set_name,
        processed_data_path=args.processed_data_path,
        nor_qa_dir=args.nor_qa_dir or None,
        oner_qa_pattern=args.oner_qa_pattern or None,
        k_min=args.k_min,
        k_max=args.k_max,
        evaluation_name=args.evaluation_name or None,
        logger=logger,
    )

    write_json(detailed_samples, args.output_path)
    logger.info("Wrote detailed dataset to %s", args.output_path)

    if args.simple_output_path:
        write_json(simple_samples, args.simple_output_path)
        logger.info("Wrote simplified dataset to %s", args.simple_output_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
