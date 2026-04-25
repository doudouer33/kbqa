import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_question_texts(processed_data_path: str) -> Dict[str, str]:
    path = Path(processed_data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    id_to_question: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                instance = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of processed data file: {path}"
                ) from exc

            if "question_id" not in instance:
                raise KeyError(f"Missing 'question_id' on line {line_number} of {path}")
            if "question_text" not in instance:
                raise KeyError(f"Missing 'question_text' on line {line_number} of {path}")

            question_id = str(instance["question_id"])
            question_text = instance["question_text"]
            if question_text is None:
                raise ValueError(f"'question_text' is null for id '{question_id}' in {path}")

            if question_id in id_to_question:
                raise ValueError(f"Duplicate question_id '{question_id}' found in {path}")

            id_to_question[question_id] = str(question_text)

    return id_to_question


def find_per_question_eval_file(
    experiment_dir: str, set_name: str, evaluation_name: Optional[str] = None
) -> Path:
    directory = Path(experiment_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {directory}")

    candidates = sorted(directory.glob("per_question_eval__*.json"))
    if not candidates:
        raise FileNotFoundError(f"No per_question_eval__*.json file found in {directory}")

    if evaluation_name:
        target_name = f"per_question_eval__{evaluation_name}.json"
        matches = [candidate for candidate in candidates if candidate.name == target_name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"Expected {target_name} in {directory}, found {[candidate.name for candidate in candidates]}"
            )
        raise ValueError(f"Multiple files matched {target_name} in {directory}")

    if len(candidates) == 1:
        return candidates[0]

    set_name_matches = [candidate for candidate in candidates if candidate.stem.endswith(f"__{set_name}")]
    if len(set_name_matches) == 1:
        return set_name_matches[0]
    if len(set_name_matches) > 1:
        raise ValueError(
            f"Multiple per_question_eval files matched set_name='{set_name}' in {directory}: "
            f"{[candidate.name for candidate in set_name_matches]}"
        )

    raise ValueError(
        f"Multiple per_question_eval files found in {directory}: {[candidate.name for candidate in candidates]}. "
        "Pass --evaluation_name to disambiguate."
    )


def load_per_question_eval(per_question_eval_path: str) -> Dict[str, Dict[str, object]]:
    path = Path(per_question_eval_path)
    if not path.is_file():
        raise FileNotFoundError(f"Per-question eval file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")

    results_by_id: Dict[str, Dict[str, object]] = {}
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} in {path} is not a JSON object")

        if "id" not in item:
            raise KeyError(f"Missing 'id' for item {index} in {path}")
        if "em" not in item:
            raise KeyError(f"Missing 'em' for id '{item['id']}' in {path}")
        if "f1" not in item:
            raise KeyError(f"Missing 'f1' for id '{item['id']}' in {path}")

        question_id = str(item["id"])
        if question_id in results_by_id:
            raise ValueError(f"Duplicate id '{question_id}' found in {path}")

        prediction = item.get("predicted_answer")
        if prediction is None:
            prediction = item.get("prediction")
        if prediction is None:
            raise KeyError(
                f"Neither 'predicted_answer' nor 'prediction' exists for id '{question_id}' in {path}"
            )

        em = item["em"]
        if not isinstance(em, (int, float)) or em not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Expected EM to be 0/1 for id '{question_id}' in {path}, got {em!r}")

        f1 = item["f1"]
        if not isinstance(f1, (int, float)):
            raise ValueError(f"Expected numeric F1 for id '{question_id}' in {path}, got {f1!r}")

        results_by_id[question_id] = {
            "em": int(em),
            "f1": float(f1),
            "prediction": str(prediction),
        }

    return results_by_id


def resolve_k_experiment_dirs(
    nor_qa_dir: Optional[str],
    oner_qa_pattern: Optional[str],
    k_min: int,
    k_max: int,
    logger: logging.Logger,
) -> Dict[int, Path]:
    if k_min < 0:
        raise ValueError(f"k_min must be >= 0, got {k_min}")
    if k_max < k_min:
        raise ValueError(f"k_max must be >= k_min, got k_min={k_min}, k_max={k_max}")

    k_to_dir: Dict[int, Path] = {}

    if k_min <= 0 <= k_max:
        if not nor_qa_dir:
            raise ValueError("k=0 is requested, so --nor_qa_dir is required.")
        nor_path = Path(nor_qa_dir)
        if not nor_path.is_dir():
            raise FileNotFoundError(f"nor_qa directory not found: {nor_path}")
        k_to_dir[0] = nor_path
    elif nor_qa_dir:
        logger.info("Skipping nor_qa_dir because k=0 is outside the requested range.")

    if k_max >= 1:
        if not oner_qa_pattern:
            raise ValueError("k>=1 is requested, so --oner_qa_pattern is required.")
        if "{k}" not in oner_qa_pattern:
            raise ValueError("--oner_qa_pattern must contain the '{k}' placeholder.")

        ignored_zero_dir = Path(oner_qa_pattern.format(k=0))
        if ignored_zero_dir.is_dir():
            logger.info("Ignoring oner_qa retrieval_count=0 directory: %s", ignored_zero_dir)

        for k in range(max(1, k_min), k_max + 1):
            experiment_dir = Path(oner_qa_pattern.format(k=k))
            if not experiment_dir.is_dir():
                raise FileNotFoundError(f"oner_qa directory for k={k} not found: {experiment_dir}")
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
    processed_data_path: str,
    nor_qa_dir: Optional[str],
    oner_qa_pattern: Optional[str],
    k_min: int,
    k_max: int,
    evaluation_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    logger = logger or logging.getLogger(__name__)

    question_texts = load_question_texts(processed_data_path)
    logger.info("Loaded %d questions from %s", len(question_texts), processed_data_path)

    k_to_experiment_dir = resolve_k_experiment_dirs(
        nor_qa_dir=nor_qa_dir,
        oner_qa_pattern=oner_qa_pattern,
        k_min=k_min,
        k_max=k_max,
        logger=logger,
    )
    logger.info("Using %d experiment directories for k values: %s", len(k_to_experiment_dir), sorted(k_to_experiment_dir))

    aggregated_results: Dict[str, Dict[int, Dict[str, object]]] = defaultdict(dict)
    coverage_by_k: Dict[int, int] = {}

    for k in sorted(k_to_experiment_dir):
        experiment_dir = k_to_experiment_dir[k]
        per_question_eval_file = find_per_question_eval_file(
            experiment_dir=str(experiment_dir),
            set_name=set_name,
            evaluation_name=evaluation_name,
        )
        results_by_id = load_per_question_eval(str(per_question_eval_file))
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
