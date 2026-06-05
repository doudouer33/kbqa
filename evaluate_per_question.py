import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List

from evaluate import answer_extractor, load_experiment_config, load_ground_truths, load_predictions
from lib import get_config_file_path_from_name_or_path, infer_dataset_from_file_path
from metrics.drop_eval import get_metrics as score_drop_prediction
from metrics.squad_answer_em_f1 import score_prediction


SQUAD_STYLE_DATASETS = {"nq", "trivia", "squad"}
DROP_STYLE_DATASETS = {"hotpotqa", "2wikimultihopqa", "musique", "iirc"}
SUPPORTED_DATASETS = SQUAD_STYLE_DATASETS | DROP_STYLE_DATASETS


def prepare_prediction_answers(raw_prediction: Any) -> List[str]:
    assert isinstance(raw_prediction, (str, list, tuple))

    if isinstance(raw_prediction, str):
        if raw_prediction.strip().startswith("[") or raw_prediction.strip().endswith("]"):
            prediction = [e for e in raw_prediction.replace('"', "").replace("[", "").replace("]", "").split(",")]
        else:
            prediction = [raw_prediction]
    else:
        prediction = list(raw_prediction)

    prediction = [str(e) for e in prediction]
    prediction = [answer_extractor(prediction_) for prediction_ in prediction]
    return prediction or [""]


def prepare_gold_answers(ground_truth: Any) -> List[str]:
    if isinstance(ground_truth, (list, tuple)):
        return [str(e) for e in ground_truth]
    return [str(ground_truth)]


def score_per_question(dataset: str, prediction_answers: List[str], ground_truth: Any) -> Dict[str, float]:
    gold_answers = prepare_gold_answers(ground_truth)
    if dataset in DROP_STYLE_DATASETS:
        em, f1, precision, recall = score_drop_prediction(prediction_answers, gold_answers)
        return {"em": int(em), "f1": f1, "precision": precision, "recall": recall}

    sample_scores = score_prediction(prediction_answers, gold_answers)
    return {"em": sample_scores["em"], "f1": sample_scores["f1"]}


def main():
    parser = argparse.ArgumentParser(description="Run per-question evaluation.")
    parser.add_argument("experiment_name_or_path", type=str, help="experiment_name_or_path")
    parser.add_argument("evaluation_path", type=str, help="evaluation_path")
    parser.add_argument("--prediction-file-path", type=str, required=True, help="prediction_file_path")
    parser.add_argument("--output-file-path", type=str, required=True, help="output_file_path")
    parser.add_argument("--dataset", type=str, default="", help="dataset")
    parser.add_argument(
        "--question-type-key-value", type=str, help="':' separated question-type-key-value.", default=None
    )
    parser.add_argument("--llm_port_num", type=str, required=True, help="llm_port_num")
    args = parser.parse_args()

    config_filepath = get_config_file_path_from_name_or_path(args.experiment_name_or_path)
    dataset = args.dataset or infer_dataset_from_file_path(args.evaluation_path)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Per-question evaluation currently supports only {sorted(SUPPORTED_DATASETS)}. Got {dataset}."
        )

    experiment_config = load_experiment_config(config_filepath, SimpleNamespace(llm_port_num=args.llm_port_num))
    if experiment_config["prediction_type"] != "answer":
        raise ValueError(
            f"Per-question evaluation currently supports answer predictions only. Got {experiment_config['prediction_type']}."
        )

    question_type_key = question_type_value = None
    if args.question_type_key_value is not None:
        if args.question_type_key_value.count(":") != 1:
            raise ValueError("The question_type_key_value must be : separated.")
        question_type_key, question_type_value = args.question_type_key_value.split(":")
        question_type_key = question_type_key.strip()
        question_type_value = question_type_value.strip()

    id_to_ground_truths = load_ground_truths(
        experiment_config,
        args.evaluation_path,
        question_type_key=question_type_key,
        question_type_value=question_type_value,
    )
    id_to_predictions = load_predictions(args.prediction_file_path)

    if question_type_value is not None:
        id_to_predictions = {
            qid: prediction for qid, prediction in id_to_predictions.items() if qid in id_to_ground_truths
        }

    missing_predictions = sorted(set(id_to_ground_truths) - set(id_to_predictions))
    missing_ground_truths = sorted(set(id_to_predictions) - set(id_to_ground_truths))
    if missing_predictions or missing_ground_truths:
        raise ValueError(
            "Ids in input examples and predictions don't match. "
            f"Missing predictions for {len(missing_predictions)} ids; "
            f"missing ground truths for {len(missing_ground_truths)} ids."
        )

    per_question_results = []
    total_em = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    for qid, ground_truth in id_to_ground_truths.items():
        prediction_answers = prepare_prediction_answers(id_to_predictions[qid])
        gold_answers = prepare_gold_answers(ground_truth)
        sample_scores = score_per_question(dataset, prediction_answers, ground_truth)

        total_em += sample_scores["em"]
        total_f1 += sample_scores["f1"]
        sample_result = {
            "id": qid,
            "prediction": prediction_answers[0],
            "predicted_answer": prediction_answers[0],
            "gold_answers": gold_answers,
            "em": sample_scores["em"],
            "f1": sample_scores["f1"],
        }
        if dataset in DROP_STYLE_DATASETS:
            total_precision += sample_scores["precision"]
            total_recall += sample_scores["recall"]
            sample_result["precision"] = sample_scores["precision"]
            sample_result["recall"] = sample_scores["recall"]
        per_question_results.append(sample_result)

    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
    with open(args.output_file_path, "w") as file:
        json.dump(per_question_results, file, indent=4)

    count = len(per_question_results)
    summary = {
        "count": count,
        "avg_em": total_em / count if count else 0.0,
        "avg_f1": total_f1 / count if count else 0.0,
        "output_file_path": args.output_file_path,
    }
    if dataset in DROP_STYLE_DATASETS:
        summary["avg_precision"] = total_precision / count if count else 0.0
        summary["avg_recall"] = total_recall / count if count else 0.0
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
