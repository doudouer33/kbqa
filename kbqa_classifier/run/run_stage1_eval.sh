#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Replace these with your own dataset/checkpoint paths as needed.
# The default eval file below is only an nq example and is not hard-coded in Python.
EVAL_FILE="${EVAL_FILE:-kbqa_classifier/data/nq/test_topk_train.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-output/stage1}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-kbqa_classifier/output/classifier_eval}"
DATASET_NAME="${DATASET_NAME:-}"
SPLIT_NAME="${SPLIT_NAME:-}"
MODEL_TAG="${MODEL_TAG:-}"
MAX_LENGTH="${MAX_LENGTH:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
SEED="${SEED:-42}"

# If you want to select a GPU, set GPU=0 / 1 / ...
# Example:
#   GPU=0 bash kbqa_classifier/run/run_stage1_eval.sh
if [[ -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
fi

CMD=(
    python kbqa_classifier/train/train_stage1.py
    --do_train False
    --do_eval True
    --eval_file "${EVAL_FILE}"
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --eval_output_dir "${EVAL_OUTPUT_DIR}"
    --max_length "${MAX_LENGTH}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --seed "${SEED}"
)

if [[ -n "${DATASET_NAME}" ]]; then
    CMD+=(--dataset_name "${DATASET_NAME}")
fi

if [[ -n "${SPLIT_NAME}" ]]; then
    CMD+=(--split_name "${SPLIT_NAME}")
fi

if [[ -n "${MODEL_TAG}" ]]; then
    CMD+=(--model_tag "${MODEL_TAG}")
fi

"${CMD[@]}"
