#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Replace these with your own dataset/checkpoint paths as needed.
# The defaults below are nq examples only and are not hard-coded in Python.
STAGE1_MODEL_DIR="${STAGE1_MODEL_DIR:-output/stage1}"
STAGE2_MODEL_DIR="${STAGE2_MODEL_DIR:-output/stage2}"
TEST_FILE="${TEST_FILE:-kbqa_classifier/data/nq/test_topk_train.json}"
REPLAY_LABEL_FILE="${REPLAY_LABEL_FILE:-kbqa_classifier/data/nq/test_topk_labels.json}"
OUTPUT_DIR="${OUTPUT_DIR:-kbqa_classifier/output/end2end_replay}"
DATASET_NAME="${DATASET_NAME:-}"
SPLIT_NAME="${SPLIT_NAME:-}"
RUN_NAME="${RUN_NAME:-}"
FIXED_KS="${FIXED_KS:-0,1,3,5,10,15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-}"
STAGE1_MAX_LENGTH="${STAGE1_MAX_LENGTH:-128}"
STAGE2_MAX_LENGTH="${STAGE2_MAX_LENGTH:-64}"
DEVICE="${DEVICE:-}"
BUCKET_TO_TOPK_STRATEGY="${BUCKET_TO_TOPK_STRATEGY:-lower_bound}"
SEED="${SEED:-42}"

# If you want to select a GPU, set GPU=0 / 1 / ...
# Example:
#   GPU=0 bash kbqa_classifier/run/run_end2end_replay.sh
if [[ -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
fi

CMD=(
    python kbqa_classifier/eval/eval_end2end_replay.py
    --stage1_model_dir "${STAGE1_MODEL_DIR}"
    --stage2_model_dir "${STAGE2_MODEL_DIR}"
    --test_file "${TEST_FILE}"
    --replay_label_file "${REPLAY_LABEL_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --fixed_ks "${FIXED_KS}"
    --batch_size "${BATCH_SIZE}"
    --stage1_max_length "${STAGE1_MAX_LENGTH}"
    --stage2_max_length "${STAGE2_MAX_LENGTH}"
    --bucket_to_topk_strategy "${BUCKET_TO_TOPK_STRATEGY}"
    --seed "${SEED}"
)

if [[ -n "${DATASET_NAME}" ]]; then
    CMD+=(--dataset_name "${DATASET_NAME}")
fi

if [[ -n "${SPLIT_NAME}" ]]; then
    CMD+=(--split_name "${SPLIT_NAME}")
fi

if [[ -n "${RUN_NAME}" ]]; then
    CMD+=(--run_name "${RUN_NAME}")
fi

if [[ -n "${STAGE1_BATCH_SIZE}" ]]; then
    CMD+=(--stage1_batch_size "${STAGE1_BATCH_SIZE}")
fi

if [[ -n "${STAGE2_BATCH_SIZE}" ]]; then
    CMD+=(--stage2_batch_size "${STAGE2_BATCH_SIZE}")
fi

if [[ -n "${DEVICE}" ]]; then
    CMD+=(--device "${DEVICE}")
fi

"${CMD[@]}"
