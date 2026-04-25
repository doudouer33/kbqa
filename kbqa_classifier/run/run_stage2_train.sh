#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_FILE="${TRAIN_FILE:-kbqa_classifier/data/merged/dev_500_topk_train.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-bert-base-uncased}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/stage2}"
MAX_LENGTH="${MAX_LENGTH:-64}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SEED="${SEED:-42}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${SAVE_STEPS:-100}"

# If you want to select a GPU, set GPU=0 / 1 / ...
# Example:
#   GPU=0 bash kbqa_classifier/run/run_stage2_train.sh
if [[ -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
fi

python kbqa_classifier/train/train_stage2.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --do_train True \
    --do_eval False \
    --max_length "${MAX_LENGTH}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --logging_steps "${LOGGING_STEPS}" \
    --seed "${SEED}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}"
