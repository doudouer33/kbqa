#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DEFAULT_PYTHON="${HOME}/miniconda3/envs/adaptiverag/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
    DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

DATE="$(date +%Y_%m_%d)/$(date +%H_%M_%S)"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-bert-base-uncased}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_NAME_OR_PATH}")}"
TRAIN_FILE="${TRAIN_FILE:-kbqa_classifier_mutihop/train_data/train/train.json}"
VALID_FILE="${VALID_FILE:-kbqa_classifier_mutihop/train_data/valid/valid.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-kbqa_classifier_mutihop/outputs}"
EPOCHS="${EPOCHS:-10 20 30 40}"

MAX_LENGTH="${MAX_LENGTH:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
SEED="${SEED:-42}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
SAVE_STEPS="${SAVE_STEPS:-100}"

if [[ -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
fi

for EPOCH in ${EPOCHS}; do
    TRAIN_OUTPUT_DIR="${OUTPUT_ROOT}/classifier/model/${MODEL_TAG}/epoch/${EPOCH}/${DATE}"
    mkdir -p "${TRAIN_OUTPUT_DIR}"

    "${PYTHON_BIN}" kbqa_classifier_mutihop/train/train_classifier.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --train_file "${TRAIN_FILE}" \
        --output_dir "${TRAIN_OUTPUT_DIR}" \
        --max_length "${MAX_LENGTH}" \
        --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --num_train_epochs "${EPOCH}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --seed "${SEED}" \
        --logging_steps "${LOGGING_STEPS}" \
        --save_strategy "${SAVE_STRATEGY}" \
        --save_steps "${SAVE_STEPS}" \
        --do_train True \
        --do_eval False

    VALID_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}/valid"
    mkdir -p "${VALID_OUTPUT_DIR}"

    "${PYTHON_BIN}" kbqa_classifier_mutihop/train/train_classifier.py \
        --model_name_or_path "${TRAIN_OUTPUT_DIR}" \
        --eval_file "${VALID_FILE}" \
        --output_dir "${VALID_OUTPUT_DIR}" \
        --max_length "${MAX_LENGTH}" \
        --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
        --seed "${SEED}" \
        --do_train False \
        --do_eval True
done
