#!/usr/bin/env bash

# Reconstruct every dev_3000 ONER-GPT run for NQ, SQuAD, and Trivia.
# Experiment directories are discovered dynamically so all complete BM25 runs
# are covered. Incomplete directories without a saved config are skipped.

set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PREDICTIONS_DIR="${REPO_ROOT}/predictions/dev_3000"

PYTHON_BIN="${KBQA_RECON_PYTHON:-/home/dengxin/miniconda3/envs/adaptiverag/bin/python}"
RETRIEVER_HOST="${KBQA_RETRIEVER_HOST:-http://127.0.0.1}"
RETRIEVER_PORT="${KBQA_RETRIEVER_PORT:-8000}"
WORKERS="${KBQA_RECON_WORKERS:-8}"
CACHE_HOST="127.0.0.1"
CACHE_PORT="${KBQA_RECON_CACHE_PORT:-18000}"

# Local retriever traffic must not be routed through a configured HTTP proxy.
NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
export NO_PROXY no_proxy HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -d "${PREDICTIONS_DIR}" ]]; then
    echo "Predictions directory not found: ${PREDICTIONS_DIR}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/cached_retriever_proxy.py" \
    --host "${CACHE_HOST}" \
    --port "${CACHE_PORT}" \
    --upstream-url "${RETRIEVER_HOST}:${RETRIEVER_PORT}/retrieve/" \
    --prefetch-count 15 &
cache_pid=$!
dataset_pids=()
cleanup() {
    for dataset_pid in "${dataset_pids[@]}"; do
        pkill -TERM -P "${dataset_pid}" 2>/dev/null || true
        kill "${dataset_pid}" 2>/dev/null || true
    done
    kill "${cache_pid}" 2>/dev/null || true
    for dataset_pid in "${dataset_pids[@]}"; do
        wait "${dataset_pid}" 2>/dev/null || true
    done
    wait "${cache_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cache_ready=false
for _ in {1..50}; do
    if "${PYTHON_BIN}" -c \
        "import urllib.request; urllib.request.urlopen('http://${CACHE_HOST}:${CACHE_PORT}/health', timeout=1).read()" \
        >/dev/null 2>&1; then
        cache_ready=true
        break
    fi
    sleep 0.1
done
if [[ "${cache_ready}" != true ]]; then
    echo "Cached retriever proxy did not become ready on ${CACHE_HOST}:${CACHE_PORT}" >&2
    exit 1
fi

run_dataset() {
    local dataset="$1"
    local experiment_dirs=()
    local failed_experiments=()
    local experiment_dir experiment_name index total

    while IFS= read -r experiment_dir; do
        if compgen -G "${experiment_dir}/config__*.jsonnet" > /dev/null; then
            experiment_dirs+=("${experiment_dir}")
        else
            echo "Skipping incomplete experiment directory without a config: ${experiment_dir}" >&2
        fi
    done < <(
        find "${PREDICTIONS_DIR}" -mindepth 1 -maxdepth 1 -type d \
            -name "oner_qa_gpt_${dataset}____prompt_set_1___bm25_retrieval_count__*___distractor_count__1" \
            -print | sort -V
    )

    if (( ${#experiment_dirs[@]} == 0 )); then
        echo "No matching ONER-GPT ${dataset} experiment directories found in ${PREDICTIONS_DIR}" >&2
        return 1
    fi

    total=${#experiment_dirs[@]}

    for index in "${!experiment_dirs[@]}"; do
        experiment_dir="${experiment_dirs[index]}"
        experiment_name="$(basename -- "${experiment_dir}")"

        echo
        echo "[${dataset} $((index + 1))/${total}] Reconstructing ${experiment_name}"

        if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/reconstruct_oner.py" \
            "${experiment_dir}" \
            --retriever-host "http://${CACHE_HOST}" \
            --retriever-port "${CACHE_PORT}" \
            --workers "${WORKERS}" \
            --force \
            --strict \
            "${EXTRA_ARGS[@]}"; then
            failed_experiments+=("${experiment_name}")
        fi
    done

    echo
    if (( ${#failed_experiments[@]} > 0 )); then
        echo "${dataset}: reconstruction failed strict validation for ${#failed_experiments[@]} experiment(s):" >&2
        printf '  %s\n' "${failed_experiments[@]}" >&2
        return 1
    fi

    echo "${dataset}: successfully reconstructed and validated all ${total} experiments."
}

EXTRA_ARGS=("$@")
datasets=(nq squad trivia)
for dataset in "${datasets[@]}"; do
    run_dataset "${dataset}" &
    dataset_pids+=("$!")
done

failed_datasets=()
for index in "${!dataset_pids[@]}"; do
    if ! wait "${dataset_pids[index]}"; then
        failed_datasets+=("${datasets[index]}")
    fi
done

if (( ${#failed_datasets[@]} > 0 )); then
    echo "Reconstruction failed for dataset(s): ${failed_datasets[*]}" >&2
    exit 1
fi

echo "Successfully reconstructed and validated all dev_3000 NQ, SQuAD, and Trivia experiments."
