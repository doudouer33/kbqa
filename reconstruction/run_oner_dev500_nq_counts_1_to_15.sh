#!/usr/bin/env bash

# Reconstruct all dev_500 ONER-GPT-NQ runs with BM25 retrieval counts 1..15.
# Every run writes/overwrites its derived reconstructed_inputs JSON and checks
# the replayed question/title sequence against the saved chain.

set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${KBQA_RECON_PYTHON:-/home/dengxin/miniconda3/envs/adaptiverag/bin/python}"
RETRIEVER_HOST="${KBQA_RETRIEVER_HOST:-http://127.0.0.1}"
RETRIEVER_PORT="${KBQA_RETRIEVER_PORT:-8000}"
WORKERS="${KBQA_RECON_WORKERS:-8}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

failed_counts=()

for retrieval_count in {1..15}; do
    experiment_dir="${REPO_ROOT}/predictions/dev_500/oner_qa_gpt_nq____prompt_set_1___bm25_retrieval_count__${retrieval_count}___distractor_count__1"

    echo
    echo "[${retrieval_count}/15] Reconstructing ${experiment_dir}"

    if [[ ! -d "${experiment_dir}" ]]; then
        echo "Missing experiment directory: ${experiment_dir}" >&2
        failed_counts+=("${retrieval_count}")
        continue
    fi

    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/reconstruct_oner.py" \
        "${experiment_dir}" \
        --retriever-host "${RETRIEVER_HOST}" \
        --retriever-port "${RETRIEVER_PORT}" \
        --workers "${WORKERS}" \
        --force \
        --strict \
        "$@"; then
        failed_counts+=("${retrieval_count}")
    fi
done

echo
if (( ${#failed_counts[@]} > 0 )); then
    echo "Finished with failed retrieval counts: ${failed_counts[*]}" >&2
    exit 1
fi

echo "Successfully reconstructed and validated retrieval counts 1..15."
