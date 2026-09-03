#!/usr/bin/env bash

# Count GPT input tokens for reconstructed dev_500 ONER-GPT-NQ runs with
# BM25 retrieval counts 1..15. Invalid reconstructions are skipped per question.

set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${KBQA_RECON_PYTHON:-/home/dengxin/miniconda3/envs/adaptiverag/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

failed_counts=()

for retrieval_count in {1..15}; do
    experiment_dir="${REPO_ROOT}/predictions/dev_500/oner_qa_gpt_nq____prompt_set_1___bm25_retrieval_count__${retrieval_count}___distractor_count__1"

    echo
    echo "[${retrieval_count}/15] Counting input tokens in ${experiment_dir}"
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/count_input_tokens.py" \
        "${experiment_dir}" \
        --force \
        "$@"; then
        failed_counts+=("${retrieval_count}")
    fi
done

echo
if (( ${#failed_counts[@]} > 0 )); then
    echo "Token counting failed for retrieval counts: ${failed_counts[*]}" >&2
    exit 1
fi

echo "Successfully counted input tokens for retrieval counts 1..15."
