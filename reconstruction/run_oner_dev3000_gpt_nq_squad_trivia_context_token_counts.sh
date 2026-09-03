#!/usr/bin/env bash

# Count retrieved-document context tokens for every complete dev_3000
# ONER-GPT NQ, SQuAD, and Trivia reconstruction (BM25 counts 1..15).
# Extra arguments such as --limit 5 are forwarded to count_input_tokens.py.

set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PREDICTIONS_DIR="${REPO_ROOT}/predictions/dev_3000"
PYTHON_BIN="${KBQA_RECON_PYTHON:-/home/dengxin/miniconda3/envs/adaptiverag/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

failed_experiments=()
processed=0

for dataset in nq squad trivia; do
    while IFS= read -r experiment_dir; do
        experiment_name="$(basename -- "${experiment_dir}")"
        if ! compgen -G "${experiment_dir}/reconstructed_inputs__*.json" > /dev/null; then
            echo "Skipping experiment without a reconstruction: ${experiment_name}" >&2
            continue
        fi
        processed=$((processed + 1))

        echo
        echo "[${dataset}] Counting retrieved-context tokens in ${experiment_name}"

        if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/count_input_tokens.py" \
            "${experiment_dir}" \
            --force \
            "$@"; then
            failed_experiments+=("${experiment_name}")
        fi
    done < <(
        find "${PREDICTIONS_DIR}" -mindepth 1 -maxdepth 1 -type d \
            -name "oner_qa_gpt_${dataset}____prompt_set_1___bm25_retrieval_count__*___distractor_count__1" \
            -print | sort -V
    )
done

if (( processed == 0 )); then
    echo "No complete dev_3000 ONER-GPT reconstructions found" >&2
    exit 1
fi

if (( ${#failed_experiments[@]} > 0 )); then
    echo "Token counting failed for ${#failed_experiments[@]} experiment(s):" >&2
    printf '  %s\n' "${failed_experiments[@]}" >&2
    exit 1
fi

echo
echo "Successfully counted retrieved-context tokens for all ${processed} experiments."
