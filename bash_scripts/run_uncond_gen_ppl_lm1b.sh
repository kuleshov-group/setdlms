#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OWT_RUNNER="${REPO_ROOT}/bash_scripts/run_uncond_gen_ppl_owt.sh"

export TOKENIZER_PATH="${TOKENIZER_PATH:-bert-base-uncased}"
export OUTPUT_DATASET_NAME="${OUTPUT_DATASET_NAME:-lm1b}"
export MAX_LENGTH="${MAX_LENGTH:-128}"
export BLOCK_SIZE="${BLOCK_SIZE:-128}"
export MAX_WINDOW_SIZE="${MAX_WINDOW_SIZE:-${BLOCK_SIZE}}"
export THROUGHPUT_RUN="${THROUGHPUT_RUN:-true}"
export SKIP_MAUVE="${SKIP_MAUVE:-true}"

if [ "${THROUGHPUT_RUN}" = "true" ]; then
  export STOPPING_CONFIDENCE_THRESHOLD="${STOPPING_CONFIDENCE_THRESHOLD:-null}"
else
  export STOPPING_CONFIDENCE_THRESHOLD="${STOPPING_CONFIDENCE_THRESHOLD:-0.005}"
fi

if [ ! -f "${OWT_RUNNER}" ]; then
  echo "Delegated runner '${OWT_RUNNER}' not found." >&2
  exit 1
fi

exec bash "${OWT_RUNNER}" "$@"
