#!/usr/bin/env bash
set -euo pipefail

# Paper Table 2: ROCStories infill ROUGE + throughput on one H100 80GB.

COMMON_ENV=(
  NUM_VISIBLE_DEVICES="${NUM_VISIBLE_DEVICES:-1}"
  CKPT_FILE=none
  USE_EMA=false
  REPEAT_PENALTY=1.2
  CACHE_FULL_INFILL_CONTEXT=true
  INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT=true
  THROUGHPUT_RUN=true
  THROUGHPUT_WARMUP=50
  THROUGHPUT_MEASUREMENTS=1000
  THROUGHPUT_GLOBAL_MEASUREMENTS=true
)

run() {
  local name="$1"
  shift
  echo "==> ROCStories infill: ${name}"
  env "${COMMON_ENV[@]}" "$@" bash_scripts/run_seq2seq_eval_infill_nlp.sh
}

for infill in 1 3; do
  run "ar_infill_${infill}" \
    NUM_TARGET_SENTENCES="${infill}" \
    EVAL_MODEL_KEY=owt:ar \
    BLOCK_SIZE=1 \
    MAX_WINDOW_SIZE=1 \
    ALIGN_INPUTS_TO_BLOCKS=true

  run "mdlm_infill_${infill}" \
    NUM_TARGET_SENTENCES="${infill}" \
    EVAL_MODEL_KEY=owt:mdlm \
    BLOCK_SIZE=1024 \
    MAX_WINDOW_SIZE=1024 \
    ALIGN_INPUTS_TO_BLOCKS=true

  run "bd3lm_s16_infill_${infill}" \
    NUM_TARGET_SENTENCES="${infill}" \
    EVAL_MODEL_KEY=owt:bd3lm-s16 \
    BLOCK_SIZE=16 \
    MAX_WINDOW_SIZE=16 \
    ALIGN_INPUTS_TO_BLOCKS=true

  run "setdlm_smax32_infill_${infill}" \
    NUM_TARGET_SENTENCES="${infill}" \
    EVAL_MODEL_KEY=owt:setdlm-smax32 \
    BLOCK_SIZE=1024 \
    MAX_WINDOW_SIZE=32 \
    ALIGN_INPUTS_TO_BLOCKS=false \
    SETDLM_INFILL_CACHE_PROMOTION_ORDER=l2r
done
