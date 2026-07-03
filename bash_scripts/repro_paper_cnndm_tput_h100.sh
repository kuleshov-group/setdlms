#!/usr/bin/env bash
set -euo pipefail

# Paper Table 3: CNN/DailyMail throughput on one H100 80GB.
# This uses the standard CNN/DM evaluation pipeline with throughput logging
# enabled, not the legacy standalone throughput launcher.

COMMON_ENV=(
  NUM_VISIBLE_DEVICES="${NUM_VISIBLE_DEVICES:-1}"
  THROUGHPUT_RUN=true
  THROUGHPUT_WARMUP=50
  THROUGHPUT_MEASUREMENTS=1000
  THROUGHPUT_GLOBAL_MEASUREMENTS=true
  MAX_LENGTH=768
  MAX_NEW_TOKENS=180
  REPETITION_PENALTY=1.2
  LEN_PENALTY=1.1
  REGULATION_START=80
  KV_CACHING=true
  LOGITS_PROCESSOR_NAMES=repetition_penalty_logits_processor,exponential_decay_length_penalty
)

run() {
  local name="$1"
  shift
  echo "==> CNN/DM throughput: ${name}"
  env "${COMMON_ENV[@]}" "$@" bash_scripts/run_seq2seq_eval_cnndm.sh
}

run ar \
  EVAL_MODEL_KEY=cnndm:ar \
  BLOCK_SIZE=1 \
  MAX_WINDOW_SIZE=1 \
  ALIGN_INPUTS_TO_BLOCKS=true

run mdlm \
  EVAL_MODEL_KEY=cnndm:mdlm \
  BLOCK_SIZE=32 \
  MAX_WINDOW_SIZE=32 \
  ALIGN_INPUTS_TO_BLOCKS=true

run bd3lm_s16 \
  EVAL_MODEL_KEY=cnndm:bd3lm-s16 \
  BLOCK_SIZE=16 \
  MAX_WINDOW_SIZE=16 \
  ALIGN_INPUTS_TO_BLOCKS=true

run bd3lm_s8 \
  EVAL_MODEL_KEY=cnndm:bd3lm-s8 \
  BLOCK_SIZE=8 \
  MAX_WINDOW_SIZE=8 \
  ALIGN_INPUTS_TO_BLOCKS=true

run bd3lm_s4 \
  EVAL_MODEL_KEY=cnndm:bd3lm-s4 \
  BLOCK_SIZE=4 \
  MAX_WINDOW_SIZE=4 \
  ALIGN_INPUTS_TO_BLOCKS=true

run setdlm_smax32 \
  EVAL_MODEL_KEY=cnndm:setdlm-smax32 \
  BLOCK_SIZE=768 \
  MAX_WINDOW_SIZE=16 \
  ALIGN_INPUTS_TO_BLOCKS=false

run setdlm_smax16 \
  EVAL_MODEL_KEY=cnndm:setdlm-smax16 \
  BLOCK_SIZE=768 \
  MAX_WINDOW_SIZE=8 \
  ALIGN_INPUTS_TO_BLOCKS=false

run setdlm_smax8 \
  EVAL_MODEL_KEY=cnndm:setdlm-smax8 \
  BLOCK_SIZE=768 \
  MAX_WINDOW_SIZE=4 \
  ALIGN_INPUTS_TO_BLOCKS=false
