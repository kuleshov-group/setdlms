#!/usr/bin/env bash
set -euo pipefail

# Paper GSM8K throughput helper on one H100 80GB.
# Defaults reproduce the SetDLM d4/conf=0.9 row, while MODEL_PATH or
# EVAL_MODEL_KEY can be overridden for other GSM8K Pareto points.

env \
  NUM_VISIBLE_DEVICES="${NUM_VISIBLE_DEVICES:-1}" \
  EVAL_MODEL_KEY="${EVAL_MODEL_KEY:-gsm8k:setdlm-smax8}" \
  THROUGHPUT_RUN=true \
  THROUGHPUT_WARMUP=50 \
  THROUGHPUT_MEASUREMENTS="${THROUGHPUT_MEASUREMENTS:-200}" \
  THROUGHPUT_GLOBAL_MEASUREMENTS=true \
  CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.9}" \
  BLOCK_SIZE="${BLOCK_SIZE:-1024}" \
  MAX_WINDOW_SIZE="${MAX_WINDOW_SIZE:-4}" \
  ALIGN_INPUTS_TO_BLOCKS="${ALIGN_INPUTS_TO_BLOCKS:-false}" \
  bash_scripts/run_lm_eval_harness_tput.sh
