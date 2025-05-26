#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

MODEL_PATH="${RUN_DIR}/gsm8k-block4-bs96-keeptop14-causalencfalse-max20000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen2B_tie_redo"
OUTPUT_DIR="${MODEL_PATH}/lm_eval_harness_output"
mkdir -p ${OUTPUT_DIR}
L=256
BLOCK_SIZE=4
GREEDY=True
USE_X0_PRED=True
FIRST_HITTING=True
LOW_CONFIDENCE_REMASKING=True
KV_CACHING=True
TOP_P=1.0
CKPT_FILE="best-rank0.pt"

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-greedy-${GREEDY}-use_x0_pred-${USE_X0_PRED}-first_hitting-${FIRST_HITTING}-low_confidence_remasking-${LOW_CONFIDENCE_REMASKING}"
mkdir -p ${OUTPUT_PATH}

python scripts/eval/harness_eval.py \
  +eval/lm_eval_harness@task=gsm8k \
  pretrained_model_name_or_path=${MODEL_PATH} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation/logits_processor@logits_processor_list='[top_p_logits_wrapper]' \
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,gsm8k_regex_stopping_criteria]'
