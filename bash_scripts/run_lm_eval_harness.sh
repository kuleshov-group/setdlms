#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

MODEL_PATH="${RUN_DIR}/gsm8k-block4-keepbottomenc-1-keeptopdec14-e2d2_qwen600M_v2"
OUTPUT_DIR="${MODEL_PATH}/lm_eval_harness_output"
REVISION=null
mkdir -p ${OUTPUT_DIR}
L=256
BLOCK_SIZE=4
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
KV_CACHING=true
TOP_P=0.95
CKPT_FILE="best-rank0.pt"

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}"
mkdir -p ${OUTPUT_PATH}

accelerate launch scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=gsm8k \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  task.model.ckpt_file=${CKPT_FILE} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B-Base" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.use_cache=${KV_CACHING} \
  generation/logits_processor@logits_processor_list='[top_p_logits_wrapper]' \
  logits_processor_list.top_p_logits_wrapper.top_p=${TOP_P} \
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,gsm8k_regex_stopping_criteria]'
