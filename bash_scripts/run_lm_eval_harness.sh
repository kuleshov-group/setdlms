#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

QWEN_MODEL="Qwen/Qwen3-1.7B-Base"

MODEL_PATH="${RUN_DIR}/gsm8k_FT2B_block4_lr2e-5_b10.9_b20.9998_bsz1_warm10ba_alphaf0.5_max-dur30000ba_precfp32_enc28_TOPdec20_e2d2_ema_edit-dataset"
OUTPUT_DIR="${MODEL_PATH}/lm_eval_harness_output"
REVISION=null

mkdir -p ${OUTPUT_DIR}
L=256
BLOCK_SIZE=4
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
T=4
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
KV_CACHING=true
CKPT_FILE="latest-rank0.pt"
USE_EMA=true

OUTPUT_PATH="${OUTPUT_DIR}/L${L}_block_size${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hitting${FIRST_HITTING}-confidence_based_noising${CONFIDENCE_BASED_NOISING}"
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
  task.model.load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path=${QWEN_MODEL} \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.num_steps=${T} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_threshold=1.1 \
  generation_config.use_cache=${KV_CACHING} \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,gsm8k_regex_stopping_criteria]'
