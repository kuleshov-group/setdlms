#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

QWEN_MODEL="Qwen/Qwen3-1.7B-Base"
NUM_FEW_SHOT=0

# TODO: Uncomment a model and run

########### AR
#MODEL_PATH="${RUN_DIR}/<PATH_TO_AR_SAVED_MODEL_DIR>"
#BLOCK_SIZE=1
#KV_CACHING=true
#ALIGN_INPUTS_TO_BLOCKS=true
#USE_EMA=true

############ MDLM
#MODEL_PATH="${RUN_DIR}/<PATH_TO_MDLM_SAVED_MODEL_DIR>"
#BLOCK_SIZE=64
#KV_CACHING=false
#ALIGN_INPUTS_TO_BLOCKS=false
#USE_EMA=true

############ BD3LM
#MODEL_PATH="${RUN_DIR}/<PATH_TO_BD3LM_SAVED_MODEL_DIR>"
#BLOCK_SIZE=4
#KV_CACHING=true
#ALIGN_INPUTS_TO_BLOCKS=true
#USE_EMA=true

######## E2D2
MODEL_PATH="kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B"
# MODEL_PATH="${RUN_DIR}/<PATH_TO_E2D2_SAVED_MODEL_DIR>"
BLOCK_SIZE=4
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=true
USE_EMA=true

OUTPUT_DIR="outputs/${MODEL_PATH}/lm_eval_harness_output"
REVISION=null

L=512
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
T=${BLOCK_SIZE}
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
CONFIDENCE_MARGIN_BASED_NOISING=false
CONFIDENCE_THRESHOLD=1e6
CKPT="best"


OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}rep-penalty-${REPETITION_PENALTY}_len-penalty-${LEN_PENALTY}_reg-start${REGULATION_START}"
OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_${NUM_FEW_SHOT}shot_L${L}_block${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hit${FIRST_HITTING}-conf_noise${CONFIDENCE_BASED_NOISING}-conf_margin_noise${CONFIDENCE_MARGIN_BASED_NOISING}-conf_thold${CONFIDENCE_THRESHOLD}-align_to_blocks${ALIGN_INPUTS_TO_BLOCKS}"
mkdir -p ${OUTPUT_PATH}

accelerate launch scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=gsm8k \
  task.num_fewshot=${NUM_FEW_SHOT} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  task.model.ckpt_file="${CKPT}-rank0.pt" \
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
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,gsm8k_regex_stopping_criteria]'
