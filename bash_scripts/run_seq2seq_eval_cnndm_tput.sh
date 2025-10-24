#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# TODO: Uncomment a model and run

######## AR
#KV_CACHING=true
#ALIGN_INPUTS_TO_BLOCKS=true
#BLOCK_SIZE=1
#MODEL_PATH="${RUN_DIR}/<PATH_TO_AR_SAVED_MODEL_DIR>"

########### MDLM
#KV_CACHING=false
#ALIGN_INPUTS_TO_BLOCKS=false
#BLOCK_SIZE=32
#MODEL_PATH="${RUN_DIR}/<PATH_TO_MDLM_SAVED_MODEL_DIR>"

########### BD3LM
#KV_CACHING=true
#ALIGN_INPUTS_TO_BLOCKS=true
#BLOCK_SIZE=8
#MODEL_PATH="${RUN_DIR}/<PATH_TO_BD3LM_SAVED_MODEL_DIR>"#LEN_PENALTY=1.1

########### E2D2
BLOCK_SIZE=8
MODEL_PATH="kuleshov-group/e2d2-cnndm"
#MODEL_PATH="${RUN_DIR}/<PATH_TO_E2D2_SAVED_MODEL_DIR>"
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=false

OUTPUT_DIR="outputs/${MODEL_PATH}/cnn_dailymail"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=256
T=${BLOCK_SIZE}
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" "posterior"
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
MAX_LENGTH=4096
CKPT="best"
USE_EMA=true

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}"
PORT=29504
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=cnn_dailymail \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +model_config_overrides.length=${MAX_LENGTH} \
  +ckpt_file="${CKPT}-rank0.pt" \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B-Base" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs.stopping_criteria=null \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  +throughput_run=true
