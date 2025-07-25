#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

MODEL_PATH="${RUN_DIR}/wmt_block4_evalblock4_lr3e-4_bsz128_warm1000ba_max-dur1000000ba_layers12_hidden512_inter1536_bd3lm_scratch"
OUTPUT_DIR="${MODEL_PATH}/wmt"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=256
BLOCK_SIZE=4
T=4
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise" #"predict_and_noise" "posterior"
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
KV_CACHING=true
MAX_LENGTH=1024
CKPT_FILE="latest-rank0.pt"
USE_EMA=true

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}"
PORT=29502
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=wmt \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file=${CKPT_FILE} \
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
  generation/stopping_criteria@stopping_criteria_list='[max_length_criteria,wmt_stop_string_criteria]' \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null
#  generation/logits_processor@logits_processor_list='[repetition_penalty_logits_processor]' \
#  logits_processor_list.repetition_penalty_logits_processor.penalty=10.0
