#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

MODEL_PATH="${RUN_DIR}/wmt_block8_lr3e-4_bsz64_warm2000ba_max-dur1000000ba_enc20_dec4_hidden512_inter2048_e2d2_from-scratch_bidir-ctxt_v2"
OUTPUT_DIR="${MODEL_PATH}/wmt"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=256
BLOCK_SIZE=8
DO_SAMPLE=false # True, False
SAMPLING_STRATEGY="predict_and_noise"
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
KV_CACHING=True
MAX_LENGTH=1024
CKPT_FILE="latest-rank0.pt"
USE_EMA=true

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-dom_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}"
PORT=29501
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
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.use_cache=${KV_CACHING} \
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,wmt_stop_string_criteria]' \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
#  generation/logits_processor@logits_processor_list='[top_p_logits_wrapper,repetition_penalty_logits_processor]' \
#  logits_processor_list.top_p_logits_wrapper.top_p=0.95 \
#  logits_processor_list.repetition_penalty_logits_processor.penalty=10.0 \
