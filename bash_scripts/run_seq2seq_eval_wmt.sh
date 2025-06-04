#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

#MODEL_PATH="${RUN_DIR}/cnn-dm-block4-bs128-keep1-causalencfalse-max10000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen600m_v1"
#OUTPUT_DIR="${MODEL_PATH}/wmt"
MODEL_PATH="yairschiff/wmt-e2d2-qwen600M"
OUTPUT_DIR="${RUN_DIR}/${MODEL_PATH}"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=256
BLOCK_SIZE=4
DO_SAMPLE=false # True, False
SAMPLING_STRATEGY="predict_and_noise"
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
KV_CACHING=True
MAX_LENGTH=1024

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
  +ckpt_file=best-rank0.pt \
  +load_ema_weights=false \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B" \
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
  generation/logits_processor@logits_processor_list='[top_p_logits_wrapper,repetition_penalty_logits_processor]' \
  logits_processor_list.top_p_logits_wrapper.top_p=0.95 \
  logits_processor_list.repetition_penalty_logits_processor.penalty=10.0 \
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,wmt_stop_string_criteria]'
#  ~generation/logits_processor@logits_processor_list \
#  gen_kwargs.logits_processor=null \
