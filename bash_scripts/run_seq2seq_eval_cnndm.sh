#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

MODEL_PATH="${RUN_DIR}/cnn-dm-block4-bs128-keep1-causalencfalse-max10000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen600m_v1"

OUTPUT_DIR="${MODEL_PATH}/cnn_dailymail"
mkdir -p ${OUTPUT_DIR}
L=256
BLOCK_SIZE=4
DO_SAMPLE=false # True, False
SAMPLING_STRATEGY="predict_and_noise"
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
KV_CACHING=True
MAX_LENGTH=1024
REPETITION_PENALTY=1.5 # set to >1 for CNN/DM!
LEN_PENALTY=1.1
REGULATION_START=50

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-dom_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}_rep-penalty-${REPETITION_PENALTY}_len-penalty-${LEN_PENALTY}_reg-start${REGULATION_START}"
PORT=29501
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=cnn_dailymail \
  pretrained_model_name_or_path=${MODEL_PATH} \
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
  generation/logits_processor@logits_processor_list='[repetition_penalty_logits_processor,exponential_decay_length_penalty]' \
  logits_processor_list.repetition_penalty_logits_processor.penalty=${REPETITION_PENALTY} \
  logits_processor_list.exponential_decay_length_penalty.exponential_decay_length_penalty="[${REGULATION_START},${LEN_PENALTY}]" \
  generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,cnndm_stop_string_criteria]'
