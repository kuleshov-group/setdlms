#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
# TODO: Uncomment a model and run

# setdlm s <= 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/cnn_block1024_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_aoarm_tgt4_vlambda"
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=4
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false

# setdlm s <= 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/cnn_block1024_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_aoarm_tgt8_v3"
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=8
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false

# setdlm s <= 32
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/cnn_block1024_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_aoarm_tgt16_len1k_v2"
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=16
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false

# ar
MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/cnn_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_ar_len768_v1"
BLOCK_SIZE=1
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=true
MAX_WINDOW_SIZE=1

# mdlm
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/cnn_block_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_mdlm_len768_v1"
# BLOCK_SIZE=32
# KV_CACHING=false
# ALIGN_INPUTS_TO_BLOCKS=false

# bd3lm s = 4
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/cnn_block4_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_bd3lm_len1k_v1"
# BLOCK_SIZE=4
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# MAX_WINDOW_SIZE=4

# bd3lm s = 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/cnn_block8_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_bd3lm_len1k_v1"
# BLOCK_SIZE=8
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true

# bd3lm s = 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/cnn_block16_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_bd3lm_len1k_v1"
# BLOCK_SIZE=16
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true

OUTPUT_DIR="outputs/${MODEL_PATH}/cnn_dailymail_t1"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=null
T=${BLOCK_SIZE}
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" "posterior"
FIRST_HITTING=false
CONFIDENCE_BASED_NOISING=false
CONFIDENCE_MARGIN_BASED_NOISING=false
CONF_THRESHOLD=1e6
MAX_LENGTH=768
MAX_NEW_TOKENS=180
CKPT="best"
USE_EMA=true
LEN_PENALTY=1.1
REGULATION_START=80
REPETITION_PENALTY=1.2

echo "MODEL_PATH: ${MODEL_PATH}"
echo "BLOCK_SIZE: ${BLOCK_SIZE}"
echo "KV_CACHING: ${KV_CACHING}"
echo "ALIGN_INPUTS_TO_BLOCKS: ${ALIGN_INPUTS_TO_BLOCKS}"
echo "MAX_WINDOW_SIZE: ${MAX_WINDOW_SIZE}"
echo "LEN_PENALTY: ${LEN_PENALTY}"
echo "REGULATION_START: ${REGULATION_START}"
echo "REPETITION_PENALTY: ${REPETITION_PENALTY}"
echo "CONF_THRESHOLD: ${CONF_THRESHOLD}"
echo "MAX_LENGTH: ${MAX_LENGTH}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "CKPT: ${CKPT}"
echo "USE_EMA: ${USE_EMA}"

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}rep-penalty-${REPETITION_PENALTY}_len-penalty-${LEN_PENALTY}_reg-start${REGULATION_START}"
mkdir -p ${OUTPUT_PATH}

PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=cnn_dailymail \
  +task.dataset.cache_path="" \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file="${CKPT}-rank0.pt" \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B-Base" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${MAX_NEW_TOKENS} \
  block_size=${BLOCK_SIZE} \
  generation@generation_config=set_diffusion_generation_config \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=${CONF_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation_config.max_window_size=${MAX_WINDOW_SIZE} \
  generation/stopping_criteria@stopping_criteria_list='[cnndm_stop_string_criteria]' \
  generation/logits_processor@logits_processor_list='[repetition_penalty_logits_processor,exponential_decay_length_penalty]' \
  logits_processor_list.repetition_penalty_logits_processor.penalty=${REPETITION_PENALTY} \
  logits_processor_list.exponential_decay_length_penalty.exponential_decay_length_penalty="[${REGULATION_START},${LEN_PENALTY}]" \
  generation_config.ar_caching=true