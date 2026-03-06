#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# bd3lm
# MODEL_PATH="kuleshov-group/bd3lm-owt-block_size16"
# BLOCK_SIZE=16
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# COMPILE_BACKBONE=false

# setdlm
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block16_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# MAX_WINDOW_SIZE=16
# BLOCK_SIZE=1024
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# AR_CACHING=true
# COMPILE_BACKBONE=true

# ar
MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-ar-noeos-v4-1"
CKPT_FILE="20-300000.ckpt"
MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
ALIGN_INPUTS_TO_BLOCKS=true
AR_CACHING=true
BLOCK_SIZE=1
COMPILE_BACKBONE=true


# mdlm
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-mdlm-noeos-v4"
# CKPT_FILE="18-300000.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# ALIGN_INPUTS_TO_BLOCKS=false
# AR_CACHING=true
# BLOCK_SIZE=32
# COMPILE_BACKBONE=true

T=${BLOCK_SIZE}

OUTPUT_DIR="output/"
REVISION=null
mkdir -p ${OUTPUT_DIR}
DO_SAMPLE=true
FIRST_HITTING=false
CONFIDENCE_BASED_NOISING=false
CONFIDENCE_MARGIN_BASED_NOISING=false
CONFIDENCE_THRESHOLD=1e6
USE_EMA=true

REPETITION_PENALTY=1.1 # for each model, try 1.1, 1.2
NUCLEUS_P=1.0 # for each model, try 0.95, 1.0
MAX_LENGTH=1024

echo "MODEL_PATH: ${MODEL_PATH} BLOCK_SIZE: ${BLOCK_SIZE} MAX_WINDOW_SIZE: ${MAX_WINDOW_SIZE} NUCLEUS_P: ${NUCLEUS_P} REPETITION_PENALTY: ${REPETITION_PENALTY}"

TOKENIZER_PATH="gpt2"
# TOKENIZER_PATH="bert-base-uncased"

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-T${T}-do_sample-${DO_SAMPLE}-first_hitting-${FIRST_HITTING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}-nucleus_p${NUCLEUS_P}-repetition_penalty${REPETITION_PENALTY}-conf${CONFIDENCE_THRESHOLD}"
PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/uncond_gen_ppl.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file="${CKPT_FILE}" \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path=${TOKENIZER_PATH} \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=$((${MAX_LENGTH} - 1)) \
  block_size=${BLOCK_SIZE} \
  generation@generation_config=set_diffusion_generation_config \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation_config.max_window_size=${MAX_WINDOW_SIZE} \
  generation_config.ar_caching=true \
  generation_config.linear_unmasking=true \
  generation_config.nucleus_p=${NUCLEUS_P} \
  generation/stopping_criteria@stopping_criteria_list='[entropy_eos_stopping_criteria]' \
  batch_size=1 \
  +throughput_run=true \
  +model_config_overrides.noise_config.block_size=${MAX_LENGTH} \
  +model_config_overrides.noise_config.max_block_size=${MAX_LENGTH} \
  +model_config_overrides.noise_config.length=${MAX_LENGTH}  \
  +model_config_overrides.attn_backend=sdpa \
  +model_config_overrides.backbone_config.attn_backend=sdpa \
  generation/logits_processor@logits_processor_list='[repetition_penalty_logits_processor]' \
  logits_processor_list.repetition_penalty_logits_processor.penalty=${REPETITION_PENALTY} \
  +eval_model_name="gpt2-large" \
  +compile_backbone=${COMPILE_BACKBONE}