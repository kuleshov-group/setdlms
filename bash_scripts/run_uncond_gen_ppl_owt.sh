#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

COMPILE_BACKBONE=false
SAMPLING_STRATEGY="predict_and_noise"
NOISE_REMOVAL=false

# bd3lm s = 4
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block4_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=4
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=false
# REPETITION_PENALTY=1.0
# NUCLEUS_P=0.9
# MAX_WINDOW_SIZE=4

# bd3lm s = 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block8_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=8
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=false
# REPETITION_PENALTY=1.0
# NUCLEUS_P=0.9
# MAX_WINDOW_SIZE=8

# bd3lm s = 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block16_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=16
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=false
# REPETITION_PENALTY=1.1
# NUCLEUS_P=0.9
# MAX_WINDOW_SIZE=16

# setdlm s <= 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block4_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# MAX_WINDOW_SIZE=4
# BLOCK_SIZE=1024
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# USE_EMA=false
# REPETITION_PENALTY=1.1
# NUCLEUS_P=0.95

# setdlm s <= 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block8_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# MAX_WINDOW_SIZE=8
# BLOCK_SIZE=1024
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# USE_EMA=false
# REPETITION_PENALTY=1.1
# NUCLEUS_P=0.95

# setdlm s <= 32
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block16_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# MAX_WINDOW_SIZE=16
# BLOCK_SIZE=1024
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# USE_EMA=false
# REPETITION_PENALTY=1.05
# NUCLEUS_P=0.95

# ar
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-ar-noeos-v4-1"
# CKPT_FILE="20-300000.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# ALIGN_INPUTS_TO_BLOCKS=true
# BLOCK_SIZE=1
# USE_EMA=true
# NUCLEUS_P=$2
# REPETITION_PENALTY=$1
# MAX_WINDOW_SIZE=1

# mdlm
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-mdlm-noeos-v4"
# CKPT_FILE="18-300000.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# ALIGN_INPUTS_TO_BLOCKS=true
# BLOCK_SIZE=32
# USE_EMA=true
# NUCLEUS_P=$2
# REPETITION_PENALTY=$1
# MAX_WINDOW_SIZE=32
# KV_CACHING=false

# sedd
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-sedd-noeos-v4"
# CKPT_FILE="18-300000.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# BLOCK_SIZE=1024
# ALIGN_INPUTS_TO_BLOCKS=true
# KV_CACHING=false
# USE_EMA=true
# NUCLEUS_P=1.0
# REPETITION_PENALTY=1.0
# MAX_WINDOW_SIZE=1024
# SAMPLING_STRATEGY="analytic"
# NOISE_REMOVAL=true

# esolm
MODEL_PATH="/share/kuleshov/ma2238/dllm-dev-new/dllm-dev/esolmb-alpha0-0d125-batchsplit-0d5-250k.ckpt"
BLOCK_SIZE=1024
MAX_WINDOW_SIZE=1024
ALIGN_INPUTS_TO_BLOCKS=true
KV_CACHING=true
USE_EMA=true
NUCLEUS_P=1.0
REPETITION_PENALTY=1.0

T=${BLOCK_SIZE}

REVISION=null
DO_SAMPLE=true
FIRST_HITTING=false
CONFIDENCE_BASED_NOISING=false
CONFIDENCE_MARGIN_BASED_NOISING=false
CONFIDENCE_THRESHOLD=1e6

MAX_LENGTH=1024
THROUGHPUT_RUN=true
THROUGHPUT_SAMPLES_PER_RANK=200
NUM_SAMPLES=5000 # TODO
if [ "${THROUGHPUT_RUN}" = "true" ]; then
  NUM_SAMPLES=${THROUGHPUT_SAMPLES_PER_RANK}
fi
echo "MODEL_PATH: ${MODEL_PATH} BLOCK_SIZE: ${BLOCK_SIZE} MAX_WINDOW_SIZE: ${MAX_WINDOW_SIZE} NUCLEUS_P: ${NUCLEUS_P} REPETITION_PENALTY: ${REPETITION_PENALTY} THROUGHPUT_RUN: ${THROUGHPUT_RUN}"

TOKENIZER_PATH="gpt2"

OUTPUT_DIR="outputs/${MODEL_PATH}/owt-L-${MAX_LENGTH}-NUM_SAMPLES${NUM_SAMPLES}"
mkdir -p ${OUTPUT_DIR}
OUTPUT_PATH="${OUTPUT_DIR}/block_size-${BLOCK_SIZE}-T${T}-sampling_strategy-${SAMPLING_STRATEGY}-noise_removal-${NOISE_REMOVAL}-do_sample-${DO_SAMPLE}-first_hitting-${FIRST_HITTING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}-nucleus_p${NUCLEUS_P}-repetition_penalty${REPETITION_PENALTY}-conf${CONFIDENCE_THRESHOLD}-max_window_size${MAX_WINDOW_SIZE}"
if [ "${THROUGHPUT_RUN}" = "true" ]; then
  OUTPUT_PATH="${OUTPUT_PATH}-throughput_run"
fi

TORCHRUN_ARGS=(
  hydra.output_subdir=null
  hydra.run.dir="${PWD}"
  hydra/job_logging=disabled
  hydra/hydra_logging=disabled
  pretrained_model_name_or_path=${MODEL_PATH}
  pretrained_model_revision=${REVISION}
  +ckpt_file="${CKPT_FILE}"
  +load_ema_weights=${USE_EMA}
  tokenizer.pretrained_model_name_or_path=${TOKENIZER_PATH}
  output_path=${OUTPUT_PATH}
  generated_samples_output_path=${OUTPUT_PATH}
  max_length=${MAX_LENGTH}
  max_new_tokens=$((${MAX_LENGTH} - 1))
  block_size=${BLOCK_SIZE}
  generation@generation_config=set_diffusion_generation_config
  generation_config.num_steps=${T}
  generation_config.sampling_strategy=${SAMPLING_STRATEGY}
  generation_config.noise_removal=${NOISE_REMOVAL}
  generation_config.do_sample=${DO_SAMPLE}
  generation_config.first_hitting=${FIRST_HITTING}
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING}
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING}
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD}
  generation_config.use_cache=${KV_CACHING}
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS}
  generation_config.max_window_size=${MAX_WINDOW_SIZE}
  generation_config.linear_unmasking=true
  generation_config.nucleus_p=${NUCLEUS_P}
  generation/stopping_criteria@stopping_criteria_list='[entropy_eos_stopping_criteria]'
  batch_size=1
  +throughput_run=${THROUGHPUT_RUN}
  +throughput_samples_per_rank=${THROUGHPUT_SAMPLES_PER_RANK}
  +model_config_overrides.attn_backend=sdpa
  +model_config_overrides.backbone_config.attn_backend=sdpa
  generation/logits_processor@logits_processor_list='[repetition_penalty_logits_processor]'
  logits_processor_list.repetition_penalty_logits_processor.penalty=${REPETITION_PENALTY}
  +compile_backbone=${COMPILE_BACKBONE}
  num_samples=${NUM_SAMPLES}
)

PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} \
  scripts/eval/uncond_gen_ppl.py "${TORCHRUN_ARGS[@]}"
