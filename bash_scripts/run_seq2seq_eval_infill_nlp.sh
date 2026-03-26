#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

COMPILE_BACKBONE=true

# TODO: Uncomment a model and run
# bd3lm
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block16_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=16
# MAX_WINDOW_SIZE=16
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=true
# USE_EMA=false

# setdlm
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block16_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# MAX_WINDOW_SIZE=32
# BLOCK_SIZE=1024
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# USE_EMA=false

# ar
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-ar-noeos-v4-1/20-300000.ckpt"
# BLOCK_SIZE=1
# USE_EMA=true

# mdlm
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-mdlm-noeos-v4/18-300000.ckpt"
# ALIGN_INPUTS_TO_BLOCKS=true
# BLOCK_SIZE=1024
# USE_EMA=true

echo "MODEL_PATH: ${MODEL_PATH}"

OUTPUT_DIR="outputs/${MODEL_PATH}/roc_stories"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=1024
T=${BLOCK_SIZE}
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" "posterior"
FIRST_HITTING=false
CONFIDENCE_BASED_NOISING=false
CONF_THRESHOLD=1e6
MAX_LENGTH=1024

# TODO: Uncomment an infilling setting and run
# NUM_TARGET_SENTENCES=1
# REPEAT_PENALTY=1.0

NUM_TARGET_SENTENCES=3
REPEAT_PENALTY=1.1

echo "NUM_TARGET_SENTENCES: ${NUM_TARGET_SENTENCES} REPEAT_PENALTY: ${REPEAT_PENALTY}"

OUTPUT_PATH="${OUTPUT_DIR}/num_target_sentences${NUM_TARGET_SENTENCES}/L-${L}-block_size-${BLOCK_SIZE}-T${T}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-repeat_penalty${REPEAT_PENALTY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}"
PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=roc_stories \
  task.dataset.num_target_sentences=${NUM_TARGET_SENTENCES} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file=${CKPT_FILE} \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="gpt2" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation@generation_config=set_diffusion_generation_config \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_threshold=${CONF_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation_config.max_window_size=${MAX_WINDOW_SIZE} \
  generation_config.linear_unmasking=true \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs.stopping_criteria=null \
  generation/logits_processor@logits_processor_list='[repetition_penalty_logits_processor]' \
  logits_processor_list.repetition_penalty_logits_processor.penalty=${REPEAT_PENALTY} \
  +model_config_overrides.backbone_config.attn_backend=sdpa \
  +model_config_overrides.attn_backend=sdpa \
  +compile_backbone=${COMPILE_BACKBONE}
