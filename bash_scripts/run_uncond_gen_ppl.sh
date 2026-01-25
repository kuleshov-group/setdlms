#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# BLOCK_SIZE=4
# MAX_WINDOW_SIZE=4
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_mdlm_adaln_cleanbos_antithetic_hparam_v2"
# KV_CACHING=false
# ALIGN_INPUTS_TO_BLOCKS=false

MAX_LENGTH=128
L=$((MAX_LENGTH - 1)) # for block diffusion / aoarm, we can override the length here for the correct attn masks. mdlm will use sliding window.

#### BDL3M
BLOCK_SIZE=16
MAX_WINDOW_SIZE=16
# MODEL_PATH="kuleshov-group/bd3lm-owt-block_size16"
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/lm1b_block16_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_dropout0.1_normlayernorm_hparam_v3"
# MODEL_PATH="/home/ubuntu/mar/runs/ablation_bs16_loglinear_final/last-v1.ckpt"
# MODEL_PATH="/home/ubuntu/mar/runs/ablation_bs4_loglinear_final/last-v1.ckpt"
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=true
T=${BLOCK_SIZE}


# #### SBD
# PROMPT_TEXT=null
# BLOCK_SIZE=128
# MAX_WINDOW_SIZE=4
# MODEL_PATH="/home/ubuntu/mar/runs/lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired4_max128_v5"
# # MODEL_PATH="/home/ubuntu/mar/runs/lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired8_max128_v5"
# # MODEL_PATH="/home/ubuntu/mar/runs/lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired16_vlambda"
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false
# # MODEL_PATH="/home/ubuntu/mar/runs/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block16_ft_v5"
# # MODEL_PATH="/home/ubuntu/mar/runs/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block8_ft_v3"
# # MODEL_PATH="/home/ubuntu/mar/runs/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block4_ft_v5"
# T=${MAX_LENGTH}


OUTPUT_DIR="output/"
REVISION=null
mkdir -p ${OUTPUT_DIR}
DO_SAMPLE=true
FIRST_HITTING=false
CONFIDENCE_BASED_NOISING=false
CONFIDENCE_MARGIN_BASED_NOISING=false
CONFIDENCE_THRESHOLD=0.9
CKPT="best"
USE_EMA=true

REPETITION_PENALTY=2.0
NUCLEUS_P=0.9

# TOKENIZER_PATH="gpt2"
TOKENIZER_PATH="bert-base-uncased"

echo "MODEL_PATH: ${MODEL_PATH}"

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-T${T}-do_sample-${DO_SAMPLE}-first_hitting-${FIRST_HITTING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}"
PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/uncond_gen_ppl.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file="${CKPT}-rank0.pt" \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path=${TOKENIZER_PATH} \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation_config.max_window_size=${MAX_WINDOW_SIZE} \
  generation/stopping_criteria@stopping_criteria_list='[entropy_eos_stopping_criteria]' \
  gen_kwargs.return_dict_in_generate=true \
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
  +generation_config.nucleus_p=${NUCLEUS_P}