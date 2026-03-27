#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Model arch
# Local baseline borrowed from:
#   dllm-dev/bash_scripts/run_train_setdlm_gsm8k.sh
# Official Eso-LMs tunables borrowed from:
#   s-sahoo/Eso-LMs/scripts/esolm/train_owt_esolmb_alpha0_1.sh
#   https://raw.githubusercontent.com/s-sahoo/Eso-LMs/main/scripts/esolm/train_owt_esolmb_alpha0_1.sh
SEQ_LEN=1024
N_LAYERS=28
TOP_LAYERS=false
REINIT_MODEL=false

# Eso-LMs-specific knobs.
# Defaults chosen to mirror the official training shell script and algo config:
#   train_owt_esolmb_alpha0_1.sh sets:
#     algo.alpha_0=1.0
#     algo.batch_split=1.0
#     algo.diffusion_shuffle=True
#     algo.diffusion_attn_mode=causal
#     algo.loss_type=low_var
#   configs/algo/esolm.yaml defaults:
#     alpha_0=0.0
#     batch_split=0.0
#     diffusion_shuffle=False
#     diffusion_attn_mode=bidirectional
#     sequential_shuffle=False
#     sequential_attn_mode=causal
#     loss_type=elbo
# Source:
#   https://raw.githubusercontent.com/s-sahoo/Eso-LMs/main/scripts/esolm/train_owt_esolmb_alpha0_1.sh
#   https://raw.githubusercontent.com/s-sahoo/Eso-LMs/main/configs/algo/esolm.yaml
ALPHA_0=1.0
BATCH_SPLIT=1.0
DIFFUSION_SHUFFLE=true
DIFFUSION_ATTN_MODE="causal"
SEQUENTIAL_SHUFFLE=false
SEQUENTIAL_ATTN_MODE="causal"

# Hyperparameters
# The local optimization schedule follows the existing GSM8K SetDLM launcher.
# Change from official Eso-LMs:
#   the upstream shell script trains on OpenWebText with global batch size 64
#   and 250k steps (`loader.batch_size=64`, `trainer.max_steps=250000`), while
#   this repo's GSM8K distillation setup uses much smaller effective batches and
#   a Composer duration schedule.
LR=1e-5
WARMUP_DURATION="100ba"
ALPHA_F=0.5
BATCH_SIZE=1
MAX_DURATION="75000ba"
PRECISION="amp_bf16"

# Debug: Limit training/eval samples per epoch (set to null or remove to use full dataset)
MAX_TRAIN_SAMPLES=null
MAX_EVAL_SAMPLES=null

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B-Base
NUM_SHOT=0

UPSTREAM_LOSS_TYPE="low_var"

TAG="eso_a${ALPHA_0}_bsplit${BATCH_SPLIT}_dshuf${DIFFUSION_SHUFFLE}_dattn${DIFFUSION_ATTN_MODE}_sshuf${SEQUENTIAL_SHUFFLE}_sattn${SEQUENTIAL_ATTN_MODE}_${UPSTREAM_LOSS_TYPE}"
if [ "${DIFFUSION_SHUFFLE}" == "true" ]; then
  WANDB_DIFFUSION_SHUFFLE="1"
else
  WANDB_DIFFUSION_SHUFFLE="0"
fi
if [ "${SEQUENTIAL_SHUFFLE}" == "true" ]; then
  WANDB_SEQUENTIAL_SHUFFLE="1"
else
  WANDB_SEQUENTIAL_SHUFFLE="0"
fi
if [ "${DIFFUSION_ATTN_MODE}" == "causal" ]; then
  WANDB_DIFFUSION_ATTN="c"
else
  WANDB_DIFFUSION_ATTN="${DIFFUSION_ATTN_MODE}"
fi
if [ "${SEQUENTIAL_ATTN_MODE}" == "causal" ]; then
  WANDB_SEQUENTIAL_ATTN="c"
else
  WANDB_SEQUENTIAL_ATTN="${SEQUENTIAL_ATTN_MODE}"
fi
WANDB_TAG="a${ALPHA_0}_b${BATCH_SPLIT}_d${WANDB_DIFFUSION_SHUFFLE}${WANDB_DIFFUSION_ATTN}_s${WANDB_SEQUENTIAL_SHUFFLE}${WANDB_SEQUENTIAL_ATTN}"
if [ "${TOP_LAYERS}" == "true" ]; then
  LAYERS="TOPlayers${N_LAYERS}"
  WANDB_LAYERS="top${N_LAYERS}"
else
  LAYERS="layers${N_LAYERS}"
  WANDB_LAYERS="l${N_LAYERS}"
fi
RUN_NAME=gsm8k-${NUM_SHOT}shot_block${SEQ_LEN}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_alphaf${ALPHA_F}_max-dur${MAX_DURATION}_${PRECISION}_${LAYERS}_${TAG}
WANDB_NAME="gsm8k_${WANDB_LAYERS}_${WANDB_TAG}"
if [ "${REINIT_MODEL}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit"
  WANDB_NAME="${WANDB_NAME}_reinit"
fi
WANDB_NAME="${WANDB_NAME:0:120}"

MICRO_BATCH_SIZE=1
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=gsm8k_train_distill \
  dataset@eval_dataset=gsm8k_eval_distill \
  +train_dataset.max_samples=${MAX_TRAIN_SAMPLES} \
  +eval_dataset.max_samples=${MAX_EVAL_SAMPLES} \
  composer.optimizer.lr=${LR} \
  composer.trainer.precision=${PRECISION} \
  composer.trainer.eval_interval="1000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  composer.lr_scheduler.alpha_f=${ALPHA_F} \
  training.compile_backbone=false \
  model=esolm_upstream \
  model.config.length=${SEQ_LEN} \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.backbone_config.reinit_model=${REINIT_MODEL} \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=${TOP_LAYERS} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  block_size=null \
  eval_block_size=null \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="2000ba" \
  composer.loggers.name=${WANDB_NAME} \
  composer.loggers.init_kwargs.id=${WANDB_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  eval_dataloader.batch_size=1 \
  noise@model.config.noise_config=esolm_loglinear \
  model.config.alpha_0=${ALPHA_0} \
  model.config.batch_split=${BATCH_SPLIT} \
  model.config.diffusion_shuffle=${DIFFUSION_SHUFFLE} \
  model.config.diffusion_attn_mode=${DIFFUSION_ATTN_MODE} \
  model.config.sequential_shuffle=${SEQUENTIAL_SHUFFLE} \
  model.config.sequential_attn_mode=${SEQUENTIAL_ATTN_MODE} \
  model.config.loss_type=${UPSTREAM_LOSS_TYPE}
