#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Model arch
BLOCK_SIZE=1024
EVAL_BLOCK_SIZE=1024
N_LAYERS=1
TOP_LAYERS=false
REINIT_MODEL=false

DESIRED_BLOCK_SIZE=16
MAX_BLOCK_SIZE=48 # 16, 24, 32, 48, 64
ANNEAL_STEPS="0ba"

K=null # null, 0.1, 0.3, 0.5

# Hyperparameters
LR=1e-5
WARMUP_DURATION="100ba"
ALPHA_F=0.5
BATCH_SIZE=1
MAX_DURATION="75000ba"
PRECISION="amp_bf16"

# Debug: Limit training/eval samples per epoch (set to null or remove to use full dataset)
MAX_TRAIN_SAMPLES=null  # Set to null or remove this line to use full dataset
MAX_EVAL_SAMPLES=null  # Set to null or remove this line to use full dataset

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B-Base
# PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base
NUM_SHOT=0

TAG="setdlm_tgt${DESIRED_BLOCK_SIZE}_distill_k${K}_maxblock${MAX_BLOCK_SIZE}_sweep"
if [ "${TOP_LAYERS}" == "true" ]; then
  LAYERS="TOPlayers${N_LAYERS}"
else
  LAYERS="layers${N_LAYERS}"
fi
RUN_NAME=gsm8k-${NUM_SHOT}shot_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_alphaf${ALPHA_F}_max-dur${MAX_DURATION}_${PRECISION}_${LAYERS}_${TAG}
if [ "${REINIT_MODEL}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit"
fi

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
  model=setdlm \
  model.config.length=1024 \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.backbone_config.reinit_model=${REINIT_MODEL} \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=${TOP_LAYERS} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="2000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  eval_dataloader.batch_size=4 \
  noise@model.config.noise_config=staggered \
  model.config.noise_config.desired_block_size=${DESIRED_BLOCK_SIZE} \
  model.config.noise_config.length=1024 \
  +composer/callbacks=log_gradient_variance \
  composer.callbacks.log_gradient_variance.accumulation_steps=2 \
  eval_dataloader.batch_size=1 \
  +model.config.noise_config.k=${K} \
  model.config.noise_config.max_block_size=${MAX_BLOCK_SIZE}