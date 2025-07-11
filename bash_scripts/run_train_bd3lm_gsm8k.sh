#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Important variables (fix during hyperparam sweep)
BLOCK_SIZE=8
EVAL_BLOCK_SIZE=8
N_LAYERS=28
TOP_LAYERS=false
REINIT_MODEL=false
LOGIT_SHIFT=false

# Hyperparameters
LR=5e-6 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="10ba" # 0.1, 0.3, 0.5
ALPHA_F=0.5
BATCH_SIZE=1 # 96, 128, 256
MAX_DURATION="30000ba"

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B-Base

TAG=bd3lm
if [ "${TOP_LAYERS}" == "true" ]; then
  LAYERS="TOPlayers${N_LAYERS}"
else
  LAYERS="layers${N_LAYERS}"
fi
RUN_NAME=gsm8k_FT-2B_block${BLOCK_SIZE}_evalblock${EVAL_BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_alphaf${ALPHA_F}_max-dur${MAX_DURATION}_${LAYERS}_${TAG}
if [ "${LOGIT_SHIFT}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_logit-shift"
fi
if [ "${REINIT_MODEL}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit"
fi

MICRO_BATCH_SIZE=1 #$(( BATCH_SIZE / NUM_VISIBLE_DEVICES ))
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="100ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  composer.lr_scheduler.alpha_f=${ALPHA_F} \
  training.compile_backbone=false \
  model=bd3lm \
  model.config.length=768 \
  model.config.shift_logits=${LOGIT_SHIFT} \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.backbone_config.reinit_model=${REINIT_MODEL} \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=${TOP_LAYERS} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="100ep" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  eval_dataloader.batch_size=$(( NUM_VISIBLE_DEVICES * 4 ))
