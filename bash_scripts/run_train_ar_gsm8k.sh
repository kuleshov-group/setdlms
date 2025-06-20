#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Hyperparameters
LR=1e-4 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
BATCH_SIZE=128 # 96, 128, 256
MAX_DURATION="20000ba" # 20000ba, 10000ba, 5000ba

REINIT_MODEL=true
HIDDEN_SIZE=768
N_LAYERS=28
TOP_LAYERS=false
PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base
TAG=ar_qwen600M_v2
if [ "${TOP_LAYERS}" == "true" ]; then
  LAYERS="TOPlayers${N_LAYERS}"
else
  LAYERS="layers${N_LAYERS}"
fi
RUN_NAME=gsm8k_lr${LR}_bsz${BATCH_SIZE}_layers${LAYERS}_${TAG}
if [ "${REINIT_MODEL}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit"
fi

MICRO_BATCH_SIZE=4
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="5ep" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=ar \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.length=768 \
  model.config.backbone_config.reinit_model=${REINIT_MODEL} \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=${TOP_LAYERS} \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="5ep" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.save_local=true \
  composer.callbacks.hf_compatible_checkpointing.save_to_hub=false
