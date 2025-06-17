#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Hyperparameters
LR=1e-5 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
BATCH_SIZE=96 # 96, 128, 256
MAX_DURATION="20000ba" # 20000ba, 10000ba, 5000ba
PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base
TAG=ar-block1-decode_qwen600M_v2
RUN_NAME=gsm8k-${TAG}

MICRO_BATCH_SIZE=8
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
