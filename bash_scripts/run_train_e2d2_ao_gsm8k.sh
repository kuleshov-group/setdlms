#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Important variables (fix during hyperparam sweep)
BLOCK_SIZE=4
USE_ENCODER_CAUSAL_MASK=false # true, false
KEEP_BOTTOM_N_ENCODER_LAYERS=-1
KEEP_TOP_N_DECODER_LAYERS=14

# Hyperparameters
LR=1e-5 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
BATCH_SIZE=96 # 96, 128, 256
MAX_DURATION="20000ba" # 20000ba, 10000ba, 5000ba

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B-Base # Qwen/Qwen3-0.6B-Base, Qwen/Qwen3-1.7B-Base, microsoft/Phi-4-mini-reasoning

TAG=e2d2_ao_qwen600M_refactor_v1
RUN_NAME=gsm8k-block${BLOCK_SIZE}-keepbottomenc${KEEP_BOTTOM_N_ENCODER_LAYERS}-keeptopdec${KEEP_TOP_N_DECODER_LAYERS}-${TAG}

MICRO_BATCH_SIZE=1
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
  model=ao_bd3lm \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder \
  model.config.length=768 \
  model.config.backbone_config.keep_bottom_n_encoder_layers=${KEEP_BOTTOM_N_ENCODER_LAYERS} \
  model.config.backbone_config.keep_top_n_decoder_layers=${KEEP_TOP_N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=true \
  model.config.backbone_config.use_encoder_causal_mask=${USE_ENCODER_CAUSAL_MASK} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  block_size=${BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="5ep" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.save_local=false \
  composer.callbacks.hf_compatible_checkpointing.save_to_hub=true \
  composer.callbacks.hf_compatible_checkpointing.hub_repo_id=yairschiff/${RUN_NAME}
