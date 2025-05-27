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
BATCH_SIZE=128 # 96, 128, 256
GRAD_CLIP=1.0 # 0.25, 0.5, 0.75, 1.0
WEIGHT_DECAY=1e-5 # 1e-5, 1e-3
MAX_DURATION="10000ba" # 20000ba, 10000ba, 5000ba

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base # Qwen/Qwen3-0.6B-Base, Qwen/Qwen3-1.7B-Base, microsoft/Phi-4-mini-reasoning

TAG=e2d2_qwen600M
RUN_NAME=wmt-block${BLOCK_SIZE}-bs${BATCH_SIZE}-keepbottomenc${KEEP_BOTTOM_N_ENCODER_LAYERS}-keeptopdec${KEEP_TOP_N_DECODER_LAYERS}-causalenc${USE_ENCODER_CAUSAL_MASK}-max${MAX_DURATION}-lr${LR}-warmup${WARMUP_DURATION}-gc${GRAD_CLIP}-wd${WEIGHT_DECAY}-${TAG}

MICRO_BATCH_SIZE=16
NUM_WORKERS=8

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=wmt_train \
  dataset@eval_dataset=wmt_eval \
  composer.optimizer.lr=${LR} \
  composer.optimizer.weight_decay=${WEIGHT_DECAY} \
  composer.algorithms.gradient_clipping.clipping_threshold=${GRAD_CLIP} \
  composer.trainer.eval_interval="1ep" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=-1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=bd3lm \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder \
  model.config.max_length=1024 \
  model.config.backbone_config.keep_bottom_n_encoder_layers=${KEEP_BOTTOM_N_ENCODER_LAYERS} \
  model.config.backbone_config.keep_top_n_decoder_layers=${KEEP_TOP_N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=true \
  model.config.backbone_config.use_encoder_causal_mask=${USE_ENCODER_CAUSAL_MASK} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  block_size=${BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="2000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS}
