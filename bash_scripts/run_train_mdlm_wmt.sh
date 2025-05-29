#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Important variables (fix during hyperparam sweep)
KEEP_BOTTOM_N_LAYERS=21

# Hyperparameters
LR=1e-5 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
BATCH_SIZE=128 # 96, 128, 256
GRAD_CLIP=1.0 # 0.25, 0.5, 0.75, 1.0
WEIGHT_DECAY=1e-5 # 1e-5, 1e-3
MAX_DURATION="10000ba" # 20000ba, 10000ba, 5000ba

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG=mdlm_qwen600M
RUN_NAME=wmt-block${BLOCK_SIZE}-bs${BATCH_SIZE}-keepbottom${KEEP_BOTTOM_N_LAYERS}-max${MAX_DURATION}-lr${LR}-warmup${WARMUP_DURATION}-gc${GRAD_CLIP}-wd${WEIGHT_DECAY}-${TAG}

MICRO_BATCH_SIZE=1
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=wmt_train \
  dataset@eval_dataset=wmt_eval \
  composer.optimizer.lr=${LR} \
  composer.optimizer.weight_decay=${WEIGHT_DECAY} \
  composer.algorithms.gradient_clipping.clipping_threshold=${GRAD_CLIP} \
  composer.trainer.eval_interval="1000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=-1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=mdlm \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.shift_logits=true \
  model.config.length=1024 \
  model.config.backbone_config.keep_bottom_n_layers=${KEEP_BOTTOM_N_LAYERS} \
  model.config.backbone_config.reinit_model=false \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="2000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.save_local=false \
  composer.callbacks.hf_compatible_checkpointing.save_to_hub=true \
  composer.callbacks.hf_compatible_checkpointing.hub_repo_id=yairschiff/wmt-bd3lm-qwen600M
