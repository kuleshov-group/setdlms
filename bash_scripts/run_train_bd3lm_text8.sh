#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Model arch
BLOCK_SIZE=4
EVAL_BLOCK_SIZE=4
HIDDEN_SIZE=256
INTERMEDIATE_SIZE=768
N_LAYERS=12  # 12 or 16

# Hyperparameters
LR=3e-4
WARMUP_DURATION="1000ba"
BATCH_SIZE=16
MAX_DURATION="1000000ba"
SCALE=4.0
PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG="bd3lm_baseline"
LAYERS="layers${N_LAYERS}"
RUN_NAME=text8_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_${LAYERS}_hidden${HIDDEN_SIZE}_inter${INTERMEDIATE_SIZE}_${TAG}

GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -E 's/.*(A[0-9]+|H100|A6000).*/\1/' | head -n 1)
if [[ "$GPU_TYPE" == "A100" || "$GPU_TYPE" == "H100" ]]; then
    MICRO_BATCH_SIZE=2
elif [[ "$GPU_TYPE" == "A6000" ]]; then
    MICRO_BATCH_SIZE=2
else
    MICRO_BATCH_SIZE=2
fi
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  tokenizer=text8_tokenizer \
  dataset@train_dataset=text8_train \
  dataset@eval_dataset=text8_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="250ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=aoarm_efficient \
  model.config.attn_backend="sdpa" \
  training.compile_backbone=false \
  model.config.length=2048 \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.backbone_config.reinit_model=true \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=${TOP_LAYERS} \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  +model.config.backbone_config.intermediate_size=${INTERMEDIATE_SIZE} \
  +model.config.backbone_config.vocab_size=35 \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  eval_dataloader.batch_size=${BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="250ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  noise@model.config.noise_config=staggered \
  model.config.noise_config.scale=${SCALE} \
  +model.config.train_complement=false \
  +model.config.rm_attn_to_masked_tokens=true \
  +model.config.attend_to_dummy_tokens=true