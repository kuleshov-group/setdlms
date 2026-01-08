#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Model arch
LENGTH=128
BLOCK_SIZE=${LENGTH}
EVAL_BLOCK_SIZE=${LENGTH}
HIDDEN_SIZE=768
INTERMEDIATE_SIZE=3072
N_LAYERS=12
N_HEADS=12
VOCAB_SIZE=30522
DROPOUT=0.1
NORM_TYPE=layernorm
ATTN_BACKEND=sdpa

DESIRED_BLOCK_SIZE=16
MAX_BLOCK_SIZE=${LENGTH}

# Hyperparameters
LR=3e-4
WARMUP_DURATION="2500ba"
BATCH_SIZE=512
MAX_DURATION="1000000ba"

PRETRAINED_MODEL_NAME_OR_PATH=null

TAG="aoarm_dropout${DROPOUT}_norm${NORM_TYPE}_hparam_desired${DESIRED_BLOCK_SIZE}_max${MAX_BLOCK_SIZE}_v1"
LAYERS="layers${N_LAYERS}"
RUN_NAME=lm1b_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_${LAYERS}_hidden${HIDDEN_SIZE}_inter${INTERMEDIATE_SIZE}_${TAG}

GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -E 's/.*(A[0-9]+|H100|A6000).*/\1/' | head -n 1)
if [[ "$GPU_TYPE" == "A100" || "$GPU_TYPE" == "H100" ]]; then
    MICRO_BATCH_SIZE=32
elif [[ "$GPU_TYPE" == "A6000" ]]; then
    MICRO_BATCH_SIZE=64
else
    MICRO_BATCH_SIZE=16
fi
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  tokenizer=autotokenizer \
  tokenizer.pretrained_model_name_or_path=bert-base-uncased \
  dataset@train_dataset=lm1b_train \
  dataset@eval_dataset=lm1b_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="10000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=aoarm_efficient \
  model.config.attn_backend=${ATTN_BACKEND} \
  training.compile_backbone=false \
  model.config.length=${LENGTH} \
  model/backbone@model.config.backbone_config=dit \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  model.config.backbone_config.n_heads=${N_HEADS} \
  model.config.backbone_config.vocab_size=${VOCAB_SIZE} \
  model.config.backbone_config.attn_backend=${ATTN_BACKEND} \
  model.config.backbone_config.dropout=${DROPOUT} \
  model.config.backbone_config.norm_type=${NORM_TYPE} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  eval_dataloader.batch_size=${MICRO_BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="10000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  composer.optimizer.betas=[0.9,0.999] \
  composer.optimizer.weight_decay=0 \
  model.config.keep_clean_bos=true \
  noise@model.config.noise_config=power \
  model.config.noise_config.desired_block_size=${DESIRED_BLOCK_SIZE} \
  model.config.noise_config.max_block_size=${MAX_BLOCK_SIZE} \
  model.config.noise_config.length=${LENGTH} \
  model.config.noise_config.plot_schedule=false \
  model.config.noise_config.int_min=0.1