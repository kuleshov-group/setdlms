#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Important variables (fix during hyperparam sweep)
BLOCK_SIZE=4
EVAL_BLOCK_SIZE=4
HIDDEN_SIZE=768
INTERMEDIATE_SIZE=$(( 4 * HIDDEN_SIZE ))
N_LAYERS=12
TOP_LAYERS=false
REINIT_MODEL=true

export CUDA_LAUNCH_BLOCKING=0

# Hyperparameters
LR=3e-4
WARMUP_DURATION="2000ba"
BATCH_SIZE=1
MAX_DURATION="5000ba"

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG=bd3lm_gpt2_prof_flex_v3
if [ "${TOP_LAYERS}" == "true" ]; then
  LAYERS="TOPlayers${N_LAYERS}"
else
  LAYERS="layers${N_LAYERS}"
fi
RUN_NAME=owt_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_${LAYERS}_hidden${HIDDEN_SIZE}_inter${INTERMEDIATE_SIZE}_${TAG}
if [ "${REINIT_MODEL}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit"
fi

MICRO_BATCH_SIZE=1
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  tokenizer.pretrained_model_name_or_path="gpt2" \
  dataset@train_dataset=owt_train_gpt2 \
  dataset@eval_dataset=owt_eval_gpt2 \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="10000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=10 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=bd3lm \
  model.config.attn_backend="sdpa" \
  training.compile_backbone=false \
  model.config.length=1024 \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.backbone_config.reinit_model=${REINIT_MODEL} \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=${TOP_LAYERS} \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  +model.config.backbone_config.intermediate_size=${INTERMEDIATE_SIZE} \
  +model.config.backbone_config.vocab_size=50258 \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="1ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true
