#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Important variables (fix during hyperparam sweep)
HIDDEN_SIZE=128
INTERMEDIATE_SIZE=284 #$(( 4 * HIDDEN_SIZE ))
N_LAYERS=42
TOP_LAYERS=false
REINIT_MODEL=true
LOGIT_SHIFT=false

# Hyperparameters
LR=3e-4 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
BATCH_SIZE=128
MAX_DURATION="1000000ba" # 20000ba, 10000ba, 5000ba

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG=mdlm_scratch
if [ "${TOP_LAYERS}" == "true" ]; then
  LAYERS="TOPlayers${N_LAYERS}"
else
  LAYERS="layers${N_LAYERS}"
fi
RUN_NAME=cnn_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_max-dur${MAX_DURATION}_${LAYERS}_hidden${HIDDEN_SIZE}_inter${INTERMEDIATE_SIZE}_${TAG}
if [ "${LOGIT_SHIFT}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_logit-shift"
fi

MICRO_BATCH_SIZE=8
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=cnn_dailymail_train \
  dataset@eval_dataset=cnn_dailymail_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="5000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=-1 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=mdlm \
  training.compile_backbone=true \
  model.config.length=1024 \
  model.config.shift_logits=${LOGIT_SHIFT} \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.backbone_config.reinit_model=${REINIT_MODEL} \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=${TOP_LAYERS} \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  +model.config.backbone_config.intermediate_size=${INTERMEDIATE_SIZE} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="1000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true
