#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh


# Important variables (fix during hyperparam sweep)
N_LAYERS=28
TOP_LAYERS=false
REINIT_MODEL=false

# Hyperparameters
LR=1e-5 # 1e-5, 1e-4, 1e-3
ALPHA_F=0.0
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
BATCH_SIZE=96
MAX_DURATION="20000ba"
PRECISION="amp_bf16" # amp_bf16 fp32

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B-Base
NUM_SHOT=0
TRAIN_ON_CONTEXT=false

TAG=ar
if [ "${TOP_LAYERS}" == "true" ]; then
  LAYERS="TOPlayers${N_LAYERS}"
else
  LAYERS="layers${N_LAYERS}"
fi
RUN_NAME=gsm8k-${NUM_SHOT}shot_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_alphaf${ALPHA_F}_max-dur${MAX_DURATION}_${PRECISION}_${LAYERS}_${TAG}
if [ "${REINIT_MODEL}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit"
fi

MICRO_BATCH_SIZE=1
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  train_dataset.num_shot=${NUM_SHOT} \
  composer.optimizer.lr=${LR} \
  composer.trainer.precision=${PRECISION} \
  composer.trainer.eval_interval="1ep" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  composer.lr_scheduler.alpha_f=${ALPHA_F} \
  model=ar \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.length=768 \
  model.config.backbone_config.reinit_model=${REINIT_MODEL} \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=${TOP_LAYERS} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="1ep" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  eval_dataloader.batch_size=8 \
  model.config.train_on_context=${TRAIN_ON_CONTEXT}
