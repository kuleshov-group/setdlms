#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh


# Important variables (fix during hyperparam sweep)
N_LAYERS=12
TOP_LAYERS=false
REINIT_MODEL=true

# Hyperparameters
LR=3e-4 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="2000ba"
BATCH_SIZE=16
MAX_DURATION="5000ba"

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B-Base

TAG=ar_gpt2_prof_v22
if [ "${TOP_LAYERS}" == "true" ]; then
  LAYERS="TOPlayers${N_LAYERS}"
else
  LAYERS="layers${N_LAYERS}"
fi
RUN_NAME=owt-${NUM_SHOT}shot_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_max-dur${MAX_DURATION}_${LAYERS}_${TAG}
if [ "${REINIT_MODEL}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit"
fi

MICRO_BATCH_SIZE=16
NUM_WORKERS=0

export CUDA_LAUNCH_BLOCKING=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  tokenizer.pretrained_model_name_or_path="gpt2" \
  dataset@train_dataset=owt_train_gpt2 \
  dataset@eval_dataset=owt_eval_gpt2 \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="0ep" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=10 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=ar \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  training.compile_backbone=true \
  model.config.length=1024 \
  model.config.backbone_config.reinit_model=${REINIT_MODEL} \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=${TOP_LAYERS} \
  +model.config.backbone_config.hidden_size=768 \
  +model.config.backbone_config.intermediate_size=3072 \
  +model.config.backbone_config.use_causal_mask=True \
  +model.config.backbone_config.vocab_size=50258 \
  model.config.backbone_config.attn_implementation="sdpa" \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="1ep" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true