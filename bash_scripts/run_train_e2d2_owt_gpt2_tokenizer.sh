#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Model arch
BLOCK_SIZE=4
EVAL_BLOCK_SIZE=4
HIDDEN_SIZE=512
INTERMEDIATE_SIZE=$(( 4 * HIDDEN_SIZE ))
N_ENCODER_LAYERS=20
N_DECODER_LAYERS=4

# Hyperparameters
LR=3e-4
WARMUP_DURATION="2000ba"
BATCH_SIZE=512
MAX_DURATION="1000000ba"

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG=e2d2_gpt2
ENC_LAYERS="enc${N_ENCODER_LAYERS}"
DEC_LAYERS="dec${N_DECODER_LAYERS}"
RUN_NAME=owt_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_max-dur${MAX_DURATION}_${ENC_LAYERS}_${DEC_LAYERS}_hidden${HIDDEN_SIZE}_inter${INTERMEDIATE_SIZE}_${TAG}
MICRO_BATCH_SIZE=16
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
  composer.trainer.save_num_checkpoints_to_keep=-1 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=e2d2 \
  model.config.attn_backend="flex_attention" \
  training.compile_backbone=true \
  model.config.length=1024 \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder \
  model.config.backbone_config.use_encoder_causal_mask=false \
  model.config.backbone_config.num_encoder_layers=${N_ENCODER_LAYERS} \
  model.config.backbone_config.num_decoder_layers=${N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=false \
  model.config.backbone_config.reinit_decoder=true \
  model.config.backbone_config.reinit_encoder=true \
  model.config.backbone_config.keep_top_decoder_layers=false \
  model.config.backbone_config.keep_top_encoder_layers=false \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  +model.config.backbone_config.intermediate_size=${INTERMEDIATE_SIZE} \
  +model.config.backbone_config.vocab_size=50258 \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="1000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true
