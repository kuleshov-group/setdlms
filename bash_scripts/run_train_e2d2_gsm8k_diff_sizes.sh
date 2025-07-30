#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Important variables (fix during hyperparam sweep)
BLOCK_SIZE=4
EVAL_BLOCK_SIZE=4
N_ENCODER_LAYERS=28
ENCODER_TOP_LAYERS=false
N_DECODER_LAYERS=28
DECODER_TOP_LAYERS=false
REINIT_ENCODER=false
REINIT_DECODER=false
TIE_WEIGHTS=false
FREEZE_ENCODER=false
LOGIT_SHIFT=false
ENCODER_CAUSAL_MASK=false

# Hyperparameters
LR=1e-5
WARMUP_DURATION="100ba"
ALPHA_F=0.5
BATCH_SIZE=1
MAX_DURATION="30000ba"
PRECISION="amp_bf16" # amp_bf16 fp32

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B-Base
DECODER_PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG=e2d2_600Mdec-2Benc-FT_v2
if [ "${ENCODER_TOP_LAYERS}" == "true" ]; then
  ENC_LAYERS="TOPenc${N_ENCODER_LAYERS}"
else
  ENC_LAYERS="enc${N_ENCODER_LAYERS}"
fi
if [ "${DECODER_TOP_LAYERS}" == "true" ]; then
  DEC_LAYERS="TOPdec${N_DECODER_LAYERS}"
else
  DEC_LAYERS="dec${N_DECODER_LAYERS}"
fi
RUN_NAME=gsm8k_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_alphaf${ALPHA_F}_max-dur${MAX_DURATION}_${PRECISION}_${ENC_LAYERS}_${DEC_LAYERS}_${TAG}
if [ "${TIE_WEIGHTS}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_tie-weights"
fi
if [ "${LOGIT_SHIFT}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_logit-shift"
fi
if [ "${ENCODER_CAUSAL_MASK}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_encoder-causal-mask"
fi
if [ "${FREEZE_ENCODER}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_freeze-enc"
fi
#if [ "${REINIT_ENCODER}" == "true" ]; then
#  RUN_NAME="${RUN_NAME}_reinit-encoder"
#fi
#if [ "${REINIT_DECODER}" == "true" ]; then
#  RUN_NAME="${RUN_NAME}_reinit-decoder"
#fi
MICRO_BATCH_SIZE=1 #$(( BATCH_SIZE / NUM_VISIBLE_DEVICES ))
NUM_WORKERS=0
#  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
#  +model.config.backbone_config.intermediate_size=${INTERMEDIATE_SIZE} \


composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.precision=${PRECISION} \
  composer.trainer.eval_interval="1000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=20 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  composer.lr_scheduler.alpha_f=${ALPHA_F} \
  model=e2d2 \
  model.config.attn_backend="sdpa" \
  training.compile_backbone=false \
  model.config.length=768 \
  model.config.shift_logits=${LOGIT_SHIFT} \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder \
  +model.config.backbone_config.decoder_pretrained_model_name_or_path=${DECODER_PRETRAINED_MODEL_NAME_OR_PATH} \
  model.config.backbone_config.use_encoder_causal_mask=${ENCODER_CAUSAL_MASK} \
  model.config.backbone_config.num_encoder_layers=${N_ENCODER_LAYERS} \
  model.config.backbone_config.num_decoder_layers=${N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=${TIE_WEIGHTS} \
  model.config.backbone_config.freeze_encoder=${FREEZE_ENCODER} \
  model.config.backbone_config.reinit_decoder=${REINIT_DECODER} \
  model.config.backbone_config.reinit_encoder=${REINIT_ENCODER} \
  model.config.backbone_config.keep_top_decoder_layers=${DECODER_TOP_LAYERS} \
  model.config.backbone_config.keep_top_encoder_layers=${ENCODER_TOP_LAYERS} \
  model.config.backbone_config.use_gradient_checkpointing=false \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="100ep" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  composer.callbacks.save_best_checkpointing.save_local=false \
  eval_dataloader.batch_size=4
