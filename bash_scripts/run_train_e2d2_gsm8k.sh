#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Important variables (fix during hyperparam sweep)
BLOCK_SIZE=1
HIDDEN_SIZE=256
N_ENCODER_LAYERS=4
ENCODER_TOP_LAYERS=false
N_DECODER_LAYERS=6
DECODER_TOP_LAYERS=true
REINIT_ENCODER=true
REINIT_DECODER=true
TIE_WEIGHTS=false
LOGIT_SHIFT=false
ENCODER_CAUSAL_MASK=false

# Hyperparameters
LR=8e-3 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
BATCH_SIZE=128 # 96, 128, 256
MAX_DURATION="20000ba" # 20000ba, 10000ba, 5000ba

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG=e2d2_debug-eval-overfit
if [ "${ENCODER_TOP_LAYERS}" == "true" ]; then
  ENC_LAYERS="TOPenc-layers${N_ENCODER_LAYERS}"
else
  ENC_LAYERS="enc-layers${N_ENCODER_LAYERS}"
fi
if [ "${DECODER_TOP_LAYERS}" == "true" ]; then
  DEC_LAYERS="TOPdec-layers${N_DECODER_LAYERS}"
else
  DEC_LAYERS="dec-layers${N_DECODER_LAYERS}"
fi
RUN_NAME=gsm8k_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_hidden-dim${HIDDEN_SIZE}_${ENC_LAYERS}_${DEC_LAYERS}_${TAG}
if [ "${TIE_WEIGHTS}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_tie-weights"
fi
if [ "${LOGIT_SHIFT}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_logit-shift"
fi
if [ "${ENCODER_CAUSAL_MASK}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_encoder-causal-mask"
fi
if [ "${REINIT_ENCODER}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit-encoder"
fi
if [ "${REINIT_DECODER}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_reinit-decoder"
fi
MICRO_BATCH_SIZE=16
NUM_WORKERS=0

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=gsm8k_eval \
  dataset@eval_dataset=gsm8k_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="50ep" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  training.compile_backbone=false \
  model=bd3lm \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder \
  model.config.length=768 \
  model.config.shift_logits=${LOGIT_SHIFT} \
  model.config.backbone_config.use_encoder_causal_mask=${ENCODER_CAUSAL_MASK} \
  model.config.backbone_config.num_encoder_layers=${N_ENCODER_LAYERS} \
  model.config.backbone_config.num_decoder_layers=${N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=${TIE_WEIGHTS} \
  model.config.backbone_config.reinit_decoder=${REINIT_DECODER} \
  model.config.backbone_config.reinit_encoder=${REINIT_ENCODER} \
  model.config.backbone_config.keep_top_decoder_layers=${DECODER_TOP_LAYERS} \
  model.config.backbone_config.keep_top_encoder_layers=${ENCODER_TOP_LAYERS} \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  +model.config.backbone_config.intermediate_size=$(( 4 * HIDDEN_SIZE )) \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  block_size=${BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="50ep" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
