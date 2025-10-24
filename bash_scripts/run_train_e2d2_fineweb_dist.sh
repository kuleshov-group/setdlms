#!/bin/bash
# Example E2D2 FineWeb training script comptabile with distributed execution
set -exo pipefail

# Validate required variables
required_vars=("WANDB_API_KEY" "HUGGING_FACE_HUB_TOKEN" "WORLD_SIZE" "TAG")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var not set."
        exit 1
    fi
done

# Set variable hyperparameters (varied during hyperparam sweep)
LR=${LR:-3e-4}
WARMUP_DURATION=${WARMUP_DURATION:-"2000ba"}
MAX_DURATION=${MAX_DURATION:-"1000000ba"}
BATCH_SIZE=${BATCH_SIZE:-512}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MACRO_BATCH_SIZE=$(( BATCH_SIZE / WORLD_SIZE ))
GRAD_ACCUM=$(( MACRO_BATCH_SIZE / MICRO_BATCH_SIZE ))

# Set constant hyperparameters (fixed during hyperparam sweep)
BLOCK_SIZE=8
EVAL_BLOCK_SIZE=8
HIDDEN_SIZE=512
INTERMEDIATE_SIZE=$(( 4 * HIDDEN_SIZE ))
N_ENCODER_LAYERS=20
ENCODER_TOP_LAYERS=false
N_DECODER_LAYERS=4
DECODER_TOP_LAYERS=false
REINIT_ENCODER=true
REINIT_DECODER=true
TIE_WEIGHTS=false
ENCODER_CAUSAL_MASK=false
PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

# Set misc. optional variables
# Use explicit downsampling to avoid rate limit errors caused by:
# https://github.com/huggingface/huggingface_hub/pull/3389
# TODO: Switch back to HuggingFaceFW/fineweb-edu when #3389 is fixed
FINEWEB_DATASET_NAME=${FINEWEB_DATASET_NAME:-"PygTesting/fineweb-sample-1BT"}
NUM_WORKERS=${NUM_WORKERS:-2}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
WINDOW_SIZE=${WINDOW_SIZE:-10}

# Set run name
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
RUN_NAME=fineweb_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_max-dur${MAX_DURATION}_${ENC_LAYERS}_${DEC_LAYERS}_hidden${HIDDEN_SIZE}_inter${INTERMEDIATE_SIZE}_${TAG}
if [ "${TIE_WEIGHTS}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_tie-weights"
fi
if [ "${ENCODER_CAUSAL_MASK}" == "true" ]; then
  RUN_NAME="${RUN_NAME}_encoder-causal-mask"
fi

# Run training
composer scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=fineweb_streaming_train \
  train_dataset.name=${FINEWEB_DATASET_NAME} \
  +train_dataloader.prefetch_factor=${PREFETCH_FACTOR} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  ~composer.trainer.parallelism_config \
  ~dataset@eval_dataset \
  collator.predict_padding=false \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="10000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=20 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=e2d2 \
  model.config.attn_backend="sdpa" \
  training.compile_backbone=true \
  model.config.length=1024 \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder \
  model.config.backbone_config.use_encoder_causal_mask=${ENCODER_CAUSAL_MASK} \
  model.config.backbone_config.num_encoder_layers=${N_ENCODER_LAYERS} \
  model.config.backbone_config.num_decoder_layers=${N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=${TIE_WEIGHTS} \
  model.config.backbone_config.reinit_decoder=${REINIT_DECODER} \
  model.config.backbone_config.reinit_encoder=${REINIT_ENCODER} \
  model.config.backbone_config.keep_top_decoder_layers=${DECODER_TOP_LAYERS} \
  model.config.backbone_config.keep_top_encoder_layers=${ENCODER_TOP_LAYERS} \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  +model.config.backbone_config.intermediate_size=${INTERMEDIATE_SIZE} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=${GRAD_ACCUM} \
  ~composer.trainer.compile_config \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval="1000ba" \
  composer.loggers.name=${RUN_NAME} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true \
  composer.callbacks.speed_monitor.window_size=${WINDOW_SIZE}