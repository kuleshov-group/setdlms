# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

CUDA_VISIBLE_DEVICES=0
# GPUS_ON_NODE=$(nvidia-smi -L | wc -l)
GPUS_ON_NODE=1

# Important variables (fix during hyperparam sweep)
BLOCK_SIZE=32 # 16, 32, 64
KEEP_EVERY_N_ENCODER_LAYERS=7 # set to > 1 for debugging
KEEP_EVERY_N_DECODER_LAYERS=14 # 2, 7

# Hyperparameters
LR=1e-4 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
LR_SCHEDULER=cosine_annealing_with_warmup # linear_decay_with_warmup, cosine_decay_with_warmup
BATCH_SIZE=96 # 96, 128, 256
GRAD_CLIP=1.0 # 0.25, 0.5, 0.75, 1.0
WEIGHT_DECAY=1e-5 # 1e-5, 1e-3, 1e-1

# Additional variables
TIE_ENCODER_DECODER_WEIGHTS=true # true, false
USE_ENCODER_CAUSAL_MASK=false # true, false

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base # Qwen/Qwen3-0.6B-Base, Qwen/Qwen3-1.7B-Base, microsoft/Phi-4-mini-reasoning

TAG=qwen1
RUN_NAME=gsm8k-bs${BATCH_SIZE}-block${BLOCK_SIZE}-keep${KEEP_EVERY_N_DECODER_LAYERS}-tie${TIE_ENCODER_DECODER_WEIGHTS}-causalenc${USE_ENCODER_CAUSAL_MASK}-lr${LR}-warmup${WARMUP_DURATION}-gc${GRAD_CLIP}-wd${WEIGHT_DECAY}-${TAG}

composer -n ${GPUS_ON_NODE} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  composer.optimizer.lr=${LR} \
  composer.optimizer.weight_decay=${WEIGHT_DECAY} \
  composer.algorithms.gradient_clipping.clipping_threshold=${GRAD_CLIP} \
  composer.trainer.eval_interval='5ep' \
  composer.trainer.max_duration='100000ba' \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=${LR_SCHEDULER} \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=bd3lm \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder \
  model.config.length=768 \
  model.config.backbone_config.keep_every_n_encoder_layers=${KEEP_EVERY_N_ENCODER_LAYERS} \
  model.config.backbone_config.keep_every_n_decoder_layers=${KEEP_EVERY_N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=${TIE_ENCODER_DECODER_WEIGHTS} \
  model.config.backbone_config.use_encoder_causal_mask=${USE_ENCODER_CAUSAL_MASK} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / GPUS_ON_NODE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  block_size=${BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=/home/ubuntu/runs/dllm-dev/${RUN_NAME} \
  composer.trainer.save_interval="5ep" \
  composer.loggers.name=${RUN_NAME} \
  composer.loggers=null