# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# Important variables (fix during hyperparam sweep)
BLOCK_SIZE=4
KEEP_EVERY_N_ENCODER_LAYERS=1
USE_ENCODER_CAUSAL_MASK=false # true, false
KEEP_BOTTOM_N_ENCODER_LAYERS=17 # use < 28, or -1 for all

# Hyperparameters
LR=1e-5 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
BATCH_SIZE=128 # 96, 128, 256
GRAD_CLIP=1.0 # 0.25, 0.5, 0.75, 1.0
WEIGHT_DECAY=1e-5 # 1e-5, 1e-3
MAX_DURATION="1000000ba" # 20000ba, 10000ba, 5000ba

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base # Qwen/Qwen3-0.6B-Base, Qwen/Qwen3-1.7B-Base, microsoft/Phi-4-mini-reasoning

TAG=bd3_tiny_qwen600M_v1
RUN_NAME=owt-block${BLOCK_SIZE}-bs${BATCH_SIZE}-keepbottom${KEEP_BOTTOM_N_ENCODER_LAYERS}-keepevery${KEEP_EVERY_N_ENCODER_LAYERS}-causalenc${USE_ENCODER_CAUSAL_MASK}-max${MAX_DURATION}-lr${LR}-warmup${WARMUP_DURATION}-gc${GRAD_CLIP}-wd${WEIGHT_DECAY}-${TAG}

MICRO_BATCH_SIZE=8 # TODO: tune
NUM_WORKERS=128 # TODO: tune

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=owt_train \
  dataset@eval_dataset=owt_eval \
  composer.optimizer.lr=${LR} \
  composer.optimizer.weight_decay=${WEIGHT_DECAY} \
  composer.algorithms.gradient_clipping.clipping_threshold=${GRAD_CLIP} \
  composer.trainer.eval_interval='10000ba' \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=-1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=bd3lm \
  model/backbone@model.config.backbone_config=bd3lm_encoder_decoder \
  model.config.length=1024 \
  model.config.backbone_config.keep_every_n_encoder_layers=${KEEP_EVERY_N_ENCODER_LAYERS} \
  model.config.backbone_config.keep_bottom_n_encoder_layers=${KEEP_BOTTOM_N_ENCODER_LAYERS} \
  model.config.backbone_config.use_encoder_causal_mask=${USE_ENCODER_CAUSAL_MASK} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  block_size=${BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=/home/ubuntu/runs/dllm-dev/${RUN_NAME} \
  composer.trainer.save_interval="10000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  eval_dataloader.num_workers=0
