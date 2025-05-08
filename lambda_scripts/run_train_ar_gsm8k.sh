# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# Important variables (fix during hyperparam sweep)
KEEP_EVERY_N_LAYERS=2

# Hyperparameters
LR=1e-4 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
BATCH_SIZE=96 # 96, 128, 256
GRAD_CLIP=1.0 # 0.25, 0.5, 0.75, 1.0
WEIGHT_DECAY=1e-5 # 1e-5, 1e-3
MAX_DURATION="20000ba" # 20000ba, 10000ba, 5000ba

PRETRAINED_MODEL_NAME_OR_PATH=microsoft/Phi-4-mini-reasoning # Qwen/Qwen3-0.6B-Base, Qwen/Qwen3-1.7B-Base, microsoft/Phi-4-mini-reasoning

TAG=ar_phi_v1
RUN_NAME=gsm8k-bs${BATCH_SIZE}-keep${KEEP_EVERY_N_LAYERS}-max${MAX_DURATION}-lr${LR}-warmup${WARMUP_DURATION}-gc${GRAD_CLIP}-wd${WEIGHT_DECAY}-${TAG}

MICRO_BATCH_SIZE=2 # TODO: tune
NUM_WORKERS=64 # TODO: tune

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  composer.optimizer.lr=${LR} \
  composer.optimizer.weight_decay=${WEIGHT_DECAY} \
  composer.algorithms.gradient_clipping.clipping_threshold=${GRAD_CLIP} \
  composer.trainer.eval_interval='4ep' \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=ar \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.length=768 \
  model.config.backbone_config.keep_every_n_layers=${KEEP_EVERY_N_LAYERS} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  training.antithetic_sampling=false \
  hydra.run.dir=/home/ubuntu/runs/dllm-dev/${RUN_NAME} \
  composer.trainer.save_interval="4ep" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS}
