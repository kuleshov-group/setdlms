#!/bin/bash
#SBATCH -J gsm8k_llama_e2d2              # Job name
#SBATCH -o ../watch_folder/%x_%j.out  # Output file (%j expands to jobID)
#SBATCH -e ../watch_folder/%x_%j.err  # Error file (%j expands to jobID)
#SBATCH --get-user-env                # Retrieve the users login environment
#SBATCH --partition=kuleshov               # Request partition
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mem=64000                   # Server memory requested (per node)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --mail-user=yzs2@cornell.edu  # Email
#SBATCH --mail-type=END               # Request status by email


# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Important variables (fix during hyperparam sweep)
BLOCK_SIZE=32 # 16, 32, 64
KEEP_EVERY_N_DECODER_LAYERS=8 # 2, 4, 8

# Hyperparameters
LR=1e-5 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.2, 0.3, 0.4, 0.5
LR_SCHEDULER=constant_with_warmup # constant_with_warmup, linear_decay_with_warmup, cosine_decay_with_warmup
BATCH_SIZE=128 # 96, 128, 256
GRAD_CLIP=1.0 # 0.25, 0.5, 0.75, 1.0
WEIGHT_DECAY=1e-5 # 1e-5, 1e-3, 1e-1

# Additional variables
SHIFT_LOGITS=true # true, false
REINIT_DECODER=false # true, false

TAG=test2
RUN_NAME=gsm8k-bs${BATCH_SIZE}-block${BLOCK_SIZE}-keep${KEEP_EVERY_N_DECODER_LAYERS}-lr${LR}-warmup${WARMUP_DURATION}-sched${LR_SCHEDULER}-gc${GRAD_CLIP}-wd${WEIGHT_DECAY}-${TAG}

composer -n ${SLURM_GPUS_ON_NODE} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=Qwen/Qwen3-0.6B-Base \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  composer.optimizer.lr=${LR} \
  composer.optimizer.weight_decay=${WEIGHT_DECAY} \
  composer.algorithms.gradient_clipping.clipping_threshold=${GRAD_CLIP} \
  composer.trainer.eval_interval='1ep' \
  composer/lr_scheduler=${LR_SCHEDULER} \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=bd3lm \
  model/backbone@model.config.backbone_config=llama_as_encoder_decoder \
  model.config.length=768 \
  model.config.backbone_config.keep_every_n_encoder_layers=1 \
  model.config.backbone_config.keep_every_n_decoder_layers=${KEEP_EVERY_N_DECODER_LAYERS} \
  model.config.backbone_config.freeze_encoder=true \
  model.config.backbone_config.reinit_decoder=${REINIT_DECODER} \
  model.config.shift_logits=${SHIFT_LOGITS} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / SLURM_GPUS_ON_NODE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  block_size=${BLOCK_SIZE} \
  training.antithetic_sampling=false \
  checkpointing.save_dir=/share/kuleshov/ma2238/runs/dllm-dev/${RUN_NAME} \
  composer.loggers.name=${RUN_NAME}
