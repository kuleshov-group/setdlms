#!/bin/bash
#SBATCH -J finewev_llama_ar              # Job name
#SBATCH -o ../watch_folder/%x_%j.out  # Output file (%j expands to jobID)
#SBATCH -e ../watch_folder/%x_%j.err  # Error file (%j expands to jobID)
#SBATCH --get-user-env                # Retrieve the users login environment
#SBATCH --partition=gpu,kuleshov               # Request partition
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH -t 960:00:00                  # Time limit (hhsrun:mm:ss)
#SBATCH --mem=64000                   # Server memory requested (per node)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --exclude=brandal,davis-compute-02,ellis-compute-01,yu-compute-01,abdelfattah-compute-02,davis-compute-01,lancer-compute-01,snavely-compute-03,sablab-gpu-11,rush-compute-02,rush-compute-03,elor-compute-01,kuleshov-compute-01

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Hyperparameters
LR=1e-4 # 1e-5, 1e-4, 1e-3
WARMUP_DURATION="1000ba" # 0.1, 0.3, 0.5
LR_SCHEDULER=cosine_annealing_with_warmup # linear_decay_with_warmup, cosine_decay_with_warmup
BATCH_SIZE=96 # 96, 128, 256
GRAD_CLIP=1.0 # 0.25, 0.5, 0.75, 1.0
WEIGHT_DECAY=1e-5 # 1e-5, 1e-3, 1e-1

# Additional variables

TAG=predict_pad-false_vdebug
RUN_NAME=fineweb-ar-bs${BATCH_SIZE}-lr${LR}-warmup${WARMUP_DURATION}-gc${GRAD_CLIP}-wd${WEIGHT_DECAY}-${TAG}

MICRO_BATCH_SIZE=1 # TODO: tune

composer -n ${SLURM_GPUS_ON_NODE} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=Qwen/Qwen3-0.6B-Base \
  dataset@train_dataset=fineweb_streaming_train \
  ~dataset@eval_dataset \
  train_dataset.take_size=1000 \
  train_dataloader.prefetch_factor=2 \
  collator.predict_padding=false \
  composer.optimizer.lr=${LR} \
  composer.optimizer.weight_decay=${WEIGHT_DECAY} \
  composer.algorithms.gradient_clipping.clipping_threshold=${GRAD_CLIP} \
  composer.trainer.eval_interval='5ep' \
  composer.trainer.max_duration='100000ba' \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=${LR_SCHEDULER} \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=ar \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.max_length=768 \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / SLURM_GPUS_ON_NODE / MICRO_BATCH_SIZE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  hydra.run.dir=/share/kuleshov/ma2238/runs/dllm-dev/${RUN_NAME} \
  composer.trainer.save_interval="1ep" \
  composer.loggers.name=${RUN_NAME}
