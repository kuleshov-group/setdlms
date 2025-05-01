#!/bin/bash
#SBATCH -J c4_llama_e2d2              # Job name
#SBATCH -o ../watch_folder/%x_%j.out  # Output file (%j expands to jobID)
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

composer -n ${SLURM_GPUS_ON_NODE} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=gsm8k-qwen3-e2d2-test5 \
  pretrained_model_name_or_path=Qwen/Qwen3-0.6B-Base \
  dataset@train_dataset=gsm8k_train \
  dataset@eval_dataset=gsm8k_eval \
  model=bd3lm \
  model/backbone@model.config.backbone_config=llama_as_encoder_decoder \
  model.config.length=768 \
  model.config.backbone_config.n_encoder_layers=1 \
  model.config.backbone_config.n_decoder_layers=1 \
  model.config.backbone_config.keep_every_n_encoder_layers=1 \
  model.config.backbone_config.keep_every_n_decoder_layers=1 \
  training.global_batch_size=128 \
  training.grad_accum=$(( 128 / SLURM_GPUS_ON_NODE )) \
  ~composer.trainer.compile_config \
  ~composer.trainer.parallelism_config \
  block_size=4 \
  training.antithetic_sampling=false \
  composer.optimizer.lr=1e-5 \
  composer.trainer.eval_interval='1ep' \
  training.autoresume=false
