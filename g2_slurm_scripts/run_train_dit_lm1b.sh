#!/bin/bash
#SBATCH -J lm1b_dit                   # Job name
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
  run_name=lm1b-dit-noise_bug_fixed_with_autocast_no_adaln \
  tokenizer.pretrained_model_name_or_path=bert-base-uncased \
  dataset@train_dataset=lm1b_preprocessed_train \
  dataset@eval_dataset=lm1b_preprocessed_eval \
  model.config.length=128 \
  model/backbone@model.config.backbone_config=dit \
  model.config.backbone_config.use_adaln=false \
  training.global_batch_size=512 \
  training.grad_accum=2 \
  composer.lr_scheduler.t_warmup='2500ba' \
  ~composer.trainer.parallelism_config \
  train_dataloader.num_workers=0 \
  eval_dataloader.num_workers=0
