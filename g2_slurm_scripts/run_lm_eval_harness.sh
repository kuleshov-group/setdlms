#!/bin/bash
#SBATCH -J lm_eval_harness            # Job name
#SBATCH -o ../watch_folder/%x_%j.out  # Output file (%j expands to jobID)
#SBATCH --get-user-env                # Retrieve the users login environment
#SBATCH --partition=kuleshov               # Request partition
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mem=64000                   # Server memory requested (per node)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --mail-user=yzs2@cornell.edu  # Email
#SBATCH --mail-type=END               # Request status by email

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

MODEL_PATH="/share/kuleshov/yzs2/dllm-dev/outputs/gsm8k_train/2025.05.04/221638/checkpoints/HF_best-rank0"

python scripts/harness_eval.py \
  --tasks gsm8k \
  --model lm_eval_harness_model \
  --num_fewshot 0 \
  --batch_size 1 \
  --device cuda:0 \
  --model_args \
    max_cont_len=128,\
model_path=${MODEL_PATH},\
load_ema_weights=False,\
tokenizer_name_or_path=microsoft/Phi-4-mini-reasoning,\
num_samples=1,\
num_steps=4,\
min_t=1e-5,\
top_p=0.9,\
pad_context=False,\
greedy=True,\
use_x0_pred=True,\
first_hitting=True,\
low_confidence_remasking=True,\
disable_cache=False,\
kv_caching=False,\
max_length=768,\
block_size=4,\
shift_logits=True
