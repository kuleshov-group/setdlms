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

MODEL_PATH="/home/ubuntu/dllm-dev/outputs/gsm8k/e2d2"
OUTPUT_DIR="${MODEL_PATH}/lm_eval_harness_output"
mkdir -p ${OUTPUT_DIR}
L=128
BLOCK_SIZE=4
GREEDY=False
USE_X0_PRED=False
FIRST_HITTING=False
LOW_CONFIDENCE_REMASKING=False

OUTPUT_PATH="${OUTPUT_DIR}/L=${L}-block_size=${BLOCK_SIZE}-greedy=${GREEDY}-use_x0_pred=${USE_X0_PRED}-first_hitting=${FIRST_HITTING}-low_confidence_remasking=${LOW_CONFIDENCE_REMASKING}.json"

python scripts/harness_eval.py \
  --tasks gsm8k \
  --model lm_eval_harness_model \
  --num_fewshot 0 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ${OUTPUT_PATH} \
  --model_args \
    "max_cont_len=${L},\
model_path=${MODEL_PATH},\
load_ema_weights=True,\
tokenizer_name_or_path=microsoft/Phi-4-mini-reasoning,\
num_samples=1,\
num_steps=8,\
min_t=1e-5,\
top_p=0.9,\
pad_context=False,\
greedy=${GREEDY},\
use_x0_pred=${USE_X0_PRED},\
first_hitting=${FIRST_HITTING},\
low_confidence_remasking=${LOW_CONFIDENCE_REMASKING},\
disable_cache=False,\
kv_caching=False,\
max_length=768,\
block_size=${BLOCK_SIZE},\
shift_logits=True"
