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

#MODEL_PATH="/home/ubuntu/runs/dllm-dev/gsm8k-block4-bs96-keep4-causalencfalse-max20000ba-lr1e-4-warmup1000ba-gc1.0-wd1e-5-bd3_phi_untie_v5"
#MODEL_PATH="/home/ubuntu/test_ckpts"
# MODEL_PATH="/home/ubuntu/runs/dllm-dev/gsm8k-block4-bs96-keep2-causalencfalse-max20000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen2B_v1"
MODEL_PATH="/home/ubuntu/runs/dllm-dev/gsm8k-block4-bs96-keep1-causalencfalse-max20000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-bd3_qwen2B_noema_v2"

#MODEL_PATH="microsoft/Phi-4-mini-reasoning"
#MODEL_PATH="Qwen/Qwen3-0.6B-Base"
OUTPUT_DIR="${MODEL_PATH}/lm_eval_harness_output"
#OUTPUT_DIR="home/ubuntu/qwen3_lm_eval_harness_output"
mkdir -p ${OUTPUT_DIR}
L=256
BLOCK_SIZE=4
GREEDY=False
USE_X0_PRED=True
FIRST_HITTING=True
LOW_CONFIDENCE_REMASKING=True
KV_CACHING=False
TOP_P=1.0 # not used if greedy=True

OUTPUT_PATH="${OUTPUT_DIR}/L=${L}-block_size=${BLOCK_SIZE}-greedy=${GREEDY}-use_x0_pred=${USE_X0_PRED}-first_hitting=${FIRST_HITTING}-low_confidence_remasking=${LOW_CONFIDENCE_REMASKING}"
#OUTPUT_PATH="${OUTPUT_DIR}/outputs.json"

python scripts/harness_eval_debug_train.py \
  --tasks gsm8k \
  --model lm_eval_harness_model \
  --num_fewshot 0 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ${OUTPUT_PATH} \
  --model_args \
    "max_cont_len=${L},\
model_path=${MODEL_PATH},\
load_ema_weights=False,\
tokenizer_name_or_path=Qwen/Qwen3-0.6B-Base,\
ckpt_file=latest-rank0.pt,\
data_split=test,\
num_samples=1,\
num_steps=8,\
min_t=1e-5,\
top_p=${TOP_P},\
pad_context=False,\
greedy=${GREEDY},\
use_x0_pred=${USE_X0_PRED},\
first_hitting=${FIRST_HITTING},\
low_confidence_remasking=${LOW_CONFIDENCE_REMASKING},\
disable_cache=False,\
kv_caching=${KV_CACHING},\
max_length=768,\
block_size=${BLOCK_SIZE},\
shift_logits=True"
