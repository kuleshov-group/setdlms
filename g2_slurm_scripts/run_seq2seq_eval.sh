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

MODEL_PATH="/home/ubuntu/runs/dllm-dev/cnn-dm-block4-bs128-keep1-causalencfalse-max10000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen600m_v1"
# MODEL_PATH="Qwen/Qwen3-0.6B-Base"

DATASET="cnndm"  # "cnndm" or "wmt"
OUTPUT_DIR="outputs/${MODEL_PATH}"
mkdir -p ${OUTPUT_DIR}
L=256
BLOCK_SIZE=4
GREEDY=True # True, False
USE_X0_PRED=True
FIRST_HITTING=True
LOW_CONFIDENCE_REMASKING=True
KV_CACHING=True
TOP_P=1.0 # not used if greedy=True
MAX_LENGTH=1024
REPETITION_PENALTY=16 # set to >1 for CNN/DM!

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-greedy-${GREEDY}-use_x0_pred-${USE_X0_PRED}-first_hitting-${FIRST_HITTING}-low_confidence_remasking-${LOW_CONFIDENCE_REMASKING}"
NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
PORT=29501

torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/seq2seq_eval.py \
  --dataset ${DATASET} \
  --max_new_tokens ${L} \
  --model_path ${MODEL_PATH} \
  --output_path ${OUTPUT_PATH} \
  --ckpt_file best-rank0.pt \
  --tokenizer_name_or_path Qwen/Qwen3-0.6B-Base \
  --block_size 4 \
  --greedy ${GREEDY} \
  --use_x0_pred ${USE_X0_PRED} \
  --first_hitting ${FIRST_HITTING} \
  --low_confidence_remasking ${LOW_CONFIDENCE_REMASKING} \
  --kv_caching ${KV_CACHING} \
  --top_p ${TOP_P} \
  --max_length ${MAX_LENGTH} \
  --shift_logits True \
  --repetition_penalty ${REPETITION_PENALTY}
