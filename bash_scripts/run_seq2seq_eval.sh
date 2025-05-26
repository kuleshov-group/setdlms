#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

MODEL_PATH="${RUN_DIR}/cnn-dm-block4-bs128-keep1-causalencfalse-max10000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen600m_v1"

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
REPETITION_PENALTY=1.2 # set to >1 for CNN/DM!
LEN_PENALTY=1.1
REGULATION_START=30

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-greedy-${GREEDY}-use_x0_pred-${USE_X0_PRED}-first_hitting-${FIRST_HITTING}-low_confidence_remasking-${LOW_CONFIDENCE_REMASKING}_rep-penalty-${REPETITION_PENALTY}_len-penalty-${LEN_PENALTY}_reg-start${REGULATION_START}"
NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
PORT=29501

torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
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
  --repetition_penalty ${REPETITION_PENALTY} \
  --length_penalty ${LEN_PENALTY} \
  --regulation_start ${REGULATION_START}
