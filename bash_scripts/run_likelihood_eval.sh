#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# MODEL_PATH="/share/kuleshov/yzs2/runs/dllm-dev/wmt-block4-bs128-keep1-causalencfalse-max10000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen600m_v1"
MODEL_PATH="/share/kuleshov/yzs2/runs/dllm-dev/cnn-dm-block4-bs128-keep1-causalencfalse-max10000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen600m_v1"
OUTPUT_DIR=${MODEL_PATH}/likelihood_eval # TODO unused

python scripts/eval/likelihood_eval.py \
  --model_path ${MODEL_PATH} \
  --output_path ${OUTPUT_DIR}
