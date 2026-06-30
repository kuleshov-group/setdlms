#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"
source "${REPO_ROOT}/bash_scripts/eval_model_paths.sh"

resolve_eval_model_path "${MODEL_PATH:-${EVAL_MODEL_KEY:-}}"
CKPT_FILE="${CKPT_FILE:-best-rank0.pt}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
COMPILE_BACKBONE="${COMPILE_BACKBONE:-true}"
USE_EMA="${USE_EMA:-false}"

TOKENIZER_PATH="gpt2"
REVISION=null
MAX_LENGTH=1024
MAX_EXAMPLES=null
TEST_MODE=false
TEST_NUM_EXAMPLES_PER_BENCHMARK=32
TEST_SHUFFLE=false
NUM_IMPORTANCE_SAMPLES=1
NORMALIZE_BY_ANSWER_LENGTH=true
TASK_CONFIG="all"  # all | hellaswag | piqa | siqa

echo "MODEL_PATH: ${MODEL_PATH}"
echo "TASK_CONFIG: ${TASK_CONFIG}"
echo "TOKENIZER_PATH: ${TOKENIZER_PATH}"
echo "MAX_EXAMPLES: ${MAX_EXAMPLES}"
echo "TEST_MODE: ${TEST_MODE}"
echo "TEST_NUM_EXAMPLES_PER_BENCHMARK: ${TEST_NUM_EXAMPLES_PER_BENCHMARK}"
echo "TEST_SHUFFLE: ${TEST_SHUFFLE}"
echo "NUM_IMPORTANCE_SAMPLES: ${NUM_IMPORTANCE_SAMPLES}"
echo "NORMALIZE_BY_ANSWER_LENGTH: ${NORMALIZE_BY_ANSWER_LENGTH}"

OUTPUT_DIR="outputs/${MODEL_PATH}/mcqa_eval_output"
if [ "${TEST_MODE}" = true ]; then
  MAX_EXAMPLES=null
  TEST_SUFFIX="_test-${TEST_NUM_EXAMPLES_PER_BENCHMARK}"
else
  TEST_SUFFIX=""
fi
OUTPUT_PATH="${OUTPUT_DIR}/task-${TASK_CONFIG}_ckpt-${CKPT_FILE}_ema-${USE_EMA}_block-${BLOCK_SIZE}_maxlen-${MAX_LENGTH}_maxexamples-${MAX_EXAMPLES}${TEST_SUFFIX}_is-${NUM_IMPORTANCE_SAMPLES}"
mkdir -p "${OUTPUT_PATH}"

  )
fi

PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/mcqa_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/mcqa@task=${TASK_CONFIG} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  tokenizer.pretrained_model_name_or_path=${TOKENIZER_PATH} \
  output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  block_size=${BLOCK_SIZE} \
  task.ckpt_file=${CKPT_FILE} \
  task.load_ema_weights=${USE_EMA} \
  task.max_examples=${MAX_EXAMPLES} \
  task.test_mode=${TEST_MODE} \
  task.test_num_examples_per_benchmark=${TEST_NUM_EXAMPLES_PER_BENCHMARK} \
  task.test_shuffle=${TEST_SHUFFLE} \
  task.num_importance_samples=${NUM_IMPORTANCE_SAMPLES} \
  task.normalize_by_answer_length=${NORMALIZE_BY_ANSWER_LENGTH} \
  +compile_backbone=${COMPILE_BACKBONE} \
  +model_config_overrides.backbone_config.attn_backend=sdpa \
  +model_config_overrides.attn_backend=sdpa \
  ~generation@generation_config \
  ~generation/logits_processor@logits_processor_list \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs=null
