#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# SetDLM s <= 8
MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block4_vscratch"
CKPT_FILE="ep17-ba300000-rank0.pt"
BLOCK_SIZE=1024
COMPILE_BACKBONE=true
USE_EMA=false

# SetDLM s <= 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block8_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=1024
# COMPILE_BACKBONE=true
# USE_EMA=false

# SetDLM s <= 32
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block16_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=1024
# COMPILE_BACKBONE=true
# USE_EMA=false

# BD3LM s = 4
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block4_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=4
# COMPILE_BACKBONE=true
# USE_EMA=false

# BD3LM s = 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block8_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=8
# COMPILE_BACKBONE=true
# USE_EMA=false

# BD3LM s = 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block16_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=16
# COMPILE_BACKBONE=true
# USE_EMA=false

# AR
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-ar-noeos-v4-1"
# CKPT_FILE="20-300000.ckpt"
# MODEL_PATH="${MODEL_PATH}/${CKPT_FILE}"
# BLOCK_SIZE=1
# COMPILE_BACKBONE=true
# USE_EMA=true

# MDLM
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-mdlm-noeos-v4"
# CKPT_FILE="18-300000.ckpt"
# MODEL_PATH="${MODEL_PATH}/${CKPT_FILE}"
# BLOCK_SIZE=1024
# COMPILE_BACKBONE=true
# USE_EMA=true

TOKENIZER_PATH="gpt2"
REVISION=null
MAX_LENGTH=1024
MAX_EXAMPLES=null
TEST_MODE=false
TEST_NUM_EXAMPLES_PER_BENCHMARK=32
TEST_SHUFFLE=false
NUM_IMPORTANCE_SAMPLES=1
NORMALIZE_BY_ANSWER_LENGTH=true
REQUIRE_REFUSION_SEMANTICS=false
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
echo "REQUIRE_REFUSION_SEMANTICS: ${REQUIRE_REFUSION_SEMANTICS}"

OUTPUT_DIR="outputs/${MODEL_PATH}/mcqa_eval_output"
if [ "${TEST_MODE}" = true ]; then
  MAX_EXAMPLES=null
  TEST_SUFFIX="_test-${TEST_NUM_EXAMPLES_PER_BENCHMARK}"
else
  TEST_SUFFIX=""
fi
OUTPUT_PATH="${OUTPUT_DIR}/task-${TASK_CONFIG}_ckpt-${CKPT_FILE}_ema-${USE_EMA}_block-${BLOCK_SIZE}_maxlen-${MAX_LENGTH}_maxexamples-${MAX_EXAMPLES}${TEST_SUFFIX}_is-${NUM_IMPORTANCE_SAMPLES}"
mkdir -p "${OUTPUT_PATH}"

REFUSION_ARGS=()
if [ "${REQUIRE_REFUSION_SEMANTICS}" = true ]; then
  REFUSION_ARGS+=(
    task.require_refusion_semantics=true
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
  "${REFUSION_ARGS[@]}" \
  +compile_backbone=${COMPILE_BACKBONE} \
  +model_config_overrides.backbone_config.attn_backend=sdpa \
  +model_config_overrides.attn_backend=sdpa \
  ~generation@generation_config \
  ~generation/logits_processor@logits_processor_list \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs=null
