#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"
source "${REPO_ROOT}/bash_scripts/eval_model_paths.sh"

QWEN_MODEL="Qwen/Qwen3-1.7B-Base"
NUM_FEW_SHOT=0

resolve_eval_model_path "${MODEL_PATH:-${EVAL_MODEL_KEY:-}}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_WINDOW_SIZE="${MAX_WINDOW_SIZE:-${BLOCK_SIZE}}"
ALIGN_INPUTS_TO_BLOCKS="${ALIGN_INPUTS_TO_BLOCKS:-true}"
T="${T:-${BLOCK_SIZE}}"

echo "MODEL_PATH: ${MODEL_PATH}"

KV_CACHING="${KV_CACHING:-true}"
USE_EMA=true
OUTPUT_DIR="outputs/${MODEL_PATH}/lm_eval_harness_output"
REVISION=null

L=1024
RETURN_DICT_IN_GENERATE=true
COMPUTE_INF_BUDGET=false
DO_SAMPLE="${DO_SAMPLE:-false}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-predict_and_noise}"  # "predict_and_noise" or "posterior"
FIRST_HITTING="${FIRST_HITTING:-false}"
CONFIDENCE_BASED_NOISING="${CONFIDENCE_BASED_NOISING:-false}"
CONFIDENCE_MARGIN_BASED_NOISING="${CONFIDENCE_MARGIN_BASED_NOISING:-false}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-1e6}"
CKPT="best"
LINEAR_UNMASKING=true
COMPILE_BACKBONE="${COMPILE_BACKBONE:-false}"
COMPILE_MODE="${COMPILE_MODE:-}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-}"
SETDLM_FAST_INFERENCE="${SETDLM_FAST_INFERENCE:-false}"
SETDLM_DYNAMIC_ACTIVE_LOGITS="${SETDLM_DYNAMIC_ACTIVE_LOGITS:-false}"
SETDLM_DETERMINISTIC_SAMPLER_FASTPATH="${SETDLM_DETERMINISTIC_SAMPLER_FASTPATH:-false}"
SETDLM_VECTORIZED_REPETITION_PENALTY="${SETDLM_VECTORIZED_REPETITION_PENALTY:-false}"
SETDLM_DYNAMIC_TENSOR_ATTENTION_MASK="${SETDLM_DYNAMIC_TENSOR_ATTENTION_MASK:-false}"
SETDLM_DYNAMIC_FULL_WINDOW_FASTPATH="${SETDLM_DYNAMIC_FULL_WINDOW_FASTPATH:-false}"
SETDLM_FULL_WINDOW_FASTPATH_LABEL="${SETDLM_FULL_WINDOW_FASTPATH_LABEL:-N/A}"
SETDLM_THROUGHPUT_RUN_NAME="${SETDLM_THROUGHPUT_RUN_NAME:-N/A}"

echo "MAX WINDOW SIZE ${MAX_WINDOW_SIZE}"

THROUGHPUT_WARMUP="${THROUGHPUT_WARMUP:-100}"
THROUGHPUT_MEASUREMENTS="${THROUGHPUT_MEASUREMENTS:-100}"
THROUGHPUT_GLOBAL_MEASUREMENTS="${THROUGHPUT_GLOBAL_MEASUREMENTS:-false}"

echo "CONFIDENCE_THRESHOLD: ${CONFIDENCE_THRESHOLD} T: ${T} LINEAR_UNMASKING: ${LINEAR_UNMASKING} DO_SAMPLE: ${DO_SAMPLE} SAMPLING_STRATEGY: ${SAMPLING_STRATEGY} FIRST_HITTING: ${FIRST_HITTING} CONFIDENCE_BASED_NOISING: ${CONFIDENCE_BASED_NOISING} CONFIDENCE_MARGIN_BASED_NOISING: ${CONFIDENCE_MARGIN_BASED_NOISING} ALIGN_INPUTS_TO_BLOCKS: ${ALIGN_INPUTS_TO_BLOCKS} CKPT: ${CKPT} USE_EMA: ${USE_EMA}"
echo "THROUGHPUT_WARMUP: ${THROUGHPUT_WARMUP} THROUGHPUT_MEASUREMENTS: ${THROUGHPUT_MEASUREMENTS} THROUGHPUT_GLOBAL_MEASUREMENTS: ${THROUGHPUT_GLOBAL_MEASUREMENTS}"
echo "COMPILE_BACKBONE: ${COMPILE_BACKBONE} COMPILE_MODE: ${COMPILE_MODE:-auto} COMPILE_DYNAMIC: ${COMPILE_DYNAMIC:-auto}"
echo "SETDLM_FAST_INFERENCE: ${SETDLM_FAST_INFERENCE} SETDLM_DYNAMIC_ACTIVE_LOGITS: ${SETDLM_DYNAMIC_ACTIVE_LOGITS} SETDLM_DETERMINISTIC_SAMPLER_FASTPATH: ${SETDLM_DETERMINISTIC_SAMPLER_FASTPATH} SETDLM_VECTORIZED_REPETITION_PENALTY: ${SETDLM_VECTORIZED_REPETITION_PENALTY} SETDLM_DYNAMIC_TENSOR_ATTENTION_MASK: ${SETDLM_DYNAMIC_TENSOR_ATTENTION_MASK} SETDLM_DYNAMIC_FULL_WINDOW_FASTPATH: ${SETDLM_DYNAMIC_FULL_WINDOW_FASTPATH} SETDLM_FULL_WINDOW_FASTPATH_LABEL: ${SETDLM_FULL_WINDOW_FASTPATH_LABEL} SETDLM_THROUGHPUT_RUN_NAME: ${SETDLM_THROUGHPUT_RUN_NAME}"

OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_${NUM_FEW_SHOT}shot_L${L}_block${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hit${FIRST_HITTING}-conf_noise${CONFIDENCE_BASED_NOISING}-conf_margin_noise${CONFIDENCE_MARGIN_BASED_NOISING}-conf_thold${CONFIDENCE_THRESHOLD}-align_to_blocks${ALIGN_INPUTS_TO_BLOCKS}"
if [ -n "${OUTPUT_PATH_OVERRIDE:-}" ]; then
  OUTPUT_PATH="${OUTPUT_PATH_OVERRIDE}"
fi
mkdir -p ${OUTPUT_PATH}

MODEL_ARGS=()
COMPILE_ARGS=(+task.model.compile_backbone=${COMPILE_BACKBONE})
if [ -n "${COMPILE_MODE}" ]; then
  COMPILE_ARGS+=(+task.model.compile_mode=${COMPILE_MODE})
fi
if [ -n "${COMPILE_DYNAMIC}" ]; then
  COMPILE_ARGS+=(+task.model.compile_dynamic=${COMPILE_DYNAMIC})
fi

if [ -n "${MASTER_PORT:-}" ]; then
  PORT="${MASTER_PORT}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  PORT=$((20000 + SLURM_JOB_ID % 40000))
else
  PORT=$((RANDOM % 10000 + 29500))
fi
export MASTER_PORT="${PORT}"
echo "MASTER_PORT: ${PORT}"
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=gsm8k \
  task.num_fewshot=${NUM_FEW_SHOT} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  task.model.ckpt_file="${CKPT}-rank0.pt" \
  task.model.load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path=${QWEN_MODEL} \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${L} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  "${MODEL_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  generation@generation_config=set_diffusion_generation_config \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.num_steps=${T} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation_config.max_window_size=${MAX_WINDOW_SIZE} \
  generation_config.setdlm_fast_inference=${SETDLM_FAST_INFERENCE} \
  generation_config.setdlm_dynamic_active_logits=${SETDLM_DYNAMIC_ACTIVE_LOGITS} \
  generation_config.setdlm_deterministic_sampler_fastpath=${SETDLM_DETERMINISTIC_SAMPLER_FASTPATH} \
  generation_config.setdlm_vectorized_repetition_penalty=${SETDLM_VECTORIZED_REPETITION_PENALTY} \
  generation_config.setdlm_dynamic_tensor_attention_mask=${SETDLM_DYNAMIC_TENSOR_ATTENTION_MASK} \
  generation_config.setdlm_dynamic_full_window_fastpath=${SETDLM_DYNAMIC_FULL_WINDOW_FASTPATH} \
  gen_kwargs.return_dict_in_generate=${RETURN_DICT_IN_GENERATE} \
  generation_config.compute_inf_budget=${COMPUTE_INF_BUDGET} \
  generation_config.linear_unmasking=${LINEAR_UNMASKING} \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  generation/stopping_criteria@stopping_criteria_list='[gsm8k_regex_stopping_criteria,repeating_token]' \
  +task.model.throughput_run=true \
  +task.model.throughput_samples=${THROUGHPUT_MEASUREMENTS} \
  +task.model.throughput_warmup=${THROUGHPUT_WARMUP} \
  +task.model.throughput_global_measurements=${THROUGHPUT_GLOBAL_MEASUREMENTS}
