#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"
source "${REPO_ROOT}/bash_scripts/eval_model_paths.sh"

COMPILE_BACKBONE="${COMPILE_BACKBONE:-false}"
COMPILE_MODE="${COMPILE_MODE:-}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-}"
THROUGHPUT_RUN="${THROUGHPUT_RUN:-false}"
THROUGHPUT_WARMUP="${THROUGHPUT_WARMUP:-50}"
THROUGHPUT_MEASUREMENTS="${THROUGHPUT_MEASUREMENTS:-1000}"
THROUGHPUT_GLOBAL_MEASUREMENTS="${THROUGHPUT_GLOBAL_MEASUREMENTS:-true}"
OUTPUT_PATH_OVERRIDE="${OUTPUT_PATH_OVERRIDE:-}"

resolve_eval_model_path "${MODEL_PATH:-${EVAL_MODEL_KEY:-}}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
MAX_WINDOW_SIZE="${MAX_WINDOW_SIZE:-${BLOCK_SIZE}}"
KV_CACHING="${KV_CACHING:-true}"
ALIGN_INPUTS_TO_BLOCKS="${ALIGN_INPUTS_TO_BLOCKS:-true}"
USE_FIRST_HITTING_ORDER_IN_DECODE="${USE_FIRST_HITTING_ORDER_IN_DECODE:-false}"
SETDLM_DECODE_DIAGNOSTIC_LOG="${SETDLM_DECODE_DIAGNOSTIC_LOG:-false}"
SETDLM_DECODE_DIAGNOSTIC_MAX_STEPS="${SETDLM_DECODE_DIAGNOSTIC_MAX_STEPS:-8}"
SETDLM_DECODE_ORDER_TRACE="${SETDLM_DECODE_ORDER_TRACE:-false}"
SETDLM_DECODE_ORDER_TRACE_MAX_STEPS="${SETDLM_DECODE_ORDER_TRACE_MAX_STEPS:-8}"
SETDLM_DECODE_SNAPSHOT_LOG="${SETDLM_DECODE_SNAPSHOT_LOG:-false}"
SETDLM_DECODE_SNAPSHOT_MAX_EXAMPLES="${SETDLM_DECODE_SNAPSHOT_MAX_EXAMPLES:-4}"
SETDLM_DECODE_SNAPSHOT_MAX_SNAPSHOTS="${SETDLM_DECODE_SNAPSHOT_MAX_SNAPSHOTS:-96}"
SETDLM_DECODE_SNAPSHOT_TAIL_TOKENS="${SETDLM_DECODE_SNAPSHOT_TAIL_TOKENS:-64}"
SETDLM_DECODE_SNAPSHOT_MAX_DECODE_TOKENS="${SETDLM_DECODE_SNAPSHOT_MAX_DECODE_TOKENS:-96}"
SETDLM_L2R_EOS_FRONTIER_CONSTRAINT="${SETDLM_L2R_EOS_FRONTIER_CONSTRAINT:-false}"
CNNDM_GENERATE_TARGET_PROMPT="${CNNDM_GENERATE_TARGET_PROMPT:-false}"
CNNDM_DIAGNOSTIC_LOG="${CNNDM_DIAGNOSTIC_LOG:-false}"
CNNDM_DISABLE_STOPPING_CRITERIA="${CNNDM_DISABLE_STOPPING_CRITERIA:-false}"

OUTPUT_DIR="outputs/${MODEL_PATH}/cnn_dailymail_t1"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L="${L:-null}"
T="${T:-${BLOCK_SIZE}}"
DO_SAMPLE="${DO_SAMPLE:-false}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-predict_and_noise}"  # "predict_and_noise" "posterior"
FIRST_HITTING="${FIRST_HITTING:-false}"
CONFIDENCE_BASED_NOISING="${CONFIDENCE_BASED_NOISING:-false}"
CONFIDENCE_MARGIN_BASED_NOISING="${CONFIDENCE_MARGIN_BASED_NOISING:-false}"
CONF_THRESHOLD="${CONF_THRESHOLD:-1e6}"
MAX_LENGTH="${MAX_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-180}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-null}"
CKPT="${CKPT:-best}"
USE_EMA="${USE_EMA:-true}"
LEN_PENALTY="${LEN_PENALTY:-1.1}"
REGULATION_START="${REGULATION_START:-80}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"

echo "MODEL_PATH: ${MODEL_PATH}"
echo "BLOCK_SIZE: ${BLOCK_SIZE}"
echo "KV_CACHING: ${KV_CACHING}"
echo "ALIGN_INPUTS_TO_BLOCKS: ${ALIGN_INPUTS_TO_BLOCKS}"
echo "MAX_WINDOW_SIZE: ${MAX_WINDOW_SIZE}"
echo "LEN_PENALTY: ${LEN_PENALTY}"
echo "USE_FIRST_HITTING_ORDER_IN_DECODE: ${USE_FIRST_HITTING_ORDER_IN_DECODE}"
echo "SETDLM_DECODE_DIAGNOSTIC_LOG: ${SETDLM_DECODE_DIAGNOSTIC_LOG}"
echo "SETDLM_DECODE_DIAGNOSTIC_MAX_STEPS: ${SETDLM_DECODE_DIAGNOSTIC_MAX_STEPS}"
echo "SETDLM_DECODE_ORDER_TRACE: ${SETDLM_DECODE_ORDER_TRACE}"
echo "SETDLM_DECODE_ORDER_TRACE_MAX_STEPS: ${SETDLM_DECODE_ORDER_TRACE_MAX_STEPS}"
echo "SETDLM_DECODE_SNAPSHOT_LOG: ${SETDLM_DECODE_SNAPSHOT_LOG}"
echo "SETDLM_DECODE_SNAPSHOT_MAX_EXAMPLES: ${SETDLM_DECODE_SNAPSHOT_MAX_EXAMPLES}"
echo "SETDLM_DECODE_SNAPSHOT_MAX_SNAPSHOTS: ${SETDLM_DECODE_SNAPSHOT_MAX_SNAPSHOTS}"
echo "SETDLM_DECODE_SNAPSHOT_TAIL_TOKENS: ${SETDLM_DECODE_SNAPSHOT_TAIL_TOKENS}"
echo "SETDLM_DECODE_SNAPSHOT_MAX_DECODE_TOKENS: ${SETDLM_DECODE_SNAPSHOT_MAX_DECODE_TOKENS}"
echo "SETDLM_L2R_EOS_FRONTIER_CONSTRAINT: ${SETDLM_L2R_EOS_FRONTIER_CONSTRAINT}"
echo "CNNDM_GENERATE_TARGET_PROMPT: ${CNNDM_GENERATE_TARGET_PROMPT}"
echo "CNNDM_DIAGNOSTIC_LOG: ${CNNDM_DIAGNOSTIC_LOG}"
echo "CNNDM_DISABLE_STOPPING_CRITERIA: ${CNNDM_DISABLE_STOPPING_CRITERIA}"
echo "REGULATION_START: ${REGULATION_START}"
echo "REPETITION_PENALTY: ${REPETITION_PENALTY}"
echo "CONF_THRESHOLD: ${CONF_THRESHOLD}"
echo "MAX_LENGTH: ${MAX_LENGTH}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "MAX_EVAL_SAMPLES: ${MAX_EVAL_SAMPLES}"
echo "CKPT: ${CKPT}"
echo "USE_EMA: ${USE_EMA}"
echo "COMPILE_BACKBONE: ${COMPILE_BACKBONE}"
echo "COMPILE_MODE: ${COMPILE_MODE:-none}"
echo "COMPILE_DYNAMIC: ${COMPILE_DYNAMIC:-false}"
echo "THROUGHPUT_RUN: ${THROUGHPUT_RUN}"
echo "THROUGHPUT_WARMUP: ${THROUGHPUT_WARMUP}"
echo "THROUGHPUT_MEASUREMENTS: ${THROUGHPUT_MEASUREMENTS}"
echo "THROUGHPUT_GLOBAL_MEASUREMENTS: ${THROUGHPUT_GLOBAL_MEASUREMENTS}"
echo "OUTPUT_PATH_OVERRIDE: ${OUTPUT_PATH_OVERRIDE}"

DECODE_ORDER_SUFFIX=""
if [[ "${USE_FIRST_HITTING_ORDER_IN_DECODE}" == "true" ]]; then
  DECODE_ORDER_SUFFIX="_decode-fhtrue"
fi
EOS_FRONTIER_SUFFIX=""
if [[ "${SETDLM_L2R_EOS_FRONTIER_CONSTRAINT}" == "true" ]]; then
  EOS_FRONTIER_SUFFIX="_eos-frontiertrue"
fi
TARGET_PROMPT_SUFFIX=""
if [[ "${CNNDM_GENERATE_TARGET_PROMPT}" == "true" ]]; then
  TARGET_PROMPT_SUFFIX="_gen-target-prompttrue"
fi
SNAPSHOT_SUFFIX=""
if [[ "${SETDLM_DECODE_SNAPSHOT_LOG}" == "true" ]]; then
  SNAPSHOT_SUFFIX="_snapshots"
fi
NO_STOP_SUFFIX=""
if [[ "${CNNDM_DISABLE_STOPPING_CRITERIA}" == "true" ]]; then
  NO_STOP_SUFFIX="_no-stoptrue"
fi
DEFAULT_OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}rep-penalty-${REPETITION_PENALTY}_len-penalty-${LEN_PENALTY}_reg-start${REGULATION_START}${DECODE_ORDER_SUFFIX}${EOS_FRONTIER_SUFFIX}${TARGET_PROMPT_SUFFIX}${SNAPSHOT_SUFFIX}${NO_STOP_SUFFIX}"
OUTPUT_PATH="${OUTPUT_PATH:-${DEFAULT_OUTPUT_PATH}}"
if [[ -n "${OUTPUT_PATH_OVERRIDE}" ]]; then
  OUTPUT_PATH="${OUTPUT_PATH_OVERRIDE}"
fi
mkdir -p ${OUTPUT_PATH}

STOPPING_ARGS=()
if [[ "${CNNDM_DISABLE_STOPPING_CRITERIA}" == "true" ]]; then
  STOPPING_ARGS+=(~generation/stopping_criteria@stopping_criteria_list)
  STOPPING_ARGS+=(gen_kwargs.stopping_criteria=null)
else
  STOPPING_ARGS+=(generation/stopping_criteria@stopping_criteria_list='[cnndm_stop_string_criteria]')
fi

LOGITS_PROCESSOR_ARGS=(
  generation/logits_processor@logits_processor_list="${LOGITS_PROCESSOR_NAMES}"
  logits_processor_list.repetition_penalty_logits_processor.penalty=${REPETITION_PENALTY}
  logits_processor_list.exponential_decay_length_penalty.exponential_decay_length_penalty="[${REGULATION_START},${LEN_PENALTY}]"
)
EXTRA_ARGS=(
  +compile_backbone=${COMPILE_BACKBONE}
  +throughput_run=${THROUGHPUT_RUN}
  +throughput_warmup=${THROUGHPUT_WARMUP}
  +throughput_num_measurements=${THROUGHPUT_MEASUREMENTS}
  +throughput_global_measurements=${THROUGHPUT_GLOBAL_MEASUREMENTS}
)
if [[ -n "${COMPILE_MODE}" ]]; then
  EXTRA_ARGS+=(+compile_mode=${COMPILE_MODE})
fi
if [[ -n "${COMPILE_DYNAMIC}" ]]; then
  EXTRA_ARGS+=(+compile_dynamic=${COMPILE_DYNAMIC})
fi

PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=cnn_dailymail \
  +task.dataset.cache_path="" \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file="${CKPT}-rank0.pt" \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B-Base" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${MAX_NEW_TOKENS} \
  +max_eval_samples=${MAX_EVAL_SAMPLES} \
  block_size=${BLOCK_SIZE} \
  generation@generation_config=set_diffusion_generation_config \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=${CONF_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation_config.max_window_size=${MAX_WINDOW_SIZE} \
  ++generation_config.use_first_hitting_order_in_decode=${USE_FIRST_HITTING_ORDER_IN_DECODE} \
  ++generation_config.setdlm_decode_diagnostic_log=${SETDLM_DECODE_DIAGNOSTIC_LOG} \
  ++generation_config.setdlm_decode_diagnostic_max_steps=${SETDLM_DECODE_DIAGNOSTIC_MAX_STEPS} \
  ++generation_config.setdlm_decode_order_trace=${SETDLM_DECODE_ORDER_TRACE} \
  ++generation_config.setdlm_decode_order_trace_max_steps=${SETDLM_DECODE_ORDER_TRACE_MAX_STEPS} \
  ++generation_config.setdlm_decode_snapshot_log=${SETDLM_DECODE_SNAPSHOT_LOG} \
  ++generation_config.setdlm_decode_snapshot_max_examples=${SETDLM_DECODE_SNAPSHOT_MAX_EXAMPLES} \
  ++generation_config.setdlm_decode_snapshot_max_snapshots=${SETDLM_DECODE_SNAPSHOT_MAX_SNAPSHOTS} \
  ++generation_config.setdlm_decode_snapshot_tail_tokens=${SETDLM_DECODE_SNAPSHOT_TAIL_TOKENS} \
  ++generation_config.setdlm_decode_snapshot_max_decode_tokens=${SETDLM_DECODE_SNAPSHOT_MAX_DECODE_TOKENS} \
  ++generation_config.setdlm_l2r_eos_frontier_constraint=${SETDLM_L2R_EOS_FRONTIER_CONSTRAINT} \
  +cnndm_generate_target_prompt=${CNNDM_GENERATE_TARGET_PROMPT} \
  +cnndm_diagnostic_log=${CNNDM_DIAGNOSTIC_LOG} \
  "${STOPPING_ARGS[@]}" \
  "${LOGITS_PROCESSOR_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
