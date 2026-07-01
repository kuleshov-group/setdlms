#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"
source "${REPO_ROOT}/bash_scripts/eval_model_paths.sh"

COMPILE_BACKBONE="${COMPILE_BACKBONE:-false}"
COMPILE_MODE="${COMPILE_MODE:-}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-}"
SEED="${SEED:-1234}"
export LM_EVAL_RANK_INVARIANT_SEED="${LM_EVAL_RANK_INVARIANT_SEED:-true}"
export LM_EVAL_BASE_SEED="${LM_EVAL_BASE_SEED:-${SEED}}"
SKIP_MAUVE="${SKIP_MAUVE:-false}"
EVAL_ONLY="${EVAL_ONLY:-false}"
GENERATION_CHECKPOINT="${GENERATION_CHECKPOINT:-false}"
RESUME_GENERATION_CHECKPOINT="${RESUME_GENERATION_CHECKPOINT:-${GENERATION_CHECKPOINT}}"
GENERATION_CHECKPOINT_INTERVAL="${GENERATION_CHECKPOINT_INTERVAL:-1}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-}"
NOISE_REMOVAL="${NOISE_REMOVAL:-}"

resolve_eval_model_path "${MODEL_PATH:-${EVAL_MODEL_KEY:-}}"

is_setdlm_model() {
  case "${MODEL_PATH:-} ${EVAL_MODEL_KEY:-}" in
    *SetDLM*|*setdlm*|*aoarm*) return 0 ;;
    *) return 1 ;;
  esac
}

infer_setdlm_desired_block_size() {
  case "${MODEL_PATH:-} ${EVAL_MODEL_KEY:-}" in
    *d4*|*tgt4*|*smax8*) echo 4 ;;
    *d8*|*tgt8*|*smax16*) echo 8 ;;
    *d16*|*tgt16*|*smax32*) echo 16 ;;
    *) echo "" ;;
  esac
}

SETDLM_PPL_MODEL_CONFIG_ARGS=()
if is_setdlm_model; then
  SETDLM_DESIRED_BLOCK_SIZE="${SETDLM_DESIRED_BLOCK_SIZE:-$(infer_setdlm_desired_block_size)}"
  if [ -z "${SETDLM_DESIRED_BLOCK_SIZE}" ]; then
    echo "ERROR: Could not infer SetDLM desired block size for MODEL_PATH=${MODEL_PATH:-}." >&2
    exit 1
  fi
  MAX_BLOCK_SIZE="${MAX_BLOCK_SIZE:-$((2 * SETDLM_DESIRED_BLOCK_SIZE))}"
  SETDLM_PPL_MODEL_CONFIG_ARGS+=(+model_config_overrides.noise_config.max_block_size=${MAX_BLOCK_SIZE})
fi

MODEL_FAMILY="${MODEL_FAMILY:-}"
CKPT_FILE="${CKPT_FILE:-best-rank0.pt}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
MAX_WINDOW_SIZE="${MAX_WINDOW_SIZE:-${BLOCK_SIZE}}"
ALIGN_INPUTS_TO_BLOCKS="${ALIGN_INPUTS_TO_BLOCKS:-false}"
KV_CACHING="${KV_CACHING:-true}"
USE_EMA="${USE_EMA:-false}"
NUCLEUS_P="${NUCLEUS_P:-${2:-1.0}}"
REPETITION_PENALTY="${REPETITION_PENALTY:-${1:-1.0}}"

MODEL_PATH_LOWER="$(printf '%s' "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
if [ -z "${MODEL_FAMILY}" ]; then
  if [[ "${MODEL_PATH_LOWER}" == *sedd* ]]; then
    MODEL_FAMILY="SEDD"
  else
    MODEL_FAMILY="AUTO"
  fi
fi

if [ "${MODEL_FAMILY}" = "SEDD" ]; then
  GENERATION_CONFIG_NAME="${GENERATION_CONFIG_NAME:-sedd_generation_config}"
  SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-analytic}"
  NOISE_REMOVAL="${NOISE_REMOVAL:-true}"
  ALIGN_INPUTS_TO_BLOCKS="${ALIGN_INPUTS_TO_BLOCKS:-false}"
else
  GENERATION_CONFIG_NAME="${GENERATION_CONFIG_NAME:-set_diffusion_generation_config}"
  SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-predict_and_noise}"
  NOISE_REMOVAL="${NOISE_REMOVAL:-false}"
fi

T=${GENERATION_NUM_STEPS:-${BLOCK_SIZE}}

REVISION=null
DO_SAMPLE=true
FIRST_HITTING="${FIRST_HITTING:-false}"
CONFIDENCE_BASED_NOISING="${CONFIDENCE_BASED_NOISING:-false}"
CONFIDENCE_MARGIN_BASED_NOISING=false
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-1e6}"
FUSED_BLOCK_CACHE="${FUSED_BLOCK_CACHE:-auto}"
SETDLM_THROUGHPUT_RUN_NAME="${SETDLM_THROUGHPUT_RUN_NAME:-N/A}"

MAX_LENGTH=${MAX_LENGTH:-1024}
THROUGHPUT_RUN=${THROUGHPUT_RUN:-false}
THROUGHPUT_SAMPLES_PER_RANK=${THROUGHPUT_SAMPLES_PER_RANK:-200}
THROUGHPUT_WARMUP=${THROUGHPUT_WARMUP:-0}
THROUGHPUT_MEASUREMENTS=${THROUGHPUT_MEASUREMENTS:-${THROUGHPUT_SAMPLES_PER_RANK}}
THROUGHPUT_GLOBAL_MEASUREMENTS=${THROUGHPUT_GLOBAL_MEASUREMENTS:-false}
REQUESTED_NUM_SAMPLES=${NUM_SAMPLES:-1000}
NUM_SAMPLES=${REQUESTED_NUM_SAMPLES}
if [ "${THROUGHPUT_RUN}" = "true" ]; then
  NUM_SAMPLES=${THROUGHPUT_SAMPLES_PER_RANK}
fi
OUTPUT_NUM_SAMPLES=${OUTPUT_NUM_SAMPLES:-${REQUESTED_NUM_SAMPLES}}
MAUVE_REFERENCE_NUM_SAMPLES=${MAUVE_REFERENCE_NUM_SAMPLES:-${OUTPUT_NUM_SAMPLES}}
STOPPING_CONFIDENCE_THRESHOLD=${STOPPING_CONFIDENCE_THRESHOLD:-0.005}
STOPPING_CONFIDENCE_WINDOW=${STOPPING_CONFIDENCE_WINDOW:-128}
STOPPING_CONFIDENCE_MIN_TOKENS=${STOPPING_CONFIDENCE_MIN_TOKENS:-128}
STOPPING_CONFIDENCE_PATIENCE=${STOPPING_CONFIDENCE_PATIENCE:-4}
LOW_ENTROPY_TRUNCATION=${LOW_ENTROPY_TRUNCATION:-legacy}
TOKENIZER_PATH="${TOKENIZER_PATH:-gpt2}"
OUTPUT_DATASET_NAME="${OUTPUT_DATASET_NAME:-owt}"
echo "MODEL_PATH: ${MODEL_PATH} MODEL_FAMILY: ${MODEL_FAMILY} GENERATION_CONFIG: ${GENERATION_CONFIG_NAME} BLOCK_SIZE: ${BLOCK_SIZE} MAX_WINDOW_SIZE: ${MAX_WINDOW_SIZE} T: ${T} SAMPLING_STRATEGY: ${SAMPLING_STRATEGY} NOISE_REMOVAL: ${NOISE_REMOVAL} NUCLEUS_P: ${NUCLEUS_P} REPETITION_PENALTY: ${REPETITION_PENALTY} THROUGHPUT_RUN: ${THROUGHPUT_RUN} THROUGHPUT_WARMUP: ${THROUGHPUT_WARMUP} THROUGHPUT_MEASUREMENTS: ${THROUGHPUT_MEASUREMENTS} THROUGHPUT_GLOBAL_MEASUREMENTS: ${THROUGHPUT_GLOBAL_MEASUREMENTS} STOPPING_CONFIDENCE_THRESHOLD: ${STOPPING_CONFIDENCE_THRESHOLD} STOPPING_CONFIDENCE_MIN_TOKENS: ${STOPPING_CONFIDENCE_MIN_TOKENS} STOPPING_CONFIDENCE_PATIENCE: ${STOPPING_CONFIDENCE_PATIENCE} LOW_ENTROPY_TRUNCATION: ${LOW_ENTROPY_TRUNCATION}"
echo "SEED: ${SEED} LM_EVAL_RANK_INVARIANT_SEED: ${LM_EVAL_RANK_INVARIANT_SEED} LM_EVAL_BASE_SEED: ${LM_EVAL_BASE_SEED} SKIP_MAUVE: ${SKIP_MAUVE} EVAL_ONLY: ${EVAL_ONLY} GENERATION_CHECKPOINT: ${GENERATION_CHECKPOINT} RESUME_GENERATION_CHECKPOINT: ${RESUME_GENERATION_CHECKPOINT} GENERATION_CHECKPOINT_INTERVAL: ${GENERATION_CHECKPOINT_INTERVAL}"
echo "TOKENIZER_PATH: ${TOKENIZER_PATH} OUTPUT_DATASET_NAME: ${OUTPUT_DATASET_NAME} MAX_LENGTH: ${MAX_LENGTH} OUTPUT_NUM_SAMPLES: ${OUTPUT_NUM_SAMPLES}"
echo "COMPILE_BACKBONE: ${COMPILE_BACKBONE} COMPILE_MODE: ${COMPILE_MODE:-auto} COMPILE_DYNAMIC: ${COMPILE_DYNAMIC:-auto}"
echo "CONFIDENCE_BASED_NOISING: ${CONFIDENCE_BASED_NOISING} CONFIDENCE_THRESHOLD: ${CONFIDENCE_THRESHOLD} FIRST_HITTING: ${FIRST_HITTING} ALIGN_INPUTS_TO_BLOCKS: ${ALIGN_INPUTS_TO_BLOCKS}"
if [ "${FUSED_BLOCK_CACHE}" = "auto" ]; then
  FUSED_BLOCK_CACHE_CONFIG=null
else
  FUSED_BLOCK_CACHE_CONFIG=${FUSED_BLOCK_CACHE}
fi
echo "FUSED_BLOCK_CACHE: ${FUSED_BLOCK_CACHE} SETDLM_THROUGHPUT_RUN_NAME: ${SETDLM_THROUGHPUT_RUN_NAME}"

OUTPUT_DIR="outputs/${MODEL_PATH}/${OUTPUT_DATASET_NAME}-L-${MAX_LENGTH}-NUM_SAMPLES${OUTPUT_NUM_SAMPLES}"
mkdir -p ${OUTPUT_DIR}
OUTPUT_PATH="${OUTPUT_DIR}/block_size-${BLOCK_SIZE}-T${T}-sampling_strategy-${SAMPLING_STRATEGY}-noise_removal-${NOISE_REMOVAL}-do_sample-${DO_SAMPLE}-first_hitting-${FIRST_HITTING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT_FILE}-ema${USE_EMA}-nucleus_p${NUCLEUS_P}-repetition_penalty${REPETITION_PENALTY}-conf${CONFIDENCE_THRESHOLD}-max_window_size${MAX_WINDOW_SIZE}"
if [ "${STOPPING_CONFIDENCE_THRESHOLD}" != "null" ]; then
  OUTPUT_PATH="${OUTPUT_PATH}-stop_conf${STOPPING_CONFIDENCE_THRESHOLD}-stop_win${STOPPING_CONFIDENCE_WINDOW}-stop_min${STOPPING_CONFIDENCE_MIN_TOKENS}-stop_patience${STOPPING_CONFIDENCE_PATIENCE}"
fi
if [ "${THROUGHPUT_RUN}" = "true" ]; then
  OUTPUT_PATH="${OUTPUT_PATH}-throughput_run"
fi
if [ "${LOW_ENTROPY_TRUNCATION}" != "legacy" ]; then
  OUTPUT_PATH="${OUTPUT_PATH}-low_entropy_${LOW_ENTROPY_TRUNCATION}"
fi
if is_setdlm_model; then
  OUTPUT_PATH="${OUTPUT_PATH}-noise_max_block_size${MAX_BLOCK_SIZE}"
fi
OUTPUT_PATH="${OUTPUT_PATH_OVERRIDE:-${OUTPUT_PATH}}"

TORCHRUN_ARGS=(
  hydra.output_subdir=null
  hydra.run.dir="${PWD}"
  hydra/job_logging=disabled
  hydra/hydra_logging=disabled
  seed=${SEED}
  pretrained_model_name_or_path=${MODEL_PATH}
  pretrained_model_revision=${REVISION}
  +ckpt_file="${CKPT_FILE}"
  +load_ema_weights=${USE_EMA}
  tokenizer.pretrained_model_name_or_path=${TOKENIZER_PATH}
  output_path=${OUTPUT_PATH}
  generated_samples_output_path=${OUTPUT_PATH}
  max_length=${MAX_LENGTH}
  max_new_tokens=$((${MAX_LENGTH} - 1))
  block_size=${BLOCK_SIZE}
  generation@generation_config=${GENERATION_CONFIG_NAME}
  generation_config.num_steps=${T}
  generation_config.sampling_strategy=${SAMPLING_STRATEGY}
  generation_config.noise_removal=${NOISE_REMOVAL}
  generation_config.do_sample=${DO_SAMPLE}
  generation_config.first_hitting=${FIRST_HITTING}
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING}
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING}
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD}
  generation_config.use_cache=${KV_CACHING}
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS}
  generation_config.linear_unmasking=true
  generation_config.nucleus_p=${NUCLEUS_P}
  generation/stopping_criteria@stopping_criteria_list='[entropy_eos_stopping_criteria]'
  stopping_criteria_list.entropy_eos_stopping_criteria.confidence_threshold=${STOPPING_CONFIDENCE_THRESHOLD}
  stopping_criteria_list.entropy_eos_stopping_criteria.confidence_window=${STOPPING_CONFIDENCE_WINDOW}
  stopping_criteria_list.entropy_eos_stopping_criteria.confidence_min_tokens=${STOPPING_CONFIDENCE_MIN_TOKENS}
  stopping_criteria_list.entropy_eos_stopping_criteria.confidence_patience=${STOPPING_CONFIDENCE_PATIENCE}
  stopping_criteria_list.entropy_eos_stopping_criteria.low_entropy_truncation=${LOW_ENTROPY_TRUNCATION}
  batch_size=1
  +skip_mauve=${SKIP_MAUVE}
  +eval_only=${EVAL_ONLY}
  +generation_checkpoint=${GENERATION_CHECKPOINT}
  +resume_generation_checkpoint=${RESUME_GENERATION_CHECKPOINT}
  +generation_checkpoint_interval=${GENERATION_CHECKPOINT_INTERVAL}
  +throughput_run=${THROUGHPUT_RUN}
  +throughput_samples_per_rank=${THROUGHPUT_SAMPLES_PER_RANK}
  +throughput_warmup=${THROUGHPUT_WARMUP}
  +throughput_num_measurements=${THROUGHPUT_MEASUREMENTS}
  +throughput_global_measurements=${THROUGHPUT_GLOBAL_MEASUREMENTS}
  +diagnostic_stop_reasons=${DIAGNOSTIC_STOP_REASONS:-false}
  +model_config_overrides.attn_backend=sdpa
  +model_config_overrides.block_size=${BLOCK_SIZE}
  +model_config_overrides.backbone_config.attn_backend=sdpa
  "${SETDLM_PPL_MODEL_CONFIG_ARGS[@]}"
  generation/logits_processor@logits_processor_list='[repetition_penalty_logits_processor]'
  logits_processor_list.repetition_penalty_logits_processor.penalty=${REPETITION_PENALTY}
  +compile_backbone=${COMPILE_BACKBONE}
  num_samples=${NUM_SAMPLES}
  mauve_reference_num_samples=${MAUVE_REFERENCE_NUM_SAMPLES}
)

if [ "${GENERATION_CONFIG_NAME}" = "set_diffusion_generation_config" ]; then
  TORCHRUN_ARGS+=(
    generation_config.max_window_size=${MAX_WINDOW_SIZE}
    ++generation_config.fused_block_cache=${FUSED_BLOCK_CACHE_CONFIG}
  )
fi

if [ -n "${COMPILE_MODE}" ]; then
  TORCHRUN_ARGS+=(+compile_mode=${COMPILE_MODE})
fi
if [ -n "${COMPILE_DYNAMIC}" ]; then
  TORCHRUN_ARGS+=(+compile_dynamic=${COMPILE_DYNAMIC})
fi

PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} \
  scripts/eval/uncond_gen_ppl.py "${TORCHRUN_ARGS[@]}"
