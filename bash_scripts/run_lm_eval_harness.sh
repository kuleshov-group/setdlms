#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"
source "${REPO_ROOT}/bash_scripts/eval_model_paths.sh"

SEED="${SEED:-1234}"
export LM_EVAL_RANK_INVARIANT_SEED="${LM_EVAL_RANK_INVARIANT_SEED:-true}"
export LM_EVAL_BASE_SEED="${LM_EVAL_BASE_SEED:-${SEED}}"

resolve_eval_model_path "${MODEL_PATH:-${EVAL_MODEL_KEY:-}}"

is_setdlm_model_path() {
  case "${MODEL_PATH}" in
    *setdlm*|*SetDLM*|*aoarm*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if is_setdlm_model_path; then
  # Historical GSM8K SetDLM/Pareto-compatible defaults. Callers may override.
  KV_CACHING="${KV_CACHING:-true}"
  BLOCK_SIZE="${BLOCK_SIZE:-1024}"
  ALIGN_INPUTS_TO_BLOCKS="${ALIGN_INPUTS_TO_BLOCKS:-false}"
  NUM_VISIBLE_DEVICES="${NUM_VISIBLE_DEVICES:-2}"
else
  KV_CACHING="${KV_CACHING:-false}"
  BLOCK_SIZE="${BLOCK_SIZE:-32}"
  ALIGN_INPUTS_TO_BLOCKS="${ALIGN_INPUTS_TO_BLOCKS:-true}"
  NUM_VISIBLE_DEVICES="${NUM_VISIBLE_DEVICES:-1}"
fi

infer_setdlm_max_window_size() {
  case "${MODEL_PATH}" in
    *aoarm_tgt4_max1024_distill_again_v2*|*setdlm-gsm8k-smax8*)
      echo "4"
      ;;
    *aoarm_tgt8_max1024_distill_v23*|*setdlm-gsm8k-smax16*)
      echo "8"
      ;;
    *aoarm_tgt16_max1024_distill_again_v2*|*setdlm-gsm8k-smax32*)
      echo "16"
      ;;
  esac
}

if [ -z "${MAX_WINDOW_SIZE+x}" ] || [ -z "${MAX_WINDOW_SIZE}" ]; then
  INFERRED_MAX_WINDOW_SIZE="$(infer_setdlm_max_window_size)"
  if [ -n "${INFERRED_MAX_WINDOW_SIZE}" ]; then
    MAX_WINDOW_SIZE="${INFERRED_MAX_WINDOW_SIZE}"
  elif [[ "${MODEL_PATH}" == *setdlm* || "${MODEL_PATH}" == *aoarm* ]]; then
    echo "ERROR: MAX_WINDOW_SIZE must be set explicitly for SetDLM-like MODEL_PATH=${MODEL_PATH}." >&2
    echo "Known GSM8K paper SetDLM targets infer S<=8 -> 4, S<=16 -> 8, S<=32 -> 16." >&2
    exit 1
  else
    MAX_WINDOW_SIZE="${BLOCK_SIZE}"
  fi
fi

# Preserve historical lm-eval harness decoding unless callers opt in.
MATCH_TRAINING_CONTEXT_LENGTH="${MATCH_TRAINING_CONTEXT_LENGTH:-false}"
STOP_ON_IM_END="${STOP_ON_IM_END:-false}"

echo "MODEL_PATH: ${MODEL_PATH}"

USE_EMA=true
OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-outputs/${MODEL_PATH}/lm_eval_harness_output}"
REVISION=null
TOKENIZER_PATH="Qwen/Qwen3-1.7B-Base"


T=${BLOCK_SIZE}
L=1024
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
FIRST_HITTING=false
CONFIDENCE_BASED_NOISING="${CONFIDENCE_BASED_NOISING:-false}"
CONFIDENCE_MARGIN_BASED_NOISING="${CONFIDENCE_MARGIN_BASED_NOISING:-false}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.9}"
CKPT="best"
LINEAR_UNMASKING=true
if is_setdlm_model_path; then
  SETDLM_FHT_CACHE_ORDER="${SETDLM_FHT_CACHE_ORDER:-true}"
else
  SETDLM_FHT_CACHE_ORDER="${SETDLM_FHT_CACHE_ORDER:-false}"
fi
NOISE_MAX_BLOCK_SIZE="${NOISE_MAX_BLOCK_SIZE:-}"
if [ -z "${NOISE_MAX_BLOCK_SIZE}" ] && [ "${SETDLM_FHT_CACHE_ORDER}" = true ]; then
  NOISE_MAX_BLOCK_SIZE=$((2 * MAX_WINDOW_SIZE))
fi
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"

echo "CONFIDENCE_THRESHOLD: ${CONFIDENCE_THRESHOLD}"
echo "MATCH_TRAINING_CONTEXT_LENGTH: ${MATCH_TRAINING_CONTEXT_LENGTH}"
echo "STOP_ON_IM_END: ${STOP_ON_IM_END}"
echo "MAX_WINDOW_SIZE: ${MAX_WINDOW_SIZE}"
echo "NUM_VISIBLE_DEVICES: ${NUM_VISIBLE_DEVICES}"
echo "SEED: ${SEED}"
echo "LM_EVAL_RANK_INVARIANT_SEED: ${LM_EVAL_RANK_INVARIANT_SEED}"
echo "LM_EVAL_BASE_SEED: ${LM_EVAL_BASE_SEED}"
echo "SETDLM_FHT_CACHE_ORDER: ${SETDLM_FHT_CACHE_ORDER}"
if [ -n "${NOISE_MAX_BLOCK_SIZE}" ]; then
  echo "NOISE_MAX_BLOCK_SIZE: ${NOISE_MAX_BLOCK_SIZE}"
fi
if [ -n "${LM_EVAL_LIMIT}" ]; then
  echo "LM_EVAL_LIMIT: ${LM_EVAL_LIMIT}"
fi
MAX_NEW_TOKENS_ARGS=()
TASK_ARGS=()
if [ -n "${LM_EVAL_LIMIT}" ]; then
  TASK_ARGS+=(+task.limit=${LM_EVAL_LIMIT})
fi
if [ "${MATCH_TRAINING_CONTEXT_LENGTH}" != true ]; then
  MAX_NEW_TOKENS_ARGS+=(max_new_tokens=${L})
fi
STOPPING_CRITERIA_LIST="[gsm8k_regex_stopping_criteria,repeating_token]"
if [ "${STOP_ON_IM_END}" = true ]; then
  STOPPING_CRITERIA_LIST="[gsm8k_regex_stopping_criteria,gsm8k_im_end_stopping_criteria,repeating_token]"
fi
echo "T: ${T}"
echo "LINEAR_UNMASKING: ${LINEAR_UNMASKING}"
echo "DO_SAMPLE: ${DO_SAMPLE}"
echo "SAMPLING_STRATEGY: ${SAMPLING_STRATEGY}"
echo "FIRST_HITTING: ${FIRST_HITTING}"
echo "CONFIDENCE_BASED_NOISING: ${CONFIDENCE_BASED_NOISING}"
echo "CONFIDENCE_MARGIN_BASED_NOISING: ${CONFIDENCE_MARGIN_BASED_NOISING}"
echo "ALIGN_INPUTS_TO_BLOCKS: ${ALIGN_INPUTS_TO_BLOCKS}"
echo "TOKENIZER_PATH: ${TOKENIZER_PATH}"
echo "STOPPING_CRITERIA_LIST: ${STOPPING_CRITERIA_LIST}"

OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_L${L}_block${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hit${FIRST_HITTING}-conf_noise${CONFIDENCE_BASED_NOISING}-conf_margin_noise${CONFIDENCE_MARGIN_BASED_NOISING}-conf_thold${CONFIDENCE_THRESHOLD}-align_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-max_window_size${MAX_WINDOW_SIZE}"
if [ "${MATCH_TRAINING_CONTEXT_LENGTH}" = true ]; then
  OUTPUT_PATH="${OUTPUT_PATH}_match_train_len"
fi
if [ "${STOP_ON_IM_END}" = true ]; then
  OUTPUT_PATH="${OUTPUT_PATH}_stop_im_end"
fi
if [ -n "${LM_EVAL_LIMIT}" ]; then
  OUTPUT_PATH="${OUTPUT_PATH}_limit${LM_EVAL_LIMIT}"
fi
OUTPUT_PATH="${OUTPUT_PATH}_test"
mkdir -p ${OUTPUT_PATH}

MODEL_ARGS=()
if [ "${SETDLM_FHT_CACHE_ORDER}" = true ]; then
  MODEL_ARGS+=(+task.model.model_config_overrides.setdlm_fht_cache_order=true)
  if [ -n "${NOISE_MAX_BLOCK_SIZE}" ]; then
    MODEL_ARGS+=(+task.model.model_config_overrides.noise_config.max_block_size=${NOISE_MAX_BLOCK_SIZE})
  fi
fi
GENERATION_ARGS=(
  generation@generation_config=set_diffusion_generation_config
  generation_config.do_sample=${DO_SAMPLE}
  generation_config.sampling_strategy=${SAMPLING_STRATEGY}
  generation_config.num_steps=${T}
  generation_config.first_hitting=${FIRST_HITTING}
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING}
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING}
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD}
  generation_config.use_cache=${KV_CACHING}
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS}
  generation_config.max_window_size=${MAX_WINDOW_SIZE}
  generation_config.linear_unmasking=${LINEAR_UNMASKING}
)

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
  "${TASK_ARGS[@]}" \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  task.model.ckpt_file="${CKPT}-rank0.pt" \
  task.model.load_ema_weights=${USE_EMA} \
  task.model.stop_on_im_end=${STOP_ON_IM_END} \
  tokenizer.pretrained_model_name_or_path=${TOKENIZER_PATH} \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${L} \
  "${MAX_NEW_TOKENS_ARGS[@]}" \
  block_size=${BLOCK_SIZE} \
  "${MODEL_ARGS[@]}" \
  "${GENERATION_ARGS[@]}" \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  generation/stopping_criteria@stopping_criteria_list="${STOPPING_CRITERIA_LIST}"
