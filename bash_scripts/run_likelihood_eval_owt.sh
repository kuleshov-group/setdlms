#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}" || exit
source setup_env.sh
source "${REPO_ROOT}/bash_scripts/eval_model_paths.sh"
set -u

REQUESTED_MODEL="${MODEL_PATH:-${EVAL_MODEL_KEY:-}}"
resolve_eval_model_path "${REQUESTED_MODEL}"
MODEL_NAME="${MODEL_NAME:-${RESOLVED_MODEL_LABEL:-SetDLM}}"
CKPT_FILE="${CKPT_FILE:-best-rank0.pt}"
REVISION=null
TOKENIZER_PATH="gpt2"

is_setdlm_model() {
  case "${MODEL_NAME:-} ${MODEL_PATH:-} ${EVAL_MODEL_KEY:-} ${LM1B_MODEL_KEY:-} ${LM1B_MODEL_PATH:-}" in
    *SetDLM*|*setdlm*|*aoarm*) return 0 ;;
    *) return 1 ;;
  esac
}

infer_setdlm_desired_block_size() {
  case "${MODEL_NAME:-} ${MODEL_PATH:-} ${EVAL_MODEL_KEY:-} ${LM1B_MODEL_KEY:-} ${LM1B_MODEL_PATH:-}" in
    *d4*|*tgt4*|*smax8*) echo 4 ;;
    *d8*|*tgt8*|*smax16*) echo 8 ;;
    *d16*|*tgt16*|*smax32*) echo 16 ;;
    *) echo "" ;;
  esac
}

set_setdlm_ppl_noise_max_block_size() {
  if is_setdlm_model; then
    local desired_block_size="${SETDLM_DESIRED_BLOCK_SIZE:-$(infer_setdlm_desired_block_size)}"
    if [ -z "${desired_block_size}" ]; then
      echo "ERROR: Could not infer SetDLM desired block size for MODEL_NAME=${MODEL_NAME:-}, MODEL_PATH=${MODEL_PATH:-}." >&2
      exit 1
    fi
    MAX_BLOCK_SIZE="${MAX_BLOCK_SIZE:-$((2 * desired_block_size))}"
    MODEL_CONFIG_OVERRIDE_ARGS+=(+model_config_overrides.noise_config.max_block_size="${MAX_BLOCK_SIZE}")
  fi
}

BLOCK_SIZE="${BLOCK_SIZE:-1024}"
MAX_BLOCK_SIZE="${MAX_BLOCK_SIZE:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
USE_EMA="${USE_EMA:-false}"
COMPILE_BACKBONE="${COMPILE_BACKBONE:-true}"
EVAL_NUM_PROCESSES=${EVAL_NUM_PROCESSES:-${NUM_VISIBLE_DEVICES:-1}}

echo "Evaluating ${MODEL_NAME}"
echo "  model: ${MODEL_PATH}"
echo "  ckpt: ${CKPT_FILE}"
echo "  use_ema: ${USE_EMA}"
echo "  eval_dataset: owt_eval_gpt2"

MODEL_CONFIG_OVERRIDE_ARGS=()
set_setdlm_ppl_noise_max_block_size
if is_setdlm_model; then
  echo "  setdlm_desired_block_size: ${SETDLM_DESIRED_BLOCK_SIZE:-$(infer_setdlm_desired_block_size)}"
  echo "  noise_max_block_size: ${MAX_BLOCK_SIZE}"
fi

composer -n "${EVAL_NUM_PROCESSES}" scripts/eval/likelihood_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval@task=likelihood \
  +dataset@task.eval_dataset=owt_eval_gpt2 \
  task.load_ema_weights="${USE_EMA}" \
  task.ckpt_file="${CKPT_FILE}" \
  seed=1 \
  batch_size="${BATCH_SIZE}" \
  block_size="${BLOCK_SIZE}" \
  task.eval_dataloader.batch_size="${BATCH_SIZE}" \
  pretrained_model_name_or_path="${MODEL_PATH}" \
  pretrained_model_revision="${REVISION}" \
  tokenizer.pretrained_model_name_or_path="${TOKENIZER_PATH}" \
  output_path=null \
  +collator@task.collator=denoising \
  task.collator.global_batch_size="${BATCH_SIZE}" \
  task.collator.max_length=null \
  task.collator.restricted_t_range=null \
  task.collator.sampling_eps=1e-3 \
  task.collator.antithetic_sampling=false \
  +metrics@task.metrics='[loss,nll,bpd,perplexity]' \
  +composer/trainer@task.trainer=eval_trainer \
  ~generation@generation_config \
  ~generation/logits_processor@logits_processor_list \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs=null \
  +compile_backbone="${COMPILE_BACKBONE}" \
  "${MODEL_CONFIG_OVERRIDE_ARGS[@]}"
