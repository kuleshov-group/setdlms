#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}" || exit
source setup_env.sh
source "${REPO_ROOT}/bash_scripts/eval_model_paths.sh"
set -u

REVISION=${REVISION:-null}
BLOCK_SIZE=${BLOCK_SIZE:-128}
BATCH_SIZE=${BATCH_SIZE:-16}
if [ -z "${DATA_DIR:-}" ]; then
  if [ -d /share/kuleshov/ma2238/dllm-data ]; then
    DATA_DIR=/share/kuleshov/ma2238/dllm-data
  else
    DATA_DIR="${REPO_ROOT}/data"
  fi
fi
export DATA_DIR

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

resolve_eval_model_path "${LM1B_MODEL_PATH:-${LM1B_MODEL_KEY:-${EVAL_MODEL_KEY:-}}}"
LM1B_MODEL_PATH="${MODEL_PATH}"
LM1B_CKPT_FILE="${LM1B_CKPT_FILE:-${CKPT_FILE:-best-rank0.pt}}"
LM1B_USE_EMA="${LM1B_USE_EMA:-${USE_EMA:-false}}"

SETDLM_CHECKPOINTS=(
  "${LM1B_MODEL_NAME:-${RESOLVED_MODEL_LABEL:-SetDLM}}|${LM1B_MODEL_PATH}|${LM1B_CKPT_FILE}|${LM1B_USE_EMA}"
)

for checkpoint in "${SETDLM_CHECKPOINTS[@]}"; do
  IFS="|" read -r MODEL_NAME MODEL_PATH CKPT_FILE USE_EMA <<< "${checkpoint}"

  MODEL_CONFIG_OVERRIDE_ARGS=()
  set_setdlm_ppl_noise_max_block_size

  echo "Evaluating ${MODEL_NAME}"
  echo "  model: ${MODEL_PATH}"
  echo "  ckpt: ${CKPT_FILE}"
  echo "  use_ema: ${USE_EMA}"
  if is_setdlm_model; then
    echo "  setdlm_desired_block_size: ${SETDLM_DESIRED_BLOCK_SIZE:-$(infer_setdlm_desired_block_size)}"
    echo "  noise_max_block_size: ${MAX_BLOCK_SIZE}"
  fi

  composer -n "${NUM_VISIBLE_DEVICES}" scripts/eval/likelihood_eval.py \
    hydra.output_subdir=null \
    hydra.run.dir="${PWD}" \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    +eval@task=likelihood \
    +dataset@task.eval_dataset=lm1b_eval \
    task.load_ema_weights="${USE_EMA}" \
    task.ckpt_file="${CKPT_FILE}" \
    seed=1 \
    batch_size="${BATCH_SIZE}" \
    block_size="${BLOCK_SIZE}" \
    task.eval_dataloader.batch_size=8 \
    pretrained_model_name_or_path="${MODEL_PATH}" \
    pretrained_model_revision="${REVISION}" \
    tokenizer.pretrained_model_name_or_path=bert-base-uncased \
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
    "${MODEL_CONFIG_OVERRIDE_ARGS[@]}"
done
