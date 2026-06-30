#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}" || exit
source setup_env.sh
source "${REPO_ROOT}/bash_scripts/eval_model_paths.sh"
set -u

MODEL_NAME="${MODEL_NAME:-SetDLM}"
resolve_eval_model_path "${MODEL_PATH:-${EVAL_MODEL_KEY:-}}"
CKPT_FILE="${CKPT_FILE:-best-rank0.pt}"
REVISION=null
TOKENIZER_PATH="gpt2"

BLOCK_SIZE="${BLOCK_SIZE:-1024}"
MAX_BLOCK_SIZE="${MAX_BLOCK_SIZE:-8}"
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
if [[ "${MODEL_NAME}" == SetDLM* ]]; then
  MODEL_CONFIG_OVERRIDE_ARGS+=(+model_config_overrides.noise_config.max_block_size="${MAX_BLOCK_SIZE}")
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
