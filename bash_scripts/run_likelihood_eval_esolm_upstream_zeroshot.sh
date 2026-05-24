#!/bin/bash
# Setup environment relative to the repo, including Slurm jobs where this
# script is copied into a spool directory before execution.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${REPO_ROOT:-}" ] && [ -f "${REPO_ROOT}/setup_env.sh" ]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/../setup_env.sh" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"

# Upstream-style EsoLM likelihood evaluation over the zero-shot dataset suite.
# Reuses the paper/upstream likelihood semantics more closely:
# - default to a single process with a local batch of 32 examples so the
#   low-discrepancy timestep sampler operates over the same batch size used in
#   the reported bounds
# - use GPT-2-tokenized packed validation/test caches with `specialFalse`
# - alpha_0 and batch_split must match training-time values

MODEL_PATH="${ESOLM_MODEL_PATH:-/share/kuleshov/ma2238/dllm-dev-new/dllm-dev/esolmb-alpha0-0d125-batchsplit-0d5-250k.ckpt}"
CKPT_FILE="${ESOLM_CKPT_FILE:-null}"
TOKENIZER_PATH="${ESOLM_TOKENIZER_PATH:-gpt2}"
USE_EMA="${ESOLM_USE_EMA:-true}"

ESOLM_ALPHA_0="${ESOLM_ALPHA_0:-0.125}"
ESOLM_BATCH_SPLIT="${ESOLM_BATCH_SPLIT:-0.5}"

BLOCK_SIZE="${ESOLM_BLOCK_SIZE:-1024}"
COMPILE_BACKBONE="${ESOLM_COMPILE_BACKBONE:-false}"

EVAL_NUM_PROCESSES="${ESOLM_UPSTREAM_NUM_PROCESSES:-1}"
UPSTREAM_PER_PROCESS_BATCH_SIZE="${ESOLM_UPSTREAM_PER_PROCESS_BATCH_SIZE:-32}"

if [ "${EVAL_NUM_PROCESSES}" -le 0 ]; then
  echo "ESOLM_UPSTREAM_NUM_PROCESSES must be positive, got ${EVAL_NUM_PROCESSES}."
  exit 1
fi

if [ "${UPSTREAM_PER_PROCESS_BATCH_SIZE}" -le 0 ]; then
  echo "ESOLM_UPSTREAM_PER_PROCESS_BATCH_SIZE must be positive, got ${UPSTREAM_PER_PROCESS_BATCH_SIZE}."
  exit 1
fi

PER_PROCESS_BATCH_SIZE="${UPSTREAM_PER_PROCESS_BATCH_SIZE}"
UPSTREAM_MACHINE_BATCH_SIZE=$((UPSTREAM_PER_PROCESS_BATCH_SIZE * EVAL_NUM_PROCESSES))
TMPDIR_ROOT="${ESOLM_TMPDIR_ROOT:-/tmp/${USER}/composer_tmp}"
mkdir -p "${TMPDIR_ROOT}"
export TMPDIR="${TMPDIR_ROOT}"

read -r -a EVAL_DATASETS <<< "${ESOLM_UPSTREAM_ZERO_SHOT_DATASETS:-ptb_eval_esolm_upstream wikitext2_eval_esolm_upstream lm1b_eval_gpt2_esolm_upstream lambada_eval_esolm_upstream ag_news_eval_esolm_upstream scientific_papers_pubmed_eval_esolm_upstream scientific_papers_arxiv_eval_esolm_upstream}"

if [ "${#EVAL_DATASETS[@]}" -eq 0 ]; then
  echo "No datasets configured for zero-shot upstream eval."
  exit 1
fi

echo "Evaluating upstream-style zero-shot likelihood suite with model ${MODEL_PATH}"
echo "alpha_0=${ESOLM_ALPHA_0} batch_split=${ESOLM_BATCH_SPLIT} machine_batch=${UPSTREAM_MACHINE_BATCH_SIZE} per_process_batch=${PER_PROCESS_BATCH_SIZE} processes=${EVAL_NUM_PROCESSES}"

for EVAL_DATASET in "${EVAL_DATASETS[@]}"; do
  echo "Evaluating ${EVAL_DATASET}"

  composer -n "${EVAL_NUM_PROCESSES}" "${REPO_ROOT}/scripts/eval/likelihood_eval.py" \
    hydra.output_subdir=null \
    hydra.run.dir="${REPO_ROOT}" \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    +eval@task=likelihood \
    +dataset@task.eval_dataset="${EVAL_DATASET}" \
    task.load_ema_weights="${USE_EMA}" \
    task.ckpt_file="${CKPT_FILE}" \
    seed=1 \
    batch_size="${UPSTREAM_MACHINE_BATCH_SIZE}" \
    block_size="${BLOCK_SIZE}" \
    task.eval_dataloader.batch_size="${PER_PROCESS_BATCH_SIZE}" \
    pretrained_model_name_or_path="${MODEL_PATH}" \
    tokenizer.pretrained_model_name_or_path="${TOKENIZER_PATH}" \
    output_path=null \
    +collator@task.collator=denoising \
    task.collator.global_batch_size="${UPSTREAM_MACHINE_BATCH_SIZE}" \
    task.collator.max_length=null \
    task.collator.restricted_t_range=null \
    task.collator.sampling_eps=1e-3 \
    task.collator.antithetic_sampling=true \
    +metrics@task.metrics='[loss,nll,bpd,perplexity]' \
    +composer/trainer@task.trainer=eval_trainer \
    ~generation@generation_config \
    ~generation/logits_processor@logits_processor_list \
    ~generation/stopping_criteria@stopping_criteria_list \
    gen_kwargs=null \
    +compile_backbone="${COMPILE_BACKBONE}" \
    +model_config_overrides.alpha_0="${ESOLM_ALPHA_0}" \
    +model_config_overrides.batch_split="${ESOLM_BATCH_SPLIT}" \
    +model_config_overrides.sampling_eps=1e-3 \
    +model_config_overrides.antithetic_sampling=true \
    +model_config_overrides.diffusion_attn_mode=causal \
    +model_config_overrides.diffusion_shuffle=true \
    +model_config_overrides.sequential_attn_mode=causal \
    +model_config_overrides.sequential_shuffle=true \
    +model_config_overrides.loss_type=elbo \
    +model_config_overrides.keep_clean_bos=true \
    +model_config_overrides.mdlm_loss_scale=false \
    +model_config_overrides.attn_backend=sdpa \
    +model_config_overrides.backbone_config.attn_backend=sdpa
done
