#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"
source "${REPO_ROOT}/bash_scripts/eval_model_paths.sh"

COMPILE_BACKBONE=${COMPILE_BACKBONE:-false}
COMPILE_MODE=${COMPILE_MODE:-}
COMPILE_DYNAMIC=${COMPILE_DYNAMIC:-}
THROUGHPUT_RUN="${THROUGHPUT_RUN:-false}"
THROUGHPUT_WARMUP="${THROUGHPUT_WARMUP:-50}"
THROUGHPUT_MEASUREMENTS="${THROUGHPUT_MEASUREMENTS:-200}"
THROUGHPUT_GLOBAL_MEASUREMENTS="${THROUGHPUT_GLOBAL_MEASUREMENTS:-false}"
OUTPUT_PATH_OVERRIDE="${OUTPUT_PATH_OVERRIDE:-}"
REPRO_METADATA_PATH="${REPRO_METADATA_PATH:-}"
REPRO_ROW_ID="${REPRO_ROW_ID:-}"
REPRO_REPEAT_ID="${REPRO_REPEAT_ID:-}"
REPRO_OUTPUT_ROOT="${REPRO_OUTPUT_ROOT:-}"
REPRO_LOG_PATH_TEMPLATE="${REPRO_LOG_PATH_TEMPLATE:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
ROC_STORIES_INSERT_BOS_TOKEN="${ROC_STORIES_INSERT_BOS_TOKEN:-}"
ROC_STORIES_INSERT_EOS_TOKEN="${ROC_STORIES_INSERT_EOS_TOKEN:-}"
ROC_STORIES_SAMPLE_INDICES="${ROC_STORIES_SAMPLE_INDICES:-}"

resolve_eval_model_path "${MODEL_PATH:-${EVAL_MODEL_KEY:-}}"
CKPT_FILE="${CKPT_FILE:-best-rank0.pt}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
MAX_WINDOW_SIZE="${MAX_WINDOW_SIZE:-${BLOCK_SIZE}}"
KV_CACHING="${KV_CACHING:-true}"
ALIGN_INPUTS_TO_BLOCKS="${ALIGN_INPUTS_TO_BLOCKS:-false}"
USE_EMA="${USE_EMA:-false}"
CACHE_FULL_INFILL_CONTEXT="${CACHE_FULL_INFILL_CONTEXT:-true}"
INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT="${INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT:-false}"
SETDLM_INFILL_DIAGNOSTIC_LOG="${SETDLM_INFILL_DIAGNOSTIC_LOG:-false}"
INFILL_CONTEXT_NO_REPEAT_NGRAM_SIZE="${INFILL_CONTEXT_NO_REPEAT_NGRAM_SIZE:-0}"
INFILL_CONTEXT_NO_REPEAT_NGRAM_DIAGNOSTIC_LOG="${INFILL_CONTEXT_NO_REPEAT_NGRAM_DIAGNOSTIC_LOG:-false}"
SETDLM_INFILL_FIRST_HITTING_CACHE_DIAGNOSTIC="${SETDLM_INFILL_FIRST_HITTING_CACHE_DIAGNOSTIC:-false}"
SETDLM_INFILL_CACHE_PROMOTION_ORDER="${SETDLM_INFILL_CACHE_PROMOTION_ORDER:-legacy}"
SETDLM_INFILL_CACHE_PROMOTION_TRACE="${SETDLM_INFILL_CACHE_PROMOTION_TRACE:-false}"
SETDLM_INFILL_CACHE_PROMOTION_TRACE_INPUT_LENGTH="${SETDLM_INFILL_CACHE_PROMOTION_TRACE_INPUT_LENGTH:-null}"
SETDLM_INFILL_CACHE_PROMOTION_TRACE_MAX_STEPS="${SETDLM_INFILL_CACHE_PROMOTION_TRACE_MAX_STEPS:-8}"
USE_EOS_STOPPING_CRITERIA="${USE_EOS_STOPPING_CRITERIA:-false}"
OUTPUT_TAG="${OUTPUT_TAG:-}"

echo "MODEL_PATH: ${MODEL_PATH}"

OUTPUT_DIR="outputs/${MODEL_PATH}/roc_stories"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=1024
T=${BLOCK_SIZE}
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" "posterior"
FIRST_HITTING="${FIRST_HITTING:-false}"
CONFIDENCE_BASED_NOISING="${CONFIDENCE_BASED_NOISING:-false}"
CONF_THRESHOLD="${CONFIDENCE_THRESHOLD:-${CONF_THRESHOLD:-1e6}}"
export CONFIDENCE_THRESHOLD="${CONF_THRESHOLD}"
MAX_LENGTH=1024
NUM_TARGET_SENTENCES="${NUM_TARGET_SENTENCES:-3}"
if [[ -z "${REPEAT_PENALTY:-}" ]]; then
  if [[ "${NUM_TARGET_SENTENCES}" == "1" ]]; then
    REPEAT_PENALTY=1.1
  elif [[ "${NUM_TARGET_SENTENCES}" == "3" ]]; then
    REPEAT_PENALTY=1.5
  else
    REPEAT_PENALTY=1.1
  fi
fi

echo "NUM_TARGET_SENTENCES: ${NUM_TARGET_SENTENCES} REPEAT_PENALTY: ${REPEAT_PENALTY}"
echo "CACHE_FULL_INFILL_CONTEXT: ${CACHE_FULL_INFILL_CONTEXT} INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT: ${INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT} OUTPUT_TAG: ${OUTPUT_TAG}"
echo "INFILL_CONTEXT_NO_REPEAT_NGRAM_SIZE: ${INFILL_CONTEXT_NO_REPEAT_NGRAM_SIZE} INFILL_CONTEXT_NO_REPEAT_NGRAM_DIAGNOSTIC_LOG: ${INFILL_CONTEXT_NO_REPEAT_NGRAM_DIAGNOSTIC_LOG}"
echo "SETDLM_INFILL_DIAGNOSTIC_LOG: ${SETDLM_INFILL_DIAGNOSTIC_LOG} SETDLM_INFILL_FIRST_HITTING_CACHE_DIAGNOSTIC: ${SETDLM_INFILL_FIRST_HITTING_CACHE_DIAGNOSTIC} USE_EOS_STOPPING_CRITERIA: ${USE_EOS_STOPPING_CRITERIA}"
echo "SETDLM_INFILL_CACHE_PROMOTION_ORDER: ${SETDLM_INFILL_CACHE_PROMOTION_ORDER}"
echo "SETDLM_INFILL_CACHE_PROMOTION_TRACE: ${SETDLM_INFILL_CACHE_PROMOTION_TRACE} TRACE_INPUT_LENGTH: ${SETDLM_INFILL_CACHE_PROMOTION_TRACE_INPUT_LENGTH} TRACE_MAX_STEPS: ${SETDLM_INFILL_CACHE_PROMOTION_TRACE_MAX_STEPS}"
echo "MAX_EVAL_SAMPLES: ${MAX_EVAL_SAMPLES:-null} ROC_STORIES_INSERT_BOS_TOKEN: ${ROC_STORIES_INSERT_BOS_TOKEN:-default} ROC_STORIES_INSERT_EOS_TOKEN: ${ROC_STORIES_INSERT_EOS_TOKEN:-default} ROC_STORIES_SAMPLE_INDICES: ${ROC_STORIES_SAMPLE_INDICES:-default}"
echo "CONFIDENCE_BASED_NOISING: ${CONFIDENCE_BASED_NOISING} CONFIDENCE_THRESHOLD: ${CONF_THRESHOLD}"

OUTPUT_TAG_FRAGMENT=""
if [[ -n "${OUTPUT_TAG}" ]]; then
  OUTPUT_TAG_FRAGMENT="-${OUTPUT_TAG}"
fi

OUTPUT_PATH="${OUTPUT_DIR}/num_target_sentences${NUM_TARGET_SENTENCES}/L-${L}-block_size-${BLOCK_SIZE}-T${T}-do_sample-${DO_SAMPLE}-sampling_strategy-${SAMPLING_STRATEGY}-repeat_penalty${REPEAT_PENALTY}-first_hitting-${FIRST_HITTING}-confidence_based_noising-${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt-${CKPT_FILE}-ema${USE_EMA}${OUTPUT_TAG_FRAGMENT}"
if [[ -n "${OUTPUT_PATH_OVERRIDE}" ]]; then
  OUTPUT_PATH="${OUTPUT_PATH_OVERRIDE}"
fi

export OUTPUT_PATH
echo "THROUGHPUT_RUN: ${THROUGHPUT_RUN} THROUGHPUT_MEASUREMENTS: ${THROUGHPUT_MEASUREMENTS} THROUGHPUT_WARMUP: ${THROUGHPUT_WARMUP} THROUGHPUT_GLOBAL_MEASUREMENTS: ${THROUGHPUT_GLOBAL_MEASUREMENTS}"
echo "NUM_VISIBLE_DEVICES: ${NUM_VISIBLE_DEVICES} GPU_CONSTRAINT: ${GPU_CONSTRAINT:-} GPU_PARTITION: ${GPU_PARTITION:-}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

if [[ -n "${REPRO_METADATA_PATH}" ]]; then
  mkdir -p "$(dirname "${REPRO_METADATA_PATH}")"
  python3 - <<'PY'
import json
import os
import socket
import subprocess
from pathlib import Path

def getenv(name, default=""):
    return os.environ.get(name, default)

def gpu_names():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return "N/A"
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "; ".join(names) if names else "N/A"

log_template = getenv("REPRO_LOG_PATH_TEMPLATE", "")
job_id = getenv("SLURM_JOB_ID", "")
log_path = log_template.replace("<jobid>", job_id) if log_template else ""
metadata = {
    "git_sha": getenv("GIT_SHA") or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip(),
    "job_id": job_id,
    "slurm_job_id": job_id,
    "row_id": getenv("REPRO_ROW_ID"),
    "repeat_id": getenv("REPRO_REPEAT_ID"),
    "node_name": getenv("SLURMD_NODENAME") or socket.gethostname(),
    "gpu_name": gpu_names(),
    "num_gpus": getenv("NUM_VISIBLE_DEVICES"),
    "gpu_constraint": getenv("GPU_CONSTRAINT"),
    "gpu_partition": getenv("GPU_PARTITION"),
    "output_root": getenv("REPRO_OUTPUT_ROOT"),
    "output_path": getenv("OUTPUT_PATH"),
    "log_path": log_path,
    "model_path": getenv("MODEL_PATH"),
    "ckpt_file": getenv("CKPT_FILE"),
    "use_ema": getenv("USE_EMA"),
    "block_size": getenv("BLOCK_SIZE"),
    "max_window_size": getenv("MAX_WINDOW_SIZE"),
    "num_target_sentences": getenv("NUM_TARGET_SENTENCES"),
    "first_hitting": getenv("FIRST_HITTING"),
    "repeat_penalty": getenv("REPEAT_PENALTY"),
    "align_inputs_to_blocks": getenv("ALIGN_INPUTS_TO_BLOCKS"),
    "cache_full_infill_context": getenv("CACHE_FULL_INFILL_CONTEXT"),
    "infill_context_no_repeat_ngram_size": getenv("INFILL_CONTEXT_NO_REPEAT_NGRAM_SIZE"),
    "infill_context_no_repeat_ngram_diagnostic_log": getenv("INFILL_CONTEXT_NO_REPEAT_NGRAM_DIAGNOSTIC_LOG"),
    "infill_repetition_penalty_include_right_context": getenv("INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT"),
    "setdlm_infill_diagnostic_log": getenv("SETDLM_INFILL_DIAGNOSTIC_LOG"),
    "setdlm_infill_first_hitting_cache_diagnostic": getenv("SETDLM_INFILL_FIRST_HITTING_CACHE_DIAGNOSTIC"),
    "setdlm_infill_cache_promotion_order": getenv("SETDLM_INFILL_CACHE_PROMOTION_ORDER"),
    "setdlm_infill_cache_promotion_trace": getenv("SETDLM_INFILL_CACHE_PROMOTION_TRACE"),
    "setdlm_infill_cache_promotion_trace_input_length": getenv("SETDLM_INFILL_CACHE_PROMOTION_TRACE_INPUT_LENGTH"),
    "setdlm_infill_cache_promotion_trace_max_steps": getenv("SETDLM_INFILL_CACHE_PROMOTION_TRACE_MAX_STEPS"),
    "use_eos_stopping_criteria": getenv("USE_EOS_STOPPING_CRITERIA"),
    "throughput_run": getenv("THROUGHPUT_RUN"),
    "throughput_measured_examples": getenv("THROUGHPUT_MEASUREMENTS"),
    "throughput_warmup_examples_per_rank": getenv("THROUGHPUT_WARMUP"),
    "throughput_global_measurements": getenv("THROUGHPUT_GLOBAL_MEASUREMENTS"),
    "throughput_num_gpus": getenv("NUM_VISIBLE_DEVICES"),
    "compile_backbone": getenv("COMPILE_BACKBONE"),
    "compile_mode": getenv("COMPILE_MODE"),
    "compile_dynamic": getenv("COMPILE_DYNAMIC"),
    "compile_supported": "true" if getenv("COMPILE_BACKBONE") == "true" else "",
    "max_eval_samples": getenv("MAX_EVAL_SAMPLES"),
    "roc_stories_insert_bos_token": getenv("ROC_STORIES_INSERT_BOS_TOKEN"),
    "roc_stories_insert_eos_token": getenv("ROC_STORIES_INSERT_EOS_TOKEN"),
    "roc_stories_sample_indices": getenv("ROC_STORIES_SAMPLE_INDICES"),
}
path = Path(getenv("REPRO_METADATA_PATH"))
path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
fi

EXTRA_ARGS=()
STOPPING_ARGS=()
if [[ "${USE_EOS_STOPPING_CRITERIA}" == "true" ]]; then
  STOPPING_ARGS+=(generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria]')
else
  STOPPING_ARGS+=(~generation/stopping_criteria@stopping_criteria_list)
  STOPPING_ARGS+=(gen_kwargs.stopping_criteria=null)
fi
if [[ -n "${COMPILE_MODE}" ]]; then
  EXTRA_ARGS+=(+compile_mode=${COMPILE_MODE})
fi
if [[ -n "${COMPILE_DYNAMIC}" ]]; then
  EXTRA_ARGS+=(+compile_dynamic=${COMPILE_DYNAMIC})
fi
if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
  EXTRA_ARGS+=(+max_eval_samples=${MAX_EVAL_SAMPLES})
fi
if [[ -n "${ROC_STORIES_INSERT_BOS_TOKEN}" ]]; then
  EXTRA_ARGS+=(+task.dataset.insert_bos_token=${ROC_STORIES_INSERT_BOS_TOKEN})
fi
if [[ -n "${ROC_STORIES_INSERT_EOS_TOKEN}" ]]; then
  EXTRA_ARGS+=(+task.dataset.insert_eos_token=${ROC_STORIES_INSERT_EOS_TOKEN})
fi
if [[ -n "${ROC_STORIES_SAMPLE_INDICES}" ]]; then
  EXTRA_ARGS+=(+task.dataset.sample_indices=${ROC_STORIES_SAMPLE_INDICES})
fi

LOGITS_PROCESSOR_ARGS=(
  "generation/logits_processor@logits_processor_list=[repetition_penalty_logits_processor]"
  "logits_processor_list.repetition_penalty_logits_processor.penalty=${REPEAT_PENALTY}"
)

PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=roc_stories \
  task.dataset.num_target_sentences=${NUM_TARGET_SENTENCES} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file=${CKPT_FILE} \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="gpt2" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation@generation_config=set_diffusion_generation_config \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_threshold=${CONF_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation_config.cache_full_infill_context=${CACHE_FULL_INFILL_CONTEXT} \
  generation_config.infill_repetition_penalty_include_right_context=${INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT} \
  generation_config.infill_context_no_repeat_ngram_size=${INFILL_CONTEXT_NO_REPEAT_NGRAM_SIZE} \
  generation_config.infill_context_no_repeat_ngram_diagnostic_log=${INFILL_CONTEXT_NO_REPEAT_NGRAM_DIAGNOSTIC_LOG} \
  generation_config.setdlm_infill_diagnostic_log=${SETDLM_INFILL_DIAGNOSTIC_LOG} \
  generation_config.setdlm_infill_first_hitting_cache_diagnostic=${SETDLM_INFILL_FIRST_HITTING_CACHE_DIAGNOSTIC} \
  generation_config.setdlm_infill_cache_promotion_order=${SETDLM_INFILL_CACHE_PROMOTION_ORDER} \
  generation_config.setdlm_infill_cache_promotion_trace=${SETDLM_INFILL_CACHE_PROMOTION_TRACE} \
  generation_config.setdlm_infill_cache_promotion_trace_input_length=${SETDLM_INFILL_CACHE_PROMOTION_TRACE_INPUT_LENGTH} \
  generation_config.setdlm_infill_cache_promotion_trace_max_steps=${SETDLM_INFILL_CACHE_PROMOTION_TRACE_MAX_STEPS} \
  generation_config.max_window_size=${MAX_WINDOW_SIZE} \
  generation_config.linear_unmasking=true \
  "${STOPPING_ARGS[@]}" \
  "${LOGITS_PROCESSOR_ARGS[@]}" \
  +model_config_overrides.backbone_config.attn_backend=sdpa \
  +model_config_overrides.attn_backend=sdpa \
  +compile_backbone=${COMPILE_BACKBONE} \
  +throughput_run=${THROUGHPUT_RUN} \
  +throughput_warmup=${THROUGHPUT_WARMUP} \
  +throughput_num_measurements=${THROUGHPUT_MEASUREMENTS} \
  +throughput_global_measurements=${THROUGHPUT_GLOBAL_MEASUREMENTS} \
  "${EXTRA_ARGS[@]}"
