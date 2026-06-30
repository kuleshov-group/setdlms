#!/bin/bash

<<comment
# Usage:
#   ./sbatch_wrapper.sh <SHELL_SCRIPT>
comment


WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WRAPPER_DIR}/.." && pwd)"

if [ -z "${1:-}" ]; then
  echo "Usage: $0 script_name"
  exit 1
fi

script_name="$1"
if [[ "$script_name" != *.sh ]]; then
  script_name="${script_name}.sh"
fi

# Construct the full path
script_full_path=$(realpath "${WRAPPER_DIR}/${script_name}")

# Check if the file exists in the directory
if [ ! -e "${script_full_path}" ]; then
  echo "Script '${script_full_path}' not found."
  exit 1
fi



WATCH_FOLDER=$(realpath "${REPO_ROOT}/watch_folder")
mkdir -p ${WATCH_FOLDER}
USERNAME=$(whoami)
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp/${USERNAME}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.triton_cache/${USERNAME}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${REPO_ROOT}/.torchinductor_cache/${USERNAME}}"
if [ "${THROUGHPUT_RUN:-false}" = "true" ]; then
  NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-4}
  GPU_PARTITION=${GPU_PARTITION:-"kuleshov"}
  GPU_CONSTRAINT=${GPU_CONSTRAINT:-"[a6000]"}
  THROUGHPUT_NODELIST=${THROUGHPUT_NODELIST:-"kuleshov-compute-02"}
else
  NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-4}
  GPU_PARTITION=${GPU_PARTITION:-"gpu,kuleshov"}
  GPU_CONSTRAINT=${GPU_CONSTRAINT:-"[a6000]"}
fi
JOB_MEM=${JOB_MEM:-128000}
GPU_EXCLUDE_BASE="nikola-compute-[01-05,11-18],goyal-compute-01,snavely-compute-02,rush-compute-02,sun-compute-01,klara,rush-compute-03,ma-compute-02,ellis-compute-02,lancer-compute-01,lil-compute-04,seo-compute-02,badfellow,joachims-compute-03"
GPU_EXCLUDE_EXTRA_COMBINED="${GPU_EXCLUDE_EXTRA:-}"
if [ "${THROUGHPUT_RUN:-false}" = "true" ] && [ -n "${THROUGHPUT_NODE_EXCLUDE_EXTRA:-}" ]; then
  if [ -n "${GPU_EXCLUDE_EXTRA_COMBINED}" ]; then
    GPU_EXCLUDE_EXTRA_COMBINED="${GPU_EXCLUDE_EXTRA_COMBINED},${THROUGHPUT_NODE_EXCLUDE_EXTRA}"
  else
    GPU_EXCLUDE_EXTRA_COMBINED="${THROUGHPUT_NODE_EXCLUDE_EXTRA}"
  fi
fi
if [ -n "${GPU_EXCLUDE_EXTRA_COMBINED}" ]; then
  GPU_EXCLUDE="${GPU_EXCLUDE_BASE},${GPU_EXCLUDE_EXTRA_COMBINED}"
else
  GPU_EXCLUDE="${GPU_EXCLUDE_BASE}"
fi
GPU_NODELIST="${GPU_NODELIST:-${THROUGHPUT_NODELIST:-}}"
NODELIST_ARGS=()
if [ -n "${GPU_NODELIST}" ]; then
  NODELIST_ARGS=(--nodelist="${GPU_NODELIST}")
fi
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/runs}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
mkdir -p ${RUN_DIR}
mkdir -p ${DATA_DIR}
if ! command -v sbatch >/dev/null 2>&1; then
  if [ -x /usr/local/slurm/current/bin/sbatch ]; then
    export PATH="/usr/local/slurm/current/bin:${PATH}"
  fi
fi
SBATCH_BIN="$(command -v sbatch || true)"
if [ -z "${SBATCH_BIN}" ]; then
  echo "sbatch not found; PATH=${PATH}" >&2
  exit 127
fi
"${SBATCH_BIN}" \
  --job-name=${script_name:4:-3} \
  --output="${WATCH_FOLDER}/%x_%j.log" \
  --open-mode=append \
  --get-user-env \
  --partition="${GPU_PARTITION}" \
  --constraint="${GPU_CONSTRAINT}" \
  --time=960:00:00 \
  --mem=${JOB_MEM} \
  --nodes=1 \
  --ntasks-per-node=${NUM_VISIBLE_DEVICES} \
  --gres=gpu:${NUM_VISIBLE_DEVICES} \
  --mail-user=${USERNAME}@cornell.edu \
  --mail-type=ALL \
  --requeue \
  --exclude="${GPU_EXCLUDE}" \
  "${NODELIST_ARGS[@]}" \
  --export="ALL,NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES},RUN_DIR=${RUN_DIR},DATA_DIR=${DATA_DIR},REPO_ROOT=${REPO_ROOT}" \
  ${script_full_path} "${@:2}"
