#!/bin/bash

<<comment
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./bash_wrapper.sh <SHELL_SCRIPT>
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
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_VISIBLE_DEVICES="${NUM_VISIBLE_DEVICES:-1}"
else
  NUM_VISIBLE_DEVICES=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
fi
export NUM_VISIBLE_DEVICES
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/runs}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
mkdir -p "${RUN_DIR}"
mkdir -p "${DATA_DIR}"
export REPO_ROOT
export RUN_DIR
export DATA_DIR
source "${script_full_path}"
