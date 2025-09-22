#!/bin/bash

<<comment
#  Usage:
cd bash_scripts/
source empire_run_wrapper.sh <SHELL_SCRIPT>
comment


if [ -z "$1" ]; then
  echo "Usage: $0 script_name"
fi

script_name="$1"
if [[ "$script_name" != *.sh ]]; then
  script_name="${script_name}.sh"
fi

# Construct the full path
script_full_path=$(realpath "./${script_name}")

# Check if the file exists in the directory
if [ ! -e "${script_full_path}" ]; then
  echo "Script '$script_full_path' not found."
fi

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  NUM_VISIBLE_DEVICES=${SLURM_GPUS_ON_NODE}
else
  NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
fi
export NUM_VISIBLE_DEVICES
DATA_DIR="/mnt/lustre/cornell/$(whoami)/data"
RUN_DIR="/mnt/lustre/cornell/$(whoami)/runs/dllm-dev"
mkdir -p ${RUN_DIR}
mkdir -p ${DATA_DIR}
export RUN_DIR
export DATA_DIR
source ${script_full_path}
