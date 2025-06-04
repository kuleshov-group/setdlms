#!/bin/bash

<<comment
#  Usage:
cd bash_scripts/
source g2_run_wrapper.sh <SHELL_SCRIPT>
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

export NUM_VISIBLE_DEVICES=${SLURM_GPUS_ON_NODE}
RUN_DIR="/mnt/home/$(whoami)/runs/dllm-dev"  # TODO: Point to scratch dir when available
export RUN_DIR
source ${script_full_path}
