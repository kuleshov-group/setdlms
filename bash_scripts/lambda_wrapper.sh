#!/bin/bash

<<comment
#  Usage:
cd bash_scripts/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./lambda_wrapper.sh <SHELL_SCRIPT>
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
NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
export NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES}
RUN_DIR="/home/ubuntu/runs/dllm-dev"
DATA_DIR="/home/ubuntu/data/"
mkdir -p ${RUN_DIR}
mkdir -p ${DATA_DIR}
export RUN_DIR
export DATA_DIR
source ${script_full_path}
