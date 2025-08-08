#!/bin/bash

<<comment
#  Usage:
cd bash_scripts/
./g2_sbatch_wrapper.sh <SHELL_SCRIPT>
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

WATCH_FOLDER=$(realpath "../watch_folder")
mkdir -p ${WATCH_FOLDER}
USERNAME=$(whoami)
NUM_VISIBLE_DEVICES=1
RUN_DIR="/share/kuleshov/${USERNAME}/runs/dllm-dev/rebuttal"
DATA_DIR="/share/kuleshov/ma2238/dllm-data"
mkdir -p ${RUN_DIR}
mkdir -p ${DATA_DIR}
sbatch \
  --job-name=${script_name:4:-3} \
  --output="${WATCH_FOLDER}/%x_%j.log" \
  --open-mode=append \
  --get-user-env \
  --partition=gpu \
  --constraint="[h100]" \
  --time=960:00:00 \
  --mem=128000 \
  --nodes=1 \
  --ntasks-per-node=${NUM_VISIBLE_DEVICES} \
  --gres=gpu:${NUM_VISIBLE_DEVICES} \
  --mail-user=${USERNAME}@cornell.edu \
  --mail-type=END \
  --requeue \
  --exclude="brandal,kuleshov-compute-02" \
  --export="ALL,NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES},RUN_DIR=${RUN_DIR},DATA_DIR=${DATA_DIR}" \
  ${script_full_path}
