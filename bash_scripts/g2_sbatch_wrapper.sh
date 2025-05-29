#!/bin/bash

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

NUM_VISIBLE_DEVICES=8
RUN_DIR="/share/kuleshov/yzs2/runs/dllm-dev"
sbatch \
  --job-name=${script_name:4:-3} \
  --output="../watch_folder/%x_%j.log" \
  --open-mode=append \
  --get-user-env \
  --partition=kuleshov,gpu \
  --constraint="[a100|a6000|a5000|3090]" \
  --time=960:00:00 \
  --mem=64000 \
  --nodes=1 \
  --ntasks-per-node=${NUM_VISIBLE_DEVICES} \
  --gres=gpu:${NUM_VISIBLE_DEVICES} \
  --mail-user=yzs2@cornell.edu \
  --mail-type=END \
  --requeue \
  --exclude=brandal \
  --export="ALL,NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES},RUN_DIR=${RUN_DIR}" \
  ${script_full_path}
