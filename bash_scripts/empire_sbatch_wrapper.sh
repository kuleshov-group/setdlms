#!/bin/bash

<<comment
#  Usage:
cd bash_scripts/
./empire_sbatch_wrapper.sh <SHELL_SCRIPT>
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

PARTITION="cornell"
WATCH_FOLDER=$(realpath "../watch_folder")
mkdir -p ${WATCH_FOLDER}
USERNAME=$(whoami)
NODES=1
NUM_DEVICES_PER_NODE=8
NUM_VISIBLE_DEVICES=$(( NODES * NUM_DEVICES_PER_NODE ))
RUN_DIR="/mnt/lustre/cornell/${USERNAME}/runs/dllm-dev"
DATA_DIR="/mnt/lustre/cornell/${USERNAME}/data"
mkdir -p ${RUN_DIR}
mkdir -p ${DATA_DIR}
sbatch \
  --job-name=${script_name:4:-3} \
  --output="${WATCH_FOLDER}/%x_%j.log" \
  --open-mode=append \
  --get-user-env \
  --partition=${PARTITION} \
  --account=cornell \
  --time=100:00:00 \
  --mem=64000 \
  --nodes=${NODES} \
  --ntasks-per-node=${NUM_DEVICES_PER_NODE} \
  --gres=gpu:${NUM_DEVICES_PER_NODE} \
  --mail-user=${USERNAME}@cornell.edu \
  --mail-type=ALL \
  --requeue \
  --export="ALL,NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES},RUN_DIR=${RUN_DIR},DATA_DIR=${DATA_DIR}" \
  ${script_full_path}
