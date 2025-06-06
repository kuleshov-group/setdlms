#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091
source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
    conda activate dllm-dev
 elif [[ "${CONDA_PREFIX}" != *"/dllm-dev" ]]; then
  conda deactivate
  conda activate dllm-dev
fi

# W&B / HF Setup
source "${HOME}/setup_discdiff.sh"
export HF_HOME="${PWD}/.hf_cache"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}:${HF_HOME}/modules"

# Enforce verbose Hydra error logging
export HYDRA_FULL_ERROR=1
# export TMPDIR="${PWD}/.tmp"  # TODO: currently this is causing OSErrors

export NCCL_P2P_LEVEL=NVL
