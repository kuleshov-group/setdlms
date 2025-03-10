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

# Setup HF cache
# shellcheck disable=SC1091
export HF_HOME="${PWD}/.hf_cache"
echo "HuggingFace cache set to '${HF_HOME}'."


# Setup W&B
# shellcheck disable=SC1091
source "/home/$(whoami)/setup_discdiff.sh"
echo "Logging into W&B as '${WANDB_ENTITY}'."

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}:${HF_HOME}/modules"

# HF Login
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

# Enforce verbose Hydra error logging
export HYDRA_FULL_ERROR=1
