#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091
if [ -n "${CONDA_SHELL:-}" ] && [ -f "${CONDA_SHELL}" ]; then
    source "${CONDA_SHELL}"
elif [ -n "${CONDA_EXE:-}" ]; then
    _conda_root="$(dirname "$(dirname "${CONDA_EXE}")")"
    if [ -f "${_conda_root}/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "${_conda_root}/etc/profile.d/conda.sh"
    fi
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Unable to initialize conda shell hooks." >&2
    return 1 2>/dev/null || exit 1
fi

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
