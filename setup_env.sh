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

DLLM_CONDA_ENV="${DLLM_CONDA_ENV:-dllm-dev}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX}")" != "${DLLM_CONDA_ENV}" ]; then
    conda activate "${DLLM_CONDA_ENV}"
fi

if [ -n "${DLLM_PRIVATE_ENV:-}" ]; then
    if [ ! -f "${DLLM_PRIVATE_ENV}" ]; then
        echo "DLLM_PRIVATE_ENV points to '${DLLM_PRIVATE_ENV}', but that file does not exist." >&2
        return 1 2>/dev/null || exit 1
    fi
    # shellcheck source=/dev/null
    source "${DLLM_PRIVATE_ENV}"
fi

export HF_HOME="${HF_HOME:-${PWD}/.hf_cache}"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}:${HF_HOME}/modules${PYTHONPATH:+:${PYTHONPATH}}"

# Enforce verbose Hydra error logging
export HYDRA_FULL_ERROR=1

export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

# Use a per-job node-local temp directory under Slurm. Composer creates
# temporary rank-log directories during distributed eval; shared repo-local
# temp dirs can leave files behind and make otherwise successful eval jobs
# exit nonzero during cleanup.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _dllm_tmp_user="${USER:-$(whoami)}"
    export TMPDIR="${DLLM_SLURM_TMPDIR_BASE:-/tmp}/${_dllm_tmp_user}/dllm-${SLURM_JOB_ID}"
    mkdir -p "${TMPDIR}"
fi
