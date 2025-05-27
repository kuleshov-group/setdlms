#!/bin/bash
#SBATCH -J gsm8k_llama_e2d2              # Job name
#SBATCH -o ../watch_folder/%x_%j.out  # Output file (%j expands to jobID)
#SBATCH -e ../watch_folder/%x_%j.err  # Error file (%j expands to jobID)
#SBATCH --get-user-env                # Retrieve the users login environment
#SBATCH --partition=kuleshov,gpu      # Request partition
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --mem=64000                   # Server memory requested (per node)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH --mail-user=yzs2@cornell.edu  # Email
#SBATCH --mail-type=END               # Request status by email

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
export RUN_DIR="/share/kuleshov/yzs2/runs/dllm-dev"
source ${script_full_path}
