#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt4_max1024_distill_again_v2"

# MODEL_PATH="outputs/<PATH_TO_SAVED_MODEL_DIR>"
REVISION=null

EVAL_DATASET="gsm8k_eval_distill"
BLOCK_SIZE=1024  # TODO: Change as needed
BATCH_SIZE=1
PRETRAINED_MODEL_NAME_OR_PATH="Qwen/Qwen3-1.7B-Base"  # TODO: Change as needed
CKPT_FILE="best-rank0.pt"
USE_EMA=true
SEED=1

composer -n ${NUM_VISIBLE_DEVICES} scripts/eval/likelihood_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval@task=likelihood \
  +dataset@task.eval_dataset=${EVAL_DATASET} \
  task.load_ema_weights=${USE_EMA} \
  task.ckpt_file=${CKPT_FILE} \
  seed=${SEED} \
  batch_size=${BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  task.eval_dataloader.batch_size=1 \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  tokenizer.pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  output_path=null \
  +collator@task.collator=denoising \
  task.collator.global_batch_size=${BATCH_SIZE} \
  task.collator.max_length=null \
  task.collator.restricted_t_range=null \
  task.collator.sampling_eps=1e-3 \
  task.collator.antithetic_sampling=false \
  +metrics@task.metrics='[loss,nll,bpd,perplexity]' \
  +composer/trainer@task.trainer=eval_trainer \
  ~generation@generation_config \
  ~generation/logits_processor@logits_processor_list \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs=null \
  +model_config_overrides.block_size=${BLOCK_SIZE} \
  +model_config_overrides.eval_block_size=${BLOCK_SIZE} \
  +model_config_overrides.eval_nll=false