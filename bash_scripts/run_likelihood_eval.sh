#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

MODEL_PATH="${RUN_DIR}/rebuttal/gsm8k-bs96-keep1-max20000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-ar_qweb2B_v1"
REVISION=null

EVAL_DATASET="gsm8k_eval"
BLOCK_SIZE=null
BATCH_SIZE=32
PRETRAINED_MODEL_NAME_OR_PATH="Qwen/Qwen3-1.7B-Base"
CKPT_FILE="best-rank0.pt"
USE_EMA=false

composer -n ${NUM_VISIBLE_DEVICES} scripts/eval/likelihood_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval@task=likelihood \
  +dataset@task.eval_dataset=${EVAL_DATASET} \
  task.load_ema_weights=${USE_EMA} \
  task.ckpt_file=${CKPT_FILE} \
  task.eval_dataset.max_length=null \
  seed=1 \
  batch_size=${BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  task.eval_dataloader.batch_size=8 \
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
  gen_kwargs=null
