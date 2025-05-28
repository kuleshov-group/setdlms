#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

#MODEL_PATH="${RUN_DIR}/gsm8k-block4-bs96-keeptop14-causalencfalse-max20000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen2B_tie_redo"
#MODEL_PATH="/share/kuleshov/yzs2/runs/dllm-dev/wmt-block4-bs128-keepbottomenc-1-keeptopdec14-causalencfalse-max10000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-e2d2_qwen600M_v2"
MODEL_PATH="yairschiff/wmt-e2d2-qwen600M"
REVISION=null
EVAL_DATASET="wmt_eval"
BLOCK_SIZE=4
BATCH_SIZE=1

composer -n ${NUM_VISIBLE_DEVICES} scripts/eval/likelihood_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval@task=likelihood \
  +dataset@task.eval_dataset=${EVAL_DATASET} \
  task.eval_dataset.max_length=null \
  seed=1 \
  batch_size=${BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B" \
  output_path=null \
  +collator@task.collator=denoising \
  task.collator.max_length=null \
  task.collator.restricted_t_range=null \
  task.collator.sampling_eps=1e-3 \
  task.collator.antithetic_sampling=true \
  +metrics@task.metrics='[loss,nll,bpd,perplexity]' \
  +composer/trainer@task.trainer=eval_trainer \
  ~generation@generation_config \
  ~generation/logits_processor@logits_processor_list \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs=null
