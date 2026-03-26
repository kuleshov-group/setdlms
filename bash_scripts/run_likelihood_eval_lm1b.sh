#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# setdlm s <= 8
MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired4_max128_v5"
CKPT_FILE=ep72-ba1000000-rank0.pt

# setdlm s <= 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired8_max128_v5"
# CKPT_FILE="best-rank0.pt"

# setdlm s <= 32
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired16_vlambda"
# CKPT_FILE="best-rank0.pt"

REVISION=null

BLOCK_SIZE=128
BATCH_SIZE=16
PRETRAINED_MODEL_NAME_OR_PATH=null  # TODO: Change as needed
USE_EMA=false

composer -n ${NUM_VISIBLE_DEVICES} scripts/eval/likelihood_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval@task=likelihood \
  +dataset@task.eval_dataset=lm1b_eval \
  task.load_ema_weights=${USE_EMA} \
  task.ckpt_file=${CKPT_FILE} \
  seed=1 \
  batch_size=${BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  task.eval_dataloader.batch_size=8 \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  tokenizer.pretrained_model_name_or_path=bert-base-uncased \
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
