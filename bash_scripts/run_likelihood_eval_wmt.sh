#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

OUTPUT_DIR="/share/kuleshov/ma2238/runs/dllm-dev"
# MODEL_NAME="wmt_block4_lr3e-4_bsz128_warm1000ba_layers16_hidden512_inter1536_aoarm_efficient_v2"
MODEL_NAME="wmt_block1_lr3e-4_bsz128_warm1000ba_layers16_hidden512_inter1536_bd3lm"
# MODEL_NAME="wmt_block32_lr3e-4_bsz128_warm1000ba_layers16_hidden512_inter1536_aoarm_efficient"
# MODEL_NAME="wmt_block32_lr3e-4_bsz128_warm1000ba_layers16_hidden512_inter1536_bd3lm_baseline"
# MODEL_NAME="wmt_block32_lr3e-4_bsz128_warm1000ba_layers16_hidden512_inter1536_aoarm_efficient_v5"
MODEL_PATH="${OUTPUT_DIR}/${MODEL_NAME}"
REVISION=null

for EVAL_DATASET in "wmt_eval"; do
BLOCK_SIZE=4
BATCH_SIZE=16
PRETRAINED_MODEL_NAME_OR_PATH="Qwen/Qwen3-0.6B-Base"
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
  seed=1 \
  batch_size=${BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  task.eval_dataloader.batch_size=${BATCH_SIZE} \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  tokenizer.pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  output_path=/share/kuleshov/ma2238/dllm-dev-new/dllm-dev/outputs/${MODEL_NAME} \
  +collator@task.collator=denoising \
  +model.config.length=256 \
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
  task.collator.global_batch_size=${BATCH_SIZE}
done
