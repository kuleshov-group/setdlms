#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# setdlm s <= 8
MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block4_vscratch"
CKPT_FILE="ep17-ba300000-rank0.pt"
BLOCK_SIZE=1024

# setdlm s <= 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block8_ft_v3"
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block8_vscratch"
# CKPT_FILE="best-rank0.pt"

# setdlm s <= 32
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block16_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=1024

# setdlm s = 1024
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block1024_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=1024

# ar
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-ar-noeos-v4-1"
# CKPT_FILE="20-300000.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# BLOCK_SIZE=1

# mdlm
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-mdlm-noeos-v4"
# CKPT_FILE="18-300000.ckpt"
# # CKPT_FILE="best.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# # MODEL_PATH="kuleshov-group/mdlm-owt"
# BLOCK_SIZE=1024

# sedd
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-sedd-noeos-v4"
# CKPT_FILE="18-300000.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# BLOCK_SIZE=1024

REVISION=null

for EVAL_DATASET in "owt_eval_gpt2" "ptb_eval" "wikitext2_eval" "lm1b_eval_gpt2" "lambada_eval" "ag_news_eval" "scientific_papers_pubmed_eval" "scientific_papers_arxiv_eval"; do
  BATCH_SIZE=16
  PRETRAINED_MODEL_NAME_OR_PATH="gpt2"  # TODO: Change as needed
  USE_EMA=true
  echo "Evaluating ${EVAL_DATASET} with model ${MODEL_PATH}"

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
    output_path=null \
    +collator@task.collator=denoising \
    task.collator.global_batch_size=${BATCH_SIZE} \
    task.collator.max_length=null \
    task.collator.restricted_t_range=null \
    task.collator.sampling_eps=1e-3 \
    task.collator.antithetic_sampling=true \
    +metrics@task.metrics='[loss,nll,bpd,perplexity]' \
    +composer/trainer@task.trainer=eval_trainer \
    ~generation@generation_config \
    ~generation/logits_processor@logits_processor_list \
    ~generation/stopping_criteria@stopping_criteria_list \
    gen_kwargs=null \
    +compile_backbone=true \
    +model_config_overrides.mdlm_loss_scale=false \
    +model_config_overrides.keep_clean_bos=true
done
