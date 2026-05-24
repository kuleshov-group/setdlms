#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# setdlm s <= 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block4_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=1024
# MAX_BLOCK_SIZE=8
# COMPILE_BACKBONE=true

# setdlm s <= 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block8_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=1024
# MAX_BLOCK_SIZE=16
# COMPILE_BACKBONE=true

# setdlm s <= 32
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block16_vscratch"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=1024
# MAX_BLOCK_SIZE=32
# COMPILE_BACKBONE=true

# bd3lm s = 4
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block4_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=4
# COMPILE_BACKBONE=true

# bd3lm s = 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block8_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=8
# COMPILE_BACKBONE=true

# bd3lm s = 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block16_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
# CKPT_FILE="ep17-ba300000-rank0.pt"
# BLOCK_SIZE=16
# COMPILE_BACKBONE=true

# ar
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-ar-noeos-v4-1"
# CKPT_FILE="20-300000.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# BLOCK_SIZE=1
# COMPILE_BACKBONE=false

# mdlm
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-mdlm-noeos-v4"
# CKPT_FILE="18-300000.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# BLOCK_SIZE=1024
# COMPILE_BACKBONE=false

# sedd
# MODEL_PATH="/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-sedd-noeos-v4"
# CKPT_FILE="18-300000.ckpt"
# MODEL_PATH=${MODEL_PATH}/${CKPT_FILE}
# BLOCK_SIZE=1024
# COMPILE_BACKBONE=false

# esolm
MODEL_PATH="/share/kuleshov/ma2238/dllm-dev-new/dllm-dev/esolmb-alpha0-0d125-batchsplit-0d5-250k.ckpt"
CKPT_FILE=null
BLOCK_SIZE=1024
MAX_BLOCK_SIZE=1024
COMPILE_BACKBONE=false

TOKENIZER_PATH="gpt2"
USE_EMA=true
# This Composer eval script interprets `task.eval_dataloader.batch_size` per
# process. For EsoLM, changing the process count or per-process batch size
# changes the diffusion/sequential split on each rank, which noticeably moves
# small-dataset PPL. Keep the previous stable local defaults unless explicitly
# overridden.
EVAL_NUM_PROCESSES=${ESOLM_EVAL_NUM_PROCESSES:-${NUM_VISIBLE_DEVICES:-1}}
ESOLM_EVAL_BATCH_SIZE=${ESOLM_EVAL_BATCH_SIZE:-16}

REVISION=null
REQUIRE_REFUSION_SEMANTICS=false
REFUSION_LENGTH=null
for EVAL_DATASET in "ptb_eval" "wikitext2_eval" "lm1b_eval_gpt2" "lambada_eval" "ag_news_eval" "scientific_papers_pubmed_eval" "scientific_papers_arxiv_eval"; do
  BATCH_SIZE=${ESOLM_EVAL_BATCH_SIZE}
  echo "Evaluating ${EVAL_DATASET} with model ${MODEL_PATH}"

  REFUSION_ARGS=()
  if [ "${REQUIRE_REFUSION_SEMANTICS}" = true ]; then
    REFUSION_ARGS+=(
      task.require_refusion_semantics=true
    )
    if [ "${REFUSION_LENGTH}" != "null" ]; then
      REFUSION_ARGS+=(
        +model_config_overrides.length=${REFUSION_LENGTH}
      )
    fi
  fi

  composer -n ${EVAL_NUM_PROCESSES} scripts/eval/likelihood_eval.py \
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
    tokenizer.pretrained_model_name_or_path=${TOKENIZER_PATH} \
    output_path=null \
    +collator@task.collator=denoising \
    task.collator.global_batch_size=${BATCH_SIZE} \
    task.collator.max_length=null \
    task.collator.restricted_t_range=null \
    task.collator.sampling_eps=1e-3 \
    task.collator.antithetic_sampling=true \
    +metrics@task.metrics='[loss,nll,bpd,perplexity]' \
    +composer/trainer@task.trainer=eval_trainer \
    "${REFUSION_ARGS[@]}" \
    ~generation@generation_config \
    ~generation/logits_processor@logits_processor_list \
    ~generation/stopping_criteria@stopping_criteria_list \
    gen_kwargs=null \
    +compile_backbone=${COMPILE_BACKBONE} \
    +model_config_overrides.alpha_0=0.125 \
    +model_config_overrides.batch_split=0.5 \
    +model_config_overrides.sampling_eps=1e-3 \
    +model_config_overrides.antithetic_sampling=true \
    +model_config_overrides.diffusion_attn_mode=causal \
    +model_config_overrides.diffusion_shuffle=true \
    +model_config_overrides.sequential_attn_mode=causal \
    +model_config_overrides.sequential_shuffle=true \
    +model_config_overrides.loss_type=elbo \
    +model_config_overrides.keep_clean_bos=true \
    +model_config_overrides.mdlm_loss_scale=false \
    +model_config_overrides.attn_backend=sdpa \
    +model_config_overrides.backbone_config.attn_backend=sdpa # \
done

#     +model_config_overrides.noise_config.max_block_size=${MAX_BLOCK_SIZE}
