#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block16_ft_v5"
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block4_ft_v5"
MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block8_ft_v3"
# MODEL_PATH="kuleshov-group/mdlm-owt"
# MODEL_PATH="outputs/<PATH_TO_SAVED_MODEL_DIR>"
REVISION=null

for EVAL_DATASET in "lm1b_eval_gpt2"; do #"ptb_eval" "wikitext2_eval" "lm1b_eval" "lambada_eval" "ag_news_eval" "scientific_papers_pubmed_eval" "scientific_papers_arxiv_eval"; do
  BLOCK_SIZE=1024  # TODO: Change as needed
  BATCH_SIZE=32
  PRETRAINED_MODEL_NAME_OR_PATH="gpt2"  # TODO: Change as needed
  CKPT_FILE="best-rank0.pt"
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
    gen_kwargs=null \
    +compile_backbone=true \
    +model_config_overrides.attn_backend=sdpa \
    +model_config_overrides.backbone_config.attn_backend=sdpa
done
