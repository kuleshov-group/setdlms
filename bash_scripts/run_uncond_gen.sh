#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# TODO: Uncomment a model and run

######### AR
#PROMPT_TEXT="Translation: "
#KV_CACHING=true
#ALIGN_INPUTS_TO_BLOCKS=true
#BLOCK_SIZE=1
#MODEL_PATH="${RUN_DIR}/<PATH_TO_AR_SAVED_MODEL_DIR>"

########### MDLM
#PROMPT_TEXT=null
#KV_CACHING=false
#ALIGN_INPUTS_TO_BLOCKS=false
#BLOCK_SIZE=32
#MODEL_PATH="${RUN_DIR}/<PATH_TO_MDLM_SAVED_MODEL_DIR>"

########### BD3LM
#PROMPT_TEXT=null
#KV_CACHING=true
#ALIGN_INPUTS_TO_BLOCKS=true
#BLOCK_SIZE=4
#MODEL_PATH="${RUN_DIR}/<PATH_TO_BD3LM_SAVED_MODEL_DIR>"

######### E2D2
PROMPT_TEXT=null
BLOCK_SIZE=127
MAX_WINDOW_SIZE=16
MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired16_vlambda"
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=false

# BLOCK_SIZE=16
# MAX_WINDOW_SIZE=16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_mdlm_adaln_cleanbos_antithetic_hparam_v2"
# KV_CACHING=false
# ALIGN_INPUTS_TO_BLOCKS=false

OUTPUT_DIR="output/"
REVISION=null
mkdir -p ${OUTPUT_DIR}

L=127
T=${BLOCK_SIZE}
DO_SAMPLE=false
FIRST_HITTING=false
CONFIDENCE_BASED_NOISING=false
CONFIDENCE_MARGIN_BASED_NOISING=false
FIRST_HITTING=false
CONFIDENCE_THRESHOLD=1e6
MAX_LENGTH=128
CKPT="best"
USE_EMA=true
BATCH_SIZE=256

echo "MODEL_PATH: ${MODEL_PATH}"

OUTPUT_PATH="${OUTPUT_DIR}/L-${L}-block_size-${BLOCK_SIZE}-T${T}-do_sample-${DO_SAMPLE}-first_hitting-${FIRST_HITTING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-ckpt${CKPT}-ema${USE_EMA}"
PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/uncond_gen.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval@task=likelihood \
  +dataset@task.eval_dataset=lm1b_eval \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file="${CKPT}-rank0.pt" \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="bert-base-uncased" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation_config.max_window_size=${MAX_WINDOW_SIZE} \
  generation/stopping_criteria@stopping_criteria_list='[gsm8k_regex_stopping_criteria]' \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  +generation_config.save_likelihoods=true \
  gen_kwargs.return_dict_in_generate=true \
  +collator@task.collator=denoising \
  task.collator.global_batch_size=${BATCH_SIZE} \
  task.collator.max_length=null \
  task.collator.restricted_t_range=null \
  task.collator.sampling_eps=1e-3 \
  task.collator.antithetic_sampling=false \
  batch_size=${BATCH_SIZE}