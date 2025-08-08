#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# TODO: FOR REBUTTAL
RUN_DIR="${RUN_DIR}/rebuttal"

########### AR
#for N in 16 32; do
#  MODEL_PATH="${RUN_DIR}/wmt_block_lr3e-4_bsz128_warm1000ba_layers${N}_hidden512_inter1536_ar_reinit"
#  BLOCK_SIZE=1
#  KV_CACHING=true
#  ALIGN_INPUTS_TO_BLOCKS=true

########## MDLM
for N in -1; do
  MODEL_PATH="${RUN_DIR}/wmt_block_lr3e-4_bsz128_warm1000ba_layers32_hidden512_inter1536_mdlm_reinit"
  BLOCK_SIZE=32
  KV_CACHING=false
  ALIGN_INPUTS_TO_BLOCKS=false

########### BD3LM
#for N in 12 16; do
#  MODEL_PATH="${RUN_DIR}/wmt_block4_lr3e-4_bsz128_warm1000ba_layers${N}_hidden512_inter1536_bd3lm_reinit"
#  BLOCK_SIZE=4
#  KV_CACHING=true
#  ALIGN_INPUTS_TO_BLOCKS=true

########### E2D2
#for N in -1; do
#  MODEL_PATH="${RUN_DIR}/wmt_block4_lr3e-4_bsz128_warm1000ba_enc28_dec4_hidden512_inter1536_e2d2_reinit-encoder_reinit-decoder"
#  BLOCK_SIZE=4
#  KV_CACHING=true
#  ALIGN_INPUTS_TO_BLOCKS=false

L=256
T=${BLOCK_SIZE}
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise" #"predict_and_noise" "posterior"
FIRST_HITTING=true
CONFIDENCE_BASED_NOISING=true
MAX_LENGTH=1024
CKPT="latest"
USE_EMA=true

OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_L${L}_block_size${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hitting${FIRST_HITTING}-confidence_based_noising${CONFIDENCE_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}"
PORT=29501
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/seq2seq_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/seq2seq@task=wmt \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  +ckpt_file="${CKPT}-rank0.pt" \
  +load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-0.6B-Base" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${MAX_LENGTH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.num_steps=${T} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs.stopping_criteria=null \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  +throughput_run=true
done
