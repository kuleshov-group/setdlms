#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

QWEN_MODEL="Qwen/Qwen3-1.7B-Base"


########### AR
#for N in 1; do
#  MODEL_PATH="${RUN_DIR}/gsm8k-0shot_lr1e-5_bsz96_warm1000ba_alphaf0.0_max-dur20000ba_amp_bf16_layers28_ar_FT2B_repro_with-ema_v2"
#  BLOCK_SIZE=1
#  KV_CACHING=true
#  ALIGN_INPUTS_TO_BLOCKS=true

########### MDLM
#for N in 1; do
#  MODEL_PATH="${RUN_DIR}/gsm8k_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers-1_mdlm_repro"
#  BLOCK_SIZE=64
#  KV_CACHING=false
#  ALIGN_INPUTS_TO_BLOCKS=false

########### BD3LM
#for N in 17 21 28; do
#  MODEL_PATH="${RUN_DIR}/gsm8k_block4_evalblock4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers${N}_bd3lm_repro"
#  BLOCK_SIZE=4
#  KV_CACHING=true
#  ALIGN_INPUTS_TO_BLOCKS=true

########## E2D2
for N in 14 21 24 26; do
  MODEL_PATH="${RUN_DIR}/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec${N}_e2d2_repro_with-ema_v2_tie-weights"
  BLOCK_SIZE=4
  KV_CACHING=true
  ALIGN_INPUTS_TO_BLOCKS=true

  OUTPUT_DIR="${MODEL_PATH}/lm_eval_harness_output"
  REVISION=null

  mkdir -p ${OUTPUT_DIR}
  L=256
  T=${BLOCK_SIZE}
  DO_SAMPLE=false
  SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
  FIRST_HITTING=true
  CONFIDENCE_BASED_NOISING=true
  CONFIDENCE_MARGIN_BASED_NOISING=false
  CKPT="best"
  USE_EMA=true

  OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_${NUM_FEW_SHOT}shot_L${L}_block_size${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hitting${FIRST_HITTING}-confidence_based_noising${CONFIDENCE_BASED_NOISING}-confidence_margin_based_noising${CONFIDENCE_MARGIN_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}"
  mkdir -p ${OUTPUT_PATH}

  accelerate launch scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=gsm8k \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  task.model.ckpt_file="${CKPT}-rank0.pt" \
  task.model.load_ema_weights=${USE_EMA} \
  +task.model.throughput_run=true \
  tokenizer.pretrained_model_name_or_path=${QWEN_MODEL} \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.num_steps=${T} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=1.1 \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs.stopping_criteria=null
done
