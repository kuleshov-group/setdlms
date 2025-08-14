#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

QWEN_MODEL="Qwen/Qwen3-1.7B-Base"
NUM_FEW_SHOT=0

########### AR
#for N in false true; do
#  MODEL_PATH="${RUN_DIR}/gsm8k-0shot_lr1e-5_bsz96_warm1000ba_alphaf0.0_max-dur20000ba_amp_bf16_layers28_ar"
#  BLOCK_SIZE=1
#  KV_CACHING=true
#  ALIGN_INPUTS_TO_BLOCKS=true
#  USE_EMA=${N}

########### MDLM
#for N in 1; do
#  MODEL_PATH="${RUN_DIR}/gsm8k-0shot_block_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers-1_mdlm"
#  BLOCK_SIZE=64
#  KV_CACHING=false
#  ALIGN_INPUTS_TO_BLOCKS=true
#  USE_EMA=false

########### BD3LM
#for N in 17 21 28; do
#  MODEL_PATH="${RUN_DIR}/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers${N}_bd3lm"
#  BLOCK_SIZE=4
#  KV_CACHING=true
#  ALIGN_INPUTS_TO_BLOCKS=true
#  USE_EMA=true

######### E2D2
#for N in 14 21 24 26 27; do
for N in 14; do
#  MODEL_PATH="${RUN_DIR}/gsm8k-0shot_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec${N}_e2d2_tie-weights"
  MODEL_PATH="${RUN_DIR}/gsm8k-0shot_block8_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec14_e2d2_tie-weights"
  BLOCK_SIZE=8
  KV_CACHING=true
  ALIGN_INPUTS_TO_BLOCKS=false
  USE_EMA=true

  OUTPUT_DIR="${MODEL_PATH}/lm_eval_harness_output"
  REVISION=null

  mkdir -p ${OUTPUT_DIR}
  L=512
  DO_SAMPLE=false
  SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
  T=8 #${BLOCK_SIZE}
  FIRST_HITTING=true
  CONFIDENCE_BASED_NOISING=true
  CONFIDENCE_MARGIN_BASED_NOISING=false
  COHERENCE_BASED_NOISING=false
  CONFIDENCE_THRESHOLD=1e6
  CKPT="best"

  OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_${NUM_FEW_SHOT}shot_L${L}_block${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hit${FIRST_HITTING}-conf_noise${CONFIDENCE_BASED_NOISING}-conf_margin_noise${CONFIDENCE_MARGIN_BASED_NOISING}-coh_noise${COHERENCE_BASED_NOISING}-conf_thold${CONFIDENCE_THRESHOLD}-align_to_blocks${ALIGN_INPUTS_TO_BLOCKS}"
  mkdir -p ${OUTPUT_PATH}

  accelerate launch scripts/eval/harness_eval.py \
    hydra.output_subdir=null \
    hydra.run.dir="${PWD}" \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    +eval/lm_eval_harness@task=gsm8k \
    task.num_fewshot=${NUM_FEW_SHOT} \
    pretrained_model_name_or_path=${MODEL_PATH} \
    pretrained_model_revision=${REVISION} \
    task.model.ckpt_file="${CKPT}-rank0.pt" \
    task.model.load_ema_weights=${USE_EMA} \
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
    +generation_config.coherence_based_noising=${COHERENCE_BASED_NOISING} \
    generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD} \
    generation_config.use_cache=${KV_CACHING} \
    generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
    ~generation/logits_processor@logits_processor_list \
    gen_kwargs.logits_processor=null \
    generation/stopping_criteria@stopping_criteria_list='[eos_token_criteria,max_length_criteria,gsm8k_regex_stopping_criteria]'
#    ~generation/stopping_criteria@stopping_criteria_list \
#    gen_kwargs.stopping_criteria=null
done
