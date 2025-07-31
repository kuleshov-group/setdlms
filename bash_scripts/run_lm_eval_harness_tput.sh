#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

QWEN_MODEL="Qwen/Qwen3-1.7B-Base"

#for T in 2 8 16; do
for T in 4; do
#  for d in "gsm8k_block4_evalblock4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers17_bd3lm_2B-FT" "gsm8k_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec14_e2d2_2BFT_tie-weights" "gsm8k_block4_evalblock4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers21_bd3lm_2BFT" "gsm8k_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec21_e2d2_2B-FT_kv_tie-weights" "gsm8k_block4_evalblock4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_bd3lm_2BFT" "gsm8k_block4_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec24_e2d2_2B-FT_kv_tie-weights"; do
#  MODEL_PATH="${RUN_DIR}/gsm8k_block${T}_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur30000ba_amp_bf16_enc28_TOPdec14_e2d2_2B-FT_share_kv_tie-weights"
#  MODEL_PATH="${RUN_DIR}/${d}"
  MODEL_PATH="${RUN_DIR}/gsm8k_lr1e-5_bsz1_warm10ba_alphaf0.5_max-dur30000ba_amp_bf16_layers28_ar_FT2B"
  OUTPUT_DIR="${MODEL_PATH}/lm_eval_harness_output"
  REVISION=null

  mkdir -p ${OUTPUT_DIR}
  L=256
  BLOCK_SIZE=${T}
#  T=${BLOCK_SIZE}
  DO_SAMPLE=false
  SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
  FIRST_HITTING=true
  CONFIDENCE_BASED_NOISING=true
  CONFIDENCE_MARGIN_BASED_NOISING=false
  ALIGN_INPUTS_TO_BLOCKS=true  # "true" for BD3LM / E2D2, "false" for MDLM
  KV_CACHING=true  # "true" for BD3LM / E2D2, "false" for MDLM
  CKPT_FILE="dummy-rank0.pt"
  USE_EMA=true

  OUTPUT_PATH="${OUTPUT_DIR}/L${L}_block_size${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hitting${FIRST_HITTING}-confidence_based_noising${CONFIDENCE_BASED_NOISING}-confidence_margin_based_noising${CONFIDENCE_MARGIN_BASED_NOISING}-align_inputs_to_blocks${ALIGN_INPUTS_TO_BLOCKS}"
  mkdir -p ${OUTPUT_PATH}

  accelerate launch scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=gsm8k \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  task.model.ckpt_file=${CKPT_FILE} \
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
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  ~generation/stopping_criteria@stopping_criteria_list \
  gen_kwargs.stopping_criteria=null
done
#done
