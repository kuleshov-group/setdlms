#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# TODO: Uncomment a model and run

# base model
# MODEL_PATH="Qwen/Qwen3-1.7B-Base"

# setdlm s <= 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt4_max1024_distill_again_v2"
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=4
# ALIGN_INPUTS_TO_BLOCKS=false

# setdlm s <= 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt8_max1024_distill_v23"
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=8
# ALIGN_INPUTS_TO_BLOCKS=false

# setdlm s <= 32
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt16_max1024_distill_again_v2"
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=16
# ALIGN_INPUTS_TO_BLOCKS=false

# ar
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_ar_distill_v5"
# KV_CACHING=true

# mdlm
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_mdlm_distill_v5"
# KV_CACHING=false
# BLOCK_SIZE=32
# MAX_WINDOW_SIZE=${BLOCK_SIZE}
# ALIGN_INPUTS_TO_BLOCKS=true

# bd3lm s = 4
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-shot_block4_lr1e-5_bsz1_warm100ba_max-dur75000ba_amp_bf16_layers28_bd3lm_distill_anneal0ba_maxbs4_v10"
# KV_CACHING=true
# BLOCK_SIZE=4
# MAX_WINDOW_SIZE=${BLOCK_SIZE}
# ALIGN_INPUTS_TO_BLOCKS=true

# bd3lm s = 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-shot_block8_lr1e-5_bsz1_warm100ba_max-dur75000ba_amp_bf16_layers28_bd3lm_distill_anneal0ba_maxbs8_v10"
# KV_CACHING=true
# BLOCK_SIZE=8
# MAX_WINDOW_SIZE=${BLOCK_SIZE}
# ALIGN_INPUTS_TO_BLOCKS=true

# bd3lm s = 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-shot_block16_lr1e-5_bsz1_warm100ba_max-dur75000ba_amp_bf16_layers28_bd3lm_distill_anneal0ba_maxbs16_v10"
# KV_CACHING=true
# BLOCK_SIZE=16
# MAX_WINDOW_SIZE=${BLOCK_SIZE}
# ALIGN_INPUTS_TO_BLOCKS=true

echo "MODEL_PATH: ${MODEL_PATH}"

USE_EMA=true
OUTPUT_DIR="outputs/${MODEL_PATH}/lm_eval_harness_output"
REVISION=null

T=${BLOCK_SIZE}
L=1024
DO_SAMPLE=false
SAMPLING_STRATEGY="predict_and_noise"  # "predict_and_noise" or "posterior"
FIRST_HITTING=false
CONFIDENCE_BASED_NOISING=false
CONFIDENCE_MARGIN_BASED_NOISING=false
CONFIDENCE_THRESHOLD=1e6 # TODO: Change as needed
CKPT="best"
LINEAR_UNMASKING=true

echo "CONFIDENCE_THRESHOLD: ${CONFIDENCE_THRESHOLD}"
echo "T: ${T}"
echo "LINEAR_UNMASKING: ${LINEAR_UNMASKING}"
echo "DO_SAMPLE: ${DO_SAMPLE}"
echo "SAMPLING_STRATEGY: ${SAMPLING_STRATEGY}"
echo "FIRST_HITTING: ${FIRST_HITTING}"
echo "CONFIDENCE_BASED_NOISING: ${CONFIDENCE_BASED_NOISING}"
echo "CONFIDENCE_MARGIN_BASED_NOISING: ${CONFIDENCE_MARGIN_BASED_NOISING}"
echo "ALIGN_INPUTS_TO_BLOCKS: ${ALIGN_INPUTS_TO_BLOCKS}"

OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_L${L}_block${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hit${FIRST_HITTING}-conf_noise${CONFIDENCE_BASED_NOISING}-conf_margin_noise${CONFIDENCE_MARGIN_BASED_NOISING}-conf_thold${CONFIDENCE_THRESHOLD}-align_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-max_window_size${MAX_WINDOW_SIZE}"
mkdir -p ${OUTPUT_PATH}

PORT=$((RANDOM % 10000 + 29500))
torchrun --nproc_per_node ${NUM_VISIBLE_DEVICES} --master_port=${PORT} scripts/eval/harness_eval.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  +eval/lm_eval_harness@task=gsm8k \
  pretrained_model_name_or_path=${MODEL_PATH} \
  pretrained_model_revision=${REVISION} \
  task.model.ckpt_file="${CKPT}-rank0.pt" \
  task.model.load_ema_weights=${USE_EMA} \
  tokenizer.pretrained_model_name_or_path="Qwen/Qwen3-1.7B-Base" \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${L} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  generation_config.do_sample=${DO_SAMPLE} \
  generation_config.sampling_strategy=${SAMPLING_STRATEGY} \
  generation_config.num_steps=${T} \
  generation_config.first_hitting=${FIRST_HITTING} \
  generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING} \
  generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING} \
  generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD} \
  generation_config.use_cache=${KV_CACHING} \
  generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS} \
  generation_config.max_window_size=${MAX_WINDOW_SIZE} \
  +generation_config.ar_caching=true \
  generation_config.linear_unmasking=${LINEAR_UNMASKING} \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  generation/stopping_criteria@stopping_criteria_list='[gsm8k_regex_stopping_criteria,repeating_token]'