#!/bin/bash
# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# TODO: Uncomment a model and run

REQUIRE_REFUSION_SEMANTICS=false

# base model
# MODEL_PATH="Qwen/Qwen3-1.7B-Base"

# setdlm s <= 8
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt4_max1024_distill_again_v2"
# # MODEL_PATH="kuleshov-group/setdlm-gsm8k-smax8"
# KV_CACHING=true
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=4
# ALIGN_INPUTS_TO_BLOCKS=false

# setdlm s <= 16
# # MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt8_max1024_distill_v23"
# MODEL_PATH="kuleshov-group/setdlm-gsm8k-smax16"
# KV_CACHING=true
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=8
# ALIGN_INPUTS_TO_BLOCKS=false

# setdlm s <= 32
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt16_max1024_distill_again_v2"
# # MODEL_PATH="kuleshov-group/setdlm-gsm8k-smax32"
# KV_CACHING=true
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=32
# ALIGN_INPUTS_TO_BLOCKS=false

# ablation w = 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_setdlm_tgt16_knull_maxb16_sweep_v1"
# KV_CACHING=true
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=16
# ALIGN_INPUTS_TO_BLOCKS=false

# ablation w = 24
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_setdlm_tgt16_knull_maxb24_sweep_v1"
# KV_CACHING=true
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=16
# ALIGN_INPUTS_TO_BLOCKS=false

# ablation w = 48
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_setdlm_tgt16_knull_maxb48_sweep_v1"
# KV_CACHING=true
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=16
# ALIGN_INPUTS_TO_BLOCKS=false

# ablation w = 64
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_setdlm_tgt16_knull_maxb64_sweep_v1"
# KV_CACHING=true
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=64
# ALIGN_INPUTS_TO_BLOCKS=false

# ablation grad accum
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz4_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_setdlm_tgt16_knull_maxb32_sweep_v1"
# KV_CACHING=true
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

# papl
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_mdlm_distill_papl_v1"
# KV_CACHING=false
# BLOCK_SIZE=32
# MAX_WINDOW_SIZE=${BLOCK_SIZE}
# ALIGN_INPUTS_TO_BLOCKS=true

# esolm
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.0_max-dur75000ba_amp_bf16_layers28_eso_a1.0_bsplit1.0_dshuftrue_dattncausal_sshuffalse_sattncausal_low_var"
# KV_CACHING=true
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=${BLOCK_SIZE}
# ALIGN_INPUTS_TO_BLOCKS=false

# refusion
MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_refusion_len1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_refusion_distill_v1"
BLOCK_SIZE=1024
MAX_WINDOW_SIZE=16
KV_CACHING=true
ALIGN_INPUTS_TO_BLOCKS=false
REQUIRE_REFUSION_SEMANTICS=true

# grad accum 4
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz4_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_setdlm_tgt16_knull_maxb32_sweep_v1"
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=16
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false


# grad accum 16
# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz16_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_setdlm_tgt16_knull_maxb32_sweep_v1"
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=16
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false

# MODEL_PATH="/share/kuleshov/ma2238/runs/dllm-dev/gsm8k-0shot_block1024_lr1e-5_bsz16_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_setdlm_tgt4_distill_accum_v1"
# BLOCK_SIZE=1024
# MAX_WINDOW_SIZE=4
# KV_CACHING=true
# ALIGN_INPUTS_TO_BLOCKS=false

echo "MODEL_PATH: ${MODEL_PATH}"

USE_EMA=true
OUTPUT_DIR="outputs/${MODEL_PATH}/lm_eval_harness_output"
REVISION=null
TOKENIZER_PATH="Qwen/Qwen3-1.7B-Base"

REFUSION_LENGTH=null
REFUSION_SLOT_SIZE=8
REFUSION_SERIAL_NUM_BLOCKS=2
REFUSION_SLOT_THRESHOLD=0.9
REFUSION_TOKEN_THRESHOLD=0.9
REFUSION_TEMPERATURE=0.0

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
echo "TOKENIZER_PATH: ${TOKENIZER_PATH}"
echo "REQUIRE_REFUSION_SEMANTICS: ${REQUIRE_REFUSION_SEMANTICS}"

OUTPUT_PATH="${OUTPUT_DIR}/ema${USE_EMA}_ckpt${CKPT}_L${L}_block${BLOCK_SIZE}-do_sample${DO_SAMPLE}-sampling_strategy${SAMPLING_STRATEGY}-T${T}_first_hit${FIRST_HITTING}-conf_noise${CONFIDENCE_BASED_NOISING}-conf_margin_noise${CONFIDENCE_MARGIN_BASED_NOISING}-conf_thold${CONFIDENCE_THRESHOLD}-align_to_blocks${ALIGN_INPUTS_TO_BLOCKS}-max_window_size${MAX_WINDOW_SIZE}"
if [ "${REQUIRE_REFUSION_SEMANTICS}" = true ]; then
  OUTPUT_PATH="${OUTPUT_DIR}/refusion_ema${USE_EMA}_ckpt${CKPT}_L${L}_slot${REFUSION_SLOT_SIZE}_serial${REFUSION_SERIAL_NUM_BLOCKS}_slotth${REFUSION_SLOT_THRESHOLD}_tokenth${REFUSION_TOKEN_THRESHOLD}_temp${REFUSION_TEMPERATURE}"
fi
OUTPUT_PATH="${OUTPUT_PATH}_test"
mkdir -p ${OUTPUT_PATH}

MODEL_ARGS=()
GENERATION_ARGS=()
if [ "${REQUIRE_REFUSION_SEMANTICS}" = true ]; then
  MODEL_ARGS+=(
    +task.model.model_config_overrides.model_type=refusion
  )
  if [ "${REFUSION_LENGTH}" != "null" ]; then
    MODEL_ARGS+=(
      +task.model.model_config_overrides.length=${REFUSION_LENGTH}
    )
  fi
  GENERATION_ARGS+=(
    generation@generation_config=refusion_generation_config
    generation_config.do_sample=${DO_SAMPLE}
    generation_config.temperature=${REFUSION_TEMPERATURE}
    generation_config.use_cache=${KV_CACHING}
    generation_config.slot_size=${REFUSION_SLOT_SIZE}
    generation_config.serial_num_blocks=${REFUSION_SERIAL_NUM_BLOCKS}
    generation_config.slot_threshold=${REFUSION_SLOT_THRESHOLD}
    generation_config.token_threshold=${REFUSION_TOKEN_THRESHOLD}
  )
else
  GENERATION_ARGS+=(
    generation@generation_config=set_diffusion_generation_config
    generation_config.do_sample=${DO_SAMPLE}
    generation_config.sampling_strategy=${SAMPLING_STRATEGY}
    generation_config.num_steps=${T}
    generation_config.first_hitting=${FIRST_HITTING}
    generation_config.confidence_based_noising=${CONFIDENCE_BASED_NOISING}
    generation_config.confidence_margin_based_noising=${CONFIDENCE_MARGIN_BASED_NOISING}
    generation_config.confidence_threshold=${CONFIDENCE_THRESHOLD}
    generation_config.use_cache=${KV_CACHING}
    generation_config.align_inputs_to_blocks=${ALIGN_INPUTS_TO_BLOCKS}
    generation_config.max_window_size=${MAX_WINDOW_SIZE}
    generation_config.linear_unmasking=${LINEAR_UNMASKING}
  )
fi

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
  tokenizer.pretrained_model_name_or_path=${TOKENIZER_PATH} \
  output_path=${OUTPUT_PATH} \
  generated_samples_output_path=${OUTPUT_PATH} \
  max_length=${L} \
  max_new_tokens=${L} \
  block_size=${BLOCK_SIZE} \
  "${MODEL_ARGS[@]}" \
  "${GENERATION_ARGS[@]}" \
  ~generation/logits_processor@logits_processor_list \
  gen_kwargs.logits_processor=null \
  generation/stopping_criteria@stopping_criteria_list='[gsm8k_regex_stopping_criteria,repeating_token]'
