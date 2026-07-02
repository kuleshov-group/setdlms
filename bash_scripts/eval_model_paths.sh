#!/bin/bash

resolve_eval_model_path() {
  local requested="${1:-}"
  if [ -z "${requested}" ]; then
    echo "ERROR: Set MODEL_PATH or EVAL_MODEL_KEY to a HF repo id, local path, or known model key." >&2
    return 1
  fi

  local repo_prefix="${EVAL_MODEL_HF_PREFIX:-kuleshov-group}"
  RESOLVED_CKPT_FILE=""
  RESOLVED_USE_EMA=""
  RESOLVED_MODEL_REPO_ID=""
  RESOLVED_MODEL_SOURCE="input"

  case "${requested}" in
    */*|/*|.*)
      RESOLVED_MODEL_PATH="${requested}"
      ;;

    gsm8k:ar) RESOLVED_MODEL_PATH="${repo_prefix}/gsm8k-ar" ;;
    gsm8k:mdlm) RESOLVED_MODEL_PATH="${repo_prefix}/gsm8k-mdlm" ;;
    gsm8k:bd3lm-s4) RESOLVED_MODEL_PATH="${repo_prefix}/gsm8k-bd3lm-s4" ;;
    gsm8k:bd3lm-s8) RESOLVED_MODEL_PATH="${repo_prefix}/gsm8k-bd3lm-s8" ;;
    gsm8k:bd3lm-s16) RESOLVED_MODEL_PATH="${repo_prefix}/gsm8k-bd3lm-s16" ;;
    gsm8k:setdlm-d4) RESOLVED_MODEL_PATH="${repo_prefix}/setdlm-gsm8k-smax8" ;;
    gsm8k:setdlm-d8) RESOLVED_MODEL_PATH="${repo_prefix}/setdlm-gsm8k-smax16" ;;
    gsm8k:setdlm-d16) RESOLVED_MODEL_PATH="${repo_prefix}/setdlm-gsm8k-smax32" ;;

    owt:ar) RESOLVED_MODEL_PATH="${repo_prefix}/owt-ar" ;;
    owt:mdlm) RESOLVED_MODEL_PATH="${repo_prefix}/owt-mdlm" ;;
    owt:bd3lm-s4) RESOLVED_MODEL_PATH="${repo_prefix}/owt-bd3lm-s4" ;;
    owt:bd3lm-s8) RESOLVED_MODEL_PATH="${repo_prefix}/owt-bd3lm-s8" ;;
    owt:bd3lm-s16) RESOLVED_MODEL_PATH="${repo_prefix}/owt-bd3lm-s16" ;;
    owt:setdlm-d4) RESOLVED_MODEL_PATH="${repo_prefix}/owt-setdlm-smax8" ;;
    owt:setdlm-d8) RESOLVED_MODEL_PATH="${repo_prefix}/owt-setdlm-smax16" ;;
    owt:setdlm-d16) RESOLVED_MODEL_PATH="${repo_prefix}/owt-setdlm-smax32" ;;

    lm1b:ar) RESOLVED_MODEL_PATH="${repo_prefix}/lm1b-ar" ;;
    lm1b:mdlm) RESOLVED_MODEL_PATH="${repo_prefix}/lm1b-mdlm" ;;
    lm1b:bd3lm-s4) RESOLVED_MODEL_PATH="${repo_prefix}/lm1b-bd3lm-s4" ;;
    lm1b:bd3lm-s8) RESOLVED_MODEL_PATH="${repo_prefix}/lm1b-bd3lm-s8" ;;
    lm1b:bd3lm-s16) RESOLVED_MODEL_PATH="${repo_prefix}/lm1b-bd3lm-s16" ;;
    lm1b:setdlm-d4) RESOLVED_MODEL_PATH="${repo_prefix}/lm1b-setdlm-smax8" ;;
    lm1b:setdlm-d8) RESOLVED_MODEL_PATH="${repo_prefix}/lm1b-setdlm-smax16" ;;
    lm1b:setdlm-d16) RESOLVED_MODEL_PATH="${repo_prefix}/lm1b-setdlm-smax32" ;;

    cnndm:ar) RESOLVED_MODEL_PATH="${repo_prefix}/cnndm-ar" ;;
    cnndm:mdlm) RESOLVED_MODEL_PATH="${repo_prefix}/cnndm-mdlm" ;;
    cnndm:bd3lm-s4) RESOLVED_MODEL_PATH="${repo_prefix}/cnndm-bd3lm-s4" ;;
    cnndm:bd3lm-s8) RESOLVED_MODEL_PATH="${repo_prefix}/cnndm-bd3lm-s8" ;;
    cnndm:bd3lm-s16) RESOLVED_MODEL_PATH="${repo_prefix}/cnndm-bd3lm-s16" ;;
    cnndm:setdlm-d4) RESOLVED_MODEL_PATH="${repo_prefix}/cnndm-setdlm-smax8" ;;
    cnndm:setdlm-d8) RESOLVED_MODEL_PATH="${repo_prefix}/cnndm-setdlm-smax16" ;;
    cnndm:setdlm-d16) RESOLVED_MODEL_PATH="${repo_prefix}/cnndm-setdlm-smax32" ;;

    *)
      echo "ERROR: Unknown EVAL_MODEL_KEY . Pass a full HF id or local path in MODEL_PATH, or use a known key." >&2
      return 1
      ;;
  esac

  if [ "${RESOLVED_MODEL_PATH}" != "${requested}" ]; then
    RESOLVED_MODEL_SOURCE="known-key"
    RESOLVED_MODEL_REPO_ID="${RESOLVED_MODEL_PATH}"
  elif [[ "${RESOLVED_MODEL_PATH}" == */* && "${RESOLVED_MODEL_PATH}" != /* && "${RESOLVED_MODEL_PATH}" != ./* ]]; then
    RESOLVED_MODEL_SOURCE="hf"
    RESOLVED_MODEL_REPO_ID="${RESOLVED_MODEL_PATH}"
  else
    RESOLVED_MODEL_SOURCE="local"
  fi

  MODEL_PATH="${RESOLVED_MODEL_PATH}"
  echo "Resolved MODEL_PATH (${RESOLVED_MODEL_SOURCE}): ${MODEL_PATH}"
  return 0
}
