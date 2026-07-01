#!/bin/bash

resolve_eval_model_path() {
  local requested="${1:-}"
  if [ -z "${requested}" ]; then
    echo "ERROR: Set MODEL_PATH or EVAL_MODEL_KEY to a HF repo id, local path, or known model key." >&2
    return 1
  fi

  local repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local resolver="${repo_root}/scripts/push_hf_models.py"
  local repo_prefix="${EVAL_MODEL_HF_PREFIX:-kuleshov-group}"
  local output=""
  local candidate=""
  local candidates=()

  if [ -n "${EVAL_MODEL_RESOLVER_PYTHON:-}" ]; then
    candidates+=("${EVAL_MODEL_RESOLVER_PYTHON}")
  fi
  if [ -n "${PYTHON:-}" ]; then
    candidates+=("${PYTHON}")
  fi
  candidates+=(python python3)

  local args=("${resolver}" --resolve "${requested}" --repo-prefix "${repo_prefix}")
  if [ "${EVAL_MODEL_PREFER_LOCAL:-false}" = true ]; then
    args+=(--prefer-local)
  fi
  if [ "${EVAL_MODEL_SKIP_HF_CHECK:-false}" = true ]; then
    args+=(--no-hf-check)
  fi

  for candidate in "${candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if output="$("${candidate}" "${args[@]}")"; then
      eval "${output}"
      MODEL_PATH="${RESOLVED_MODEL_PATH}"
      if [ -n "${RESOLVED_CKPT_FILE:-}" ] && [ -z "${CKPT_FILE:-}" ]; then
        CKPT_FILE="${RESOLVED_CKPT_FILE}"
      fi
      if [ -n "${RESOLVED_USE_EMA:-}" ] && [ -z "${USE_EMA:-}" ]; then
        USE_EMA="${RESOLVED_USE_EMA}"
      fi
      echo "Resolved MODEL_PATH (${RESOLVED_MODEL_SOURCE}): ${MODEL_PATH}"
      if [ "${RESOLVED_MODEL_SOURCE}" = local ] && [ -n "${RESOLVED_MODEL_REPO_ID:-}" ]; then
        echo "  HF unavailable, using local fallback for ${RESOLVED_MODEL_REPO_ID}."
      fi
      return 0
    fi
  done

  local cache_root="${HF_HOME:-${repo_root}/.hf_cache}"
  local cache_repo="${requested//\//--}"
  local cache_snapshots="${cache_root}/models--${cache_repo}/snapshots"
  if [ -d "${cache_snapshots}" ]; then
    local cached_snapshot=""
    cached_snapshot="$(ls -dt "${cache_snapshots}"/* 2>/dev/null | head -n 1 || true)"
    if [ -n "${cached_snapshot}" ] && [ -d "${cached_snapshot}" ]; then
      RESOLVED_MODEL_PATH="${cached_snapshot}"
      RESOLVED_MODEL_SOURCE="hf-cache"
      RESOLVED_MODEL_REPO_ID="${requested}"
      MODEL_PATH="${RESOLVED_MODEL_PATH}"
      echo "Resolved MODEL_PATH (${RESOLVED_MODEL_SOURCE}): ${MODEL_PATH}"
      echo "  HF resolver unavailable, using cached snapshot for ${requested}."
      return 0
    fi
  fi

  echo "ERROR: Could not resolve model path '${requested}'." >&2
  echo "       Tried resolver: ${resolver}" >&2
  return 1
}
