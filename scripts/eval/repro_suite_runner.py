#!/usr/bin/env python3
"""Build and optionally launch the reproducibility eval command matrix.

This orchestrator intentionally calls the existing shell eval entrypoints. The
default mode is a dry run that verifies checkpoint/config paths and prints every
command without submitting jobs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_RUN_DIR = Path("/share/kuleshov/ma2238/runs/dllm-dev")
TEXTDIFFUSION_CKPT_DIR = Path("/share/kuleshov/ma2238/textdiffusion/checkpoints")
LM1B_DATA_DIR = Path("/share/kuleshov/ma2238/dllm-data")
LM1B_EVAL_DATASET = LM1B_DATA_DIR / "lm1b_test_bs128_wrapped.dat"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "repro_eval_suite"
EVAL_MODEL_HF_PREFIX = os.environ.get("EVAL_MODEL_HF_PREFIX", "kuleshov-group")

KNOWN_EVAL_MODEL_KEYS = {
    "gsm8k:ar": "gsm8k-ar",
    "gsm8k:mdlm": "gsm8k-mdlm",
    "gsm8k:bd3lm-s4": "gsm8k-bd3lm-s4",
    "gsm8k:bd3lm-s8": "gsm8k-bd3lm-s8",
    "gsm8k:bd3lm-s16": "gsm8k-bd3lm-s16",
    "gsm8k:setdlm-d4": "setdlm-gsm8k-smax8",
    "gsm8k:setdlm-d8": "setdlm-gsm8k-smax16",
    "gsm8k:setdlm-d16": "setdlm-gsm8k-smax32",
    "gsm8k:setdlm-smax8": "setdlm-gsm8k-smax8",
    "gsm8k:setdlm-smax16": "setdlm-gsm8k-smax16",
    "gsm8k:setdlm-smax32": "setdlm-gsm8k-smax32",
    "owt:ar": "owt-ar",
    "owt:mdlm": "owt-mdlm",
    "owt:bd3lm-s4": "owt-bd3lm-s4",
    "owt:bd3lm-s8": "owt-bd3lm-s8",
    "owt:bd3lm-s16": "owt-bd3lm-s16",
    "owt:setdlm-d4": "owt-setdlm-smax8",
    "owt:setdlm-d8": "owt-setdlm-smax16",
    "owt:setdlm-d16": "owt-setdlm-smax32",
    "owt:setdlm-smax8": "owt-setdlm-smax8",
    "owt:setdlm-smax16": "owt-setdlm-smax16",
    "owt:setdlm-smax32": "owt-setdlm-smax32",
    "lm1b:setdlm-d4": "lm1b-setdlm-smax8",
    "lm1b:setdlm-d8": "lm1b-setdlm-smax16",
    "lm1b:setdlm-d16": "lm1b-setdlm-smax32",
    "lm1b:setdlm-smax8": "lm1b-setdlm-smax8",
    "lm1b:setdlm-smax16": "lm1b-setdlm-smax16",
    "lm1b:setdlm-smax32": "lm1b-setdlm-smax32",
    "cnndm:setdlm-d4": "cnndm-setdlm-smax8",
    "cnndm:setdlm-d8": "cnndm-setdlm-smax16",
    "cnndm:setdlm-d16": "cnndm-setdlm-smax32",
    "cnndm:setdlm-smax8": "cnndm-setdlm-smax8",
    "cnndm:setdlm-smax16": "cnndm-setdlm-smax16",
    "cnndm:setdlm-smax32": "cnndm-setdlm-smax32",
}
THROUGHPUT_NUM_GPUS = os.environ.get("THROUGHPUT_NUM_GPUS", "4")
THROUGHPUT_MEASURED_EXAMPLES = os.environ.get("THROUGHPUT_MEASURED_EXAMPLES", "200")
THROUGHPUT_WARMUP_EXAMPLES_PER_RANK = os.environ.get("THROUGHPUT_WARMUP_EXAMPLES_PER_RANK", "50")
THROUGHPUT_SAMPLES_PER_RANK = os.environ.get("THROUGHPUT_SAMPLES_PER_RANK", "250")
OWT_TABLE4_REFERENCE_SEED = os.environ.get("OWT_TABLE4_REFERENCE_SEED", "20260701")
OWT_TABLE4_REFERENCE_SIZE = os.environ.get("OWT_TABLE4_REFERENCE_SIZE", "1000")
OWT_TABLE4_REFERENCE_SUBSETS = os.environ.get("OWT_TABLE4_REFERENCE_SUBSETS", "5")
OWT_TABLE4_MAUVE_SEED = os.environ.get("OWT_TABLE4_MAUVE_SEED", "1234")
OWT_TABLE4_DATASET_PATH = os.environ.get(
    "OWT_TABLE4_DATASET_PATH",
    str(REPO_ROOT / "data/gpt2_tokenizer/openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat"),
)
GSM8K_PARETO_THRESHOLDS = (
    "1e6",
    "0.99",
    "0.95",
    "0.90",
    "0.85",
    "0.80",
    "0.75",
    "0.70",
    "0.65",
    "0.60",
)
INFILL_THROUGHPUT_TASKS = {"infilling1_throughput", "infilling_throughput"}
THROUGHPUT_TASKS = {
    "throughput_owt",
    "throughput_owt_table4",
    "throughput_lm1b",
    "gsm8k_pareto_throughput",
    "cnndm_summarization_throughput",
    *INFILL_THROUGHPUT_TASKS,
}
A6000_CONSTRAINT = "[a6000]"
A6000_OR_A5000_CONSTRAINT = "[a6000|a5000]"
GPU_PARTITION_LIST = "gpu,kuleshov"
THROUGHPUT_GPU_PARTITION = "kuleshov"
THROUGHPUT_NODELIST = "kuleshov-compute-02"
MDLM_SAMPLING_TASKS = {
    "infilling",
    "infilling1",
    "infilling_legacy",
    "infilling1_legacy",
    "infilling1_throughput",
    "infilling_throughput",
    "cnndm_summarization",
    "cnndm_summarization_throughput",
    "gsm8k_accuracy",
    "gsm8k_pareto_accuracy",
    "gsm8k_pareto_throughput",
    "mauve",
    "mauve_owt_table4",
    "throughput_owt",
    "throughput_owt_table4",
    "throughput_lm1b",
}
GSM8K_TASKS = {
    "gsm8k_accuracy",
    "gsm8k_ppl",
    "gsm8k_pareto_accuracy",
    "gsm8k_pareto_throughput",
}
OWT_RELATED_A6000_TASKS = {
    "owt_ppl",
    "commonsense",
    "mauve",
    "mauve_owt_table4",
    "throughput_owt",
    "throughput_owt_table4",
    "infilling",
    "infilling1",
    "infilling_legacy",
    "infilling1_legacy",
    "infilling_throughput",
    "infilling1_throughput",
    "setdlm_infill3_first_hitting_diagnostic",
}


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


@functools.lru_cache(maxsize=None)
def resolve_model_path_for_eval(model_path: str) -> str:
    if not model_path or model_path == "N/A":
        return model_path
    if model_path in KNOWN_EVAL_MODEL_KEYS:
        return f"{EVAL_MODEL_HF_PREFIX}/{KNOWN_EVAL_MODEL_KEYS[model_path]}"
    return model_path


def is_remote_or_known_eval_model(model_path: str) -> bool:
    if model_path in KNOWN_EVAL_MODEL_KEYS:
        return True
    return (
        "/" in model_path
        and not model_path.startswith("/")
        and not model_path.startswith("./")
        and not model_path.startswith("../")
    )


def rank_invariant_generation_env() -> dict[str, str]:
    return {
        "LM_EVAL_RANK_INVARIANT_SEED": os.environ.get(
            "LM_EVAL_RANK_INVARIANT_SEED", "true"
        ),
        "LM_EVAL_BASE_SEED": os.environ.get("LM_EVAL_BASE_SEED", "1234"),
    }


@dataclass(frozen=True)
class Target:
    variant: str
    model_path: str
    ckpt_file: str
    use_ema: str
    eval_block_size: str
    max_window_size: str
    block_size: str = "N/A"
    desired_block_size: str = "N/A"
    kv_caching: str = "true"
    align_inputs_to_blocks: str = "true"
    num_visible_devices: str = "1"
    status: str = "ready"
    reason: str = ""
    confidence_threshold: str = "N/A"
    confidence_based_noising: str = "N/A"
    pareto_group: str = "N/A"

    @property
    def checkpoint_path(self) -> str:
        if not self.model_path or self.model_path == "N/A":
            return "N/A"
        if is_remote_or_known_eval_model(self.model_path):
            return resolve_model_path_for_eval(self.model_path)
        path = Path(self.model_path)
        if path.suffix == ".ckpt" or path.suffix == ".pt":
            return str(path)
        in_checkpoints = path / "checkpoints" / self.ckpt_file
        if in_checkpoints.exists():
            return str(in_checkpoints)
        return str(path / self.ckpt_file)

    @property
    def label(self) -> str:
        if self.variant == "BD3LM":
            label = f"BD3LM-s{self.block_size}"
        elif self.variant == "SetDLM":
            if self.desired_block_size != "N/A":
                label = f"SetDLM-smax{2 * int(self.desired_block_size)}"
            else:
                label = "SetDLM-smaxN/A"
        else:
            label = self.variant
        if self.confidence_threshold != "N/A":
            return f"{label}-conf{self.confidence_threshold}"
        return label


@dataclass
class MatrixRow:
    row_id: str
    task: str
    model_variant: str
    block_size: str
    desired_block_size: str
    eval_block_size: str
    max_window_size: str
    confidence_threshold: str
    confidence_based_noising: str
    pareto_group: str
    checkpoint_path: str
    model_path: str
    ckpt_file: str
    use_ema: str
    command: str
    local_command: str
    script: str
    output_path: str
    log_path: str
    group_log_path: str
    metadata_path: str
    job_id: str
    output_root: str
    git_sha: str
    node_name: str
    gpu_name: str
    num_gpus: str
    gpu_constraint: str
    gpu_partition: str
    throughput_measured_examples: str
    throughput_warmup_examples_per_rank: str
    throughput_global_measurements: str
    throughput_num_gpus: str
    compile_backbone: str
    compile_mode: str
    compile_dynamic: str
    compile_supported: str
    cnndm_generate_target_prompt: str
    setdlm_throughput_run_name: str
    status: str
    reason: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    script: str
    group: str
    env_builder: Callable[[Target], dict[str, str]]
    output_builder: Callable[[Target], str]


def q(value: str) -> str:
    return shlex.quote(str(value))


def env_prefix(env: dict[str, str]) -> str:
    return " ".join(f"{key}={q(value)}" for key, value in sorted(env.items()))


def script_job_name(script: str) -> str:
    stem = Path(script).stem
    return stem[4:] if stem.startswith("run_") else stem


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def resource_policy(task: "TaskSpec", target: Target) -> tuple[str, str]:
    if task.name in THROUGHPUT_TASKS:
        return THROUGHPUT_NUM_GPUS, A6000_CONSTRAINT
    if task.name == "mauve":
        return "8", A6000_CONSTRAINT
    if target.variant == "MDLM" and task.name in MDLM_SAMPLING_TASKS:
        if (
            task.name == "cnndm_summarization"
            or task.name in GSM8K_TASKS
            or task.name in OWT_RELATED_A6000_TASKS
        ):
            return "8", A6000_CONSTRAINT
        return "8", A6000_OR_A5000_CONSTRAINT
    if task.name in OWT_RELATED_A6000_TASKS:
        return "4", A6000_CONSTRAINT
    if task.name == "lm1b_ppl":
        return "4", A6000_OR_A5000_CONSTRAINT
    if task.name == "cnndm_summarization":
        return "4", A6000_CONSTRAINT
    if task.name in GSM8K_TASKS:
        return "4", A6000_CONSTRAINT
    return target.num_visible_devices, A6000_CONSTRAINT


def apply_resource_policy(
    env: dict[str, str],
    task: "TaskSpec",
    target: Target,
) -> tuple[str, str]:
    num_gpus, gpu_constraint = resource_policy(task, target)
    env["NUM_VISIBLE_DEVICES"] = num_gpus
    env["GPU_CONSTRAINT"] = gpu_constraint
    if task.name in THROUGHPUT_TASKS:
        env["GPU_PARTITION"] = THROUGHPUT_GPU_PARTITION
        env["GPU_NODELIST"] = THROUGHPUT_NODELIST
        env["THROUGHPUT_NODELIST"] = THROUGHPUT_NODELIST
    else:
        env["GPU_PARTITION"] = GPU_PARTITION_LIST
    if num_gpus.isdigit() and int(num_gpus) >= 8:
        env["JOB_MEM"] = "256000"
    if task.name == "owt_ppl":
        env["EVAL_NUM_PROCESSES"] = num_gpus
    return num_gpus, gpu_constraint


def throughput_compile_env(target: Target) -> dict[str, str]:
    """Disable inference compilation for final throughput reporting."""
    return {
        "COMPILE_BACKBONE": "false",
        "COMPILE_MODE": "none",
        "COMPILE_DYNAMIC": "false",
        "COMPILE_SUPPORTED": "false",
    }


def is_confidence_threshold_run(target: Target) -> bool:
    return target.confidence_threshold not in {"N/A", "1e6"}


def setdlm_throughput_env(target: Target) -> dict[str, str]:
    if target.variant in {"MDLM", "BD3LM"}:
        fused_cache = target.variant == "BD3LM"
        return {
            "FUSED_BLOCK_CACHE": "auto",
            "SETDLM_THROUGHPUT_RUN_NAME": (
                "bd3lm_fused_block_cache"
                if fused_cache
                else "mdlm_no_fused_block_cache"
            ),
        }
    if target.variant != "SetDLM":
        return {}
    confidence_override = os.environ.get(
        "CNNDM_SETDLM_CONFIDENCE_THRESHOLD",
        os.environ.get("CONFIDENCE_THRESHOLD", ""),
    ).strip()
    if confidence_override:
        confidence_threshold = confidence_override
    else:
        confidence_threshold = (
            target.confidence_threshold
            if target.confidence_threshold != "N/A"
            else "1e6"
        )
    confidence_run = confidence_threshold != "1e6"
    return {
        "CONFIDENCE_BASED_NOISING": "true" if confidence_run else "false",
        "CONFIDENCE_THRESHOLD": confidence_threshold,
        "CONF_THRESHOLD": confidence_threshold,
    }


def setdlm_output_id(target: Target, env: dict[str, str]) -> str:
    size = target.desired_block_size if target.desired_block_size != "N/A" else target.block_size
    if size != "N/A":
        size = f"smax{2 * int(size)}"
    run_name = env.get("SETDLM_THROUGHPUT_RUN_NAME", "setdlm_dynamic_all")
    confidence = env.get("CONFIDENCE_THRESHOLD", target.confidence_threshold)
    if confidence and confidence != "N/A":
        return f"{run_name}_{size}_conf{confidence}"
    return f"{run_name}_{size}"


def output_under_model(model_path: str, suffix: str) -> str:
    # Match the shell wrappers after HF/local model path resolution.
    return f"outputs/{resolve_model_path_for_eval(model_path)}/{suffix}"


def likelihood_stdout_only(_: Target) -> str:
    return "N/A (metrics are printed to stdout/stderr log)"


def build_owt_ppl_env(target: Target) -> dict[str, str]:
    return {
        "MODEL_NAME": target.label,
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "CKPT_FILE": target.ckpt_file,
        "USE_EMA": target.use_ema,
        "BLOCK_SIZE": target.eval_block_size,
        "MAX_BLOCK_SIZE": target.max_window_size,
        "BATCH_SIZE": "16",
        "COMPILE_BACKBONE": "true",
        "EVAL_NUM_PROCESSES": target.num_visible_devices,
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }


def build_lm1b_ppl_env(target: Target) -> dict[str, str]:
    return {
        "DATA_DIR": str(LM1B_DATA_DIR),
        "LM1B_MODEL_NAME": target.label,
        "LM1B_MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "LM1B_CKPT_FILE": target.ckpt_file,
        "LM1B_USE_EMA": target.use_ema,
        "BLOCK_SIZE": target.eval_block_size,
        "BATCH_SIZE": "16",
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }


def build_lm1b_uncond_env(target: Target) -> dict[str, str]:
    env = {
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "CKPT_FILE": target.ckpt_file,
        "USE_EMA": target.use_ema,
        "BLOCK_SIZE": target.eval_block_size,
        "MAX_WINDOW_SIZE": target.max_window_size,
        "KV_CACHING": target.kv_caching,
        "ALIGN_INPUTS_TO_BLOCKS": target.align_inputs_to_blocks,
        "TOKENIZER_PATH": "bert-base-uncased",
        "OUTPUT_DATASET_NAME": "lm1b",
        "MAX_LENGTH": "128",
        "NUM_SAMPLES": "1000",
        "OUTPUT_NUM_SAMPLES": "1000",
        "SKIP_MAUVE": "true",
        "REPETITION_PENALTY": "1.05",
        "NUCLEUS_P": "0.95",
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }
    if target.variant == "SEDD":
        env.update(
            {
                "MODEL_FAMILY": "SEDD",
                "GENERATION_CONFIG_NAME": "sedd_generation_config",
                "SAMPLING_STRATEGY": "analytic",
                "NOISE_REMOVAL": "true",
            }
        )
    return env


def build_gsm8k_ppl_env(target: Target) -> dict[str, str]:
    env = {
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "CKPT_FILE": target.ckpt_file,
        "USE_EMA": target.use_ema,
        "BLOCK_SIZE": target.eval_block_size,
        "BATCH_SIZE": "1",
        "EVAL_DATASET": "gsm8k_eval_distill",
        "PRETRAINED_MODEL_NAME_OR_PATH": "Qwen/Qwen3-1.7B-Base",
        "SEED": "1",
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }


def infill_align_inputs_to_blocks(target: Target) -> str:
    # BD3LM/MDLM infilling checkpoints were evaluated historically with input alignment.
    # Keeping this scoped to infilling avoids changing MAUVE/OWT generation rows.
    if target.variant in {"BD3LM", "MDLM"}:
        return "true"
    return target.align_inputs_to_blocks


def infill1_repeat_penalty() -> str:
    return os.environ.get("INFILL1_REPEAT_PENALTY", "1.1")


def infill3_repeat_penalty() -> str:
    return os.environ.get("INFILL3_REPEAT_PENALTY", "1.5")


def infill_right_context_penalty_enabled() -> bool:
    return os.environ.get(
        "INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT", "false"
    ).lower() in {"1", "true", "yes", "on"}


def infill_full_visible_context_cache_enabled() -> bool:
    return os.environ.get(
        "INFILL_CACHE_FULL_VISIBLE_CONTEXT", "false"
    ).lower() in {"1", "true", "yes", "on"}


def infill_output_tag(base_tag: str = "") -> str:
    tag = base_tag
    if infill_right_context_penalty_enabled():
        suffix = os.environ.get("INFILL_RIGHT_CONTEXT_OUTPUT_SUFFIX", "rightctxpenalty")
        tag = f"{tag}_{suffix}" if tag else suffix
    output_suffix = os.environ.get("INFILL_OUTPUT_TAG_SUFFIX", "").strip()
    if output_suffix:
        tag = f"{tag}_{output_suffix}" if tag else output_suffix
    return tag


def build_infill_env(target: Target) -> dict[str, str]:
    env = {
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "CKPT_FILE": target.ckpt_file,
        "USE_EMA": target.use_ema,
        "BLOCK_SIZE": target.eval_block_size,
        "MAX_WINDOW_SIZE": target.max_window_size,
        "KV_CACHING": target.kv_caching,
        "ALIGN_INPUTS_TO_BLOCKS": infill_align_inputs_to_blocks(target),
        "NUM_TARGET_SENTENCES": "3",
        "REPEAT_PENALTY": infill3_repeat_penalty(),
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }
    compile_backbone = os.environ.get("INFILL_COMPILE_BACKBONE", "").strip()
    if compile_backbone:
        env["COMPILE_BACKBONE"] = compile_backbone
    confidence_threshold = os.environ.get(
        "INFILL_CONFIDENCE_THRESHOLD", os.environ.get("CONFIDENCE_THRESHOLD", "")
    ).strip()
    confidence_based_noising = os.environ.get(
        "INFILL_CONFIDENCE_BASED_NOISING",
        os.environ.get("CONFIDENCE_BASED_NOISING", ""),
    ).strip()
    if confidence_threshold:
        env["CONFIDENCE_THRESHOLD"] = confidence_threshold
        env["CONF_THRESHOLD"] = confidence_threshold
        if not confidence_based_noising:
            confidence_based_noising = (
                "false" if confidence_threshold in {"N/A", "1e6"} else "true"
            )
    if confidence_based_noising:
        env["CONFIDENCE_BASED_NOISING"] = confidence_based_noising
    cache_order = os.environ.get("SETDLM_INFILL_CACHE_PROMOTION_ORDER", "").strip()
    if cache_order and target.variant == "SetDLM":
        env["SETDLM_INFILL_CACHE_PROMOTION_ORDER"] = cache_order
    if infill_right_context_penalty_enabled():
        env["INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT"] = "true"
        env["OUTPUT_TAG"] = infill_output_tag()
    return env


def build_infill1_env(target: Target) -> dict[str, str]:
    env = build_infill_env(target)
    env["NUM_TARGET_SENTENCES"] = "1"
    env["REPEAT_PENALTY"] = infill1_repeat_penalty()
    return env


def legacy_infill_align_inputs_to_blocks(target: Target) -> str:
    if target.variant in {"BD3LM", "MDLM"}:
        return "true"
    return "false"


def build_infill_legacy_env(target: Target) -> dict[str, str]:
    env = build_infill_env(target)
    env.update(
        {
            "ALIGN_INPUTS_TO_BLOCKS": legacy_infill_align_inputs_to_blocks(target),
            "CACHE_FULL_INFILL_CONTEXT": (
                "true" if infill_full_visible_context_cache_enabled() else "false"
            ),
            "OUTPUT_TAG": infill_output_tag("legacy_qualitative"),
            "NUM_TARGET_SENTENCES": "3",
            "REPEAT_PENALTY": infill3_repeat_penalty(),
        }
    )
    if target.variant == "SetDLM" and target.desired_block_size == "16":
        # Earlier successful SetDLM-smax32 infilling used a 32-token generation window.
        # Keep compilation enabled for throughput comparability.
        env["MAX_WINDOW_SIZE"] = "32"
        env["OUTPUT_TAG"] = infill_output_tag("legacy_mw32")
    if target.variant == "SetDLM":
        setdlm_max_window_size = os.environ.get(
            "INFILL_SETDLM_MAX_WINDOW_SIZE", ""
        ).strip()
        if setdlm_max_window_size:
            env["MAX_WINDOW_SIZE"] = setdlm_max_window_size
            env["OUTPUT_TAG"] = infill_output_tag(
                f"legacy_mw{setdlm_max_window_size}"
            )
    return env


def build_infill1_legacy_env(target: Target) -> dict[str, str]:
    env = build_infill_legacy_env(target)
    env["NUM_TARGET_SENTENCES"] = "1"
    env["REPEAT_PENALTY"] = infill1_repeat_penalty()
    return env


def _diag_first_hitting(target: Target) -> str:
    return "true" if "-fhtrue-" in target.variant else "false"


def _diag_cache_promotion_order(target: Target) -> str:
    if "-cache-first_hitting-" in target.variant:
        return "first_hitting"
    if "-cache-l2r-" in target.variant:
        return "l2r"
    return "l2r"


def _diag_family(target: Target) -> str:
    if "SetDLM-old-" in target.variant:
        return "old"
    return "current"


def _diag_use_eos(target: Target) -> str:
    return "true" if target.variant.endswith("-eostrue") else "false"


def _diag_output_tag(target: Target) -> str:
    family = "o" if _diag_family(target) == "old" else "c"
    order = "fh" if _diag_cache_promotion_order(target) == "first_hitting" else "l2r"
    eos = "1" if _diag_use_eos(target) == "true" else "0"
    return (
        f"diagco_{family}_{order}_e{eos}"
    )


def build_setdlm_infill3_first_hitting_diagnostic_env(
    target: Target,
) -> dict[str, str]:
    return {
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "CKPT_FILE": target.ckpt_file,
        "USE_EMA": target.use_ema,
        "BLOCK_SIZE": target.eval_block_size,
        "MAX_WINDOW_SIZE": "32",
        "KV_CACHING": "true",
        "ALIGN_INPUTS_TO_BLOCKS": "false",
        "CACHE_FULL_INFILL_CONTEXT": "true",
        "INFILL_REPETITION_PENALTY_INCLUDE_RIGHT_CONTEXT": "false",
        "NUM_TARGET_SENTENCES": "3",
        "REPEAT_PENALTY": "1.5",
        "FIRST_HITTING": "false",
        "COMPILE_BACKBONE": "false",
        "SETDLM_INFILL_CACHE_PROMOTION_ORDER": _diag_cache_promotion_order(
            target
        ),
        "USE_EOS_STOPPING_CRITERIA": _diag_use_eos(target),
        "OUTPUT_TAG": _diag_output_tag(target),
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }


def infill_throughput_repeat_count() -> int:
    raw = os.environ.get("INFILL_THROUGHPUT_REPEATS", "1")
    try:
        count = int(raw)
    except ValueError:
        raise SystemExit(f"INFILL_THROUGHPUT_REPEATS must be an integer, got {raw!r}")
    if count < 1:
        raise SystemExit("INFILL_THROUGHPUT_REPEATS must be >= 1")
    return count


def infill_throughput_env(env: dict[str, str], target: Target) -> dict[str, str]:
    env.update(
        {
            "THROUGHPUT_RUN": "true",
            "THROUGHPUT_WARMUP": THROUGHPUT_WARMUP_EXAMPLES_PER_RANK,
            "THROUGHPUT_MEASUREMENTS": THROUGHPUT_MEASURED_EXAMPLES,
            "THROUGHPUT_GLOBAL_MEASUREMENTS": "true",
            "THROUGHPUT_SAMPLES_PER_RANK": THROUGHPUT_SAMPLES_PER_RANK,
            "NUM_VISIBLE_DEVICES": THROUGHPUT_NUM_GPUS,
        }
    )
    env.update(throughput_compile_env(target))
    return env


def build_infill1_throughput_env(target: Target) -> dict[str, str]:
    return infill_throughput_env(build_infill1_legacy_env(target), target)


def build_infill_throughput_env(target: Target) -> dict[str, str]:
    return infill_throughput_env(build_infill_legacy_env(target), target)


def build_cnndm_env(target: Target) -> dict[str, str]:
    env = {
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "BLOCK_SIZE": target.eval_block_size,
        "MAX_WINDOW_SIZE": target.max_window_size,
        "KV_CACHING": target.kv_caching,
        "ALIGN_INPUTS_TO_BLOCKS": target.align_inputs_to_blocks,
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }
    for key in ("REGULATION_START", "LEN_PENALTY", "REPETITION_PENALTY"):
        value = os.environ.get(key, "")
        if value:
            env[key] = value
    confidence_based_noising = os.environ.get(
        "CNNDM_CONFIDENCE_BASED_NOISING",
        os.environ.get("CONFIDENCE_BASED_NOISING", ""),
    ).strip()
    if confidence_based_noising:
        env["CONFIDENCE_BASED_NOISING"] = confidence_based_noising
    confidence_threshold = os.environ.get(
        "CNNDM_CONF_THRESHOLD",
        os.environ.get("CONF_THRESHOLD", ""),
    ).strip()
    if confidence_threshold:
        env["CONF_THRESHOLD"] = confidence_threshold
        env["CONFIDENCE_THRESHOLD"] = confidence_threshold
    if target.variant == "SetDLM":
        generate_target_prompt = os.environ.get(
            "CNNDM_SETDLM_GENERATE_TARGET_PROMPT", ""
        )
        if generate_target_prompt:
            env["CNNDM_GENERATE_TARGET_PROMPT"] = generate_target_prompt
    env.update(rank_invariant_generation_env())
    return env


def build_cnndm_throughput_env(target: Target) -> dict[str, str]:
    env = build_cnndm_env(target)
    env.update(
        {
            "THROUGHPUT_RUN": "true",
            "THROUGHPUT_WARMUP": THROUGHPUT_WARMUP_EXAMPLES_PER_RANK,
            "THROUGHPUT_MEASUREMENTS": THROUGHPUT_MEASURED_EXAMPLES,
            "THROUGHPUT_GLOBAL_MEASUREMENTS": "true",
            "NUM_VISIBLE_DEVICES": THROUGHPUT_NUM_GPUS,
        }
    )
    env.update(throughput_compile_env(target))
    env.update(setdlm_throughput_env(target))
    return env


def build_gsm8k_accuracy_env(target: Target) -> dict[str, str]:
    confidence_threshold = (
        target.confidence_threshold
        if target.confidence_threshold != "N/A"
        else "1e6"
    )
    confidence_based_noising = (
        target.confidence_based_noising
        if target.confidence_based_noising != "N/A"
        else "false"
    )
    env = {
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "BLOCK_SIZE": target.eval_block_size,
        "MAX_WINDOW_SIZE": target.max_window_size,
        "KV_CACHING": target.kv_caching,
        "ALIGN_INPUTS_TO_BLOCKS": target.align_inputs_to_blocks,
        "CONFIDENCE_BASED_NOISING": confidence_based_noising,
        "CONFIDENCE_THRESHOLD": confidence_threshold,
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
        # Full GSM8K reproduction uses the train/test parity settings.
        "MATCH_TRAINING_CONTEXT_LENGTH": "true",
        "STOP_ON_IM_END": "true",
    }
    env.update(rank_invariant_generation_env())
    return env


def build_gsm8k_pareto_accuracy_env(target: Target) -> dict[str, str]:
    env = build_gsm8k_accuracy_env(target)
    env.update(
        {
            # Pareto accuracy points were generated with the historical decode
            # settings: max_new_tokens=1024 and no im_end stopping criterion.
            "MATCH_TRAINING_CONTEXT_LENGTH": "false",
            "STOP_ON_IM_END": "false",
        }
    )
    return env


def build_mcqa_env(target: Target) -> dict[str, str]:
    return {
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "CKPT_FILE": target.ckpt_file,
        "USE_EMA": target.use_ema,
        "BLOCK_SIZE": target.eval_block_size,
        "COMPILE_BACKBONE": "true",
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }


def build_mauve_env(target: Target) -> dict[str, str]:
    env = {
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "CKPT_FILE": target.ckpt_file,
        "USE_EMA": target.use_ema,
        "BLOCK_SIZE": target.eval_block_size,
        "MAX_WINDOW_SIZE": target.max_window_size,
        "KV_CACHING": target.kv_caching,
        "ALIGN_INPUTS_TO_BLOCKS": target.align_inputs_to_blocks,
        "NUM_SAMPLES": "1000",
        "OUTPUT_NUM_SAMPLES": "1000",
        "MAUVE_REFERENCE_NUM_SAMPLES": "1000",
        "STOPPING_CONFIDENCE_THRESHOLD": "0.005",
        "REPETITION_PENALTY": "1.05",
        "NUCLEUS_P": "0.95",
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }
    if target.variant == "SEDD":
        env.update(
            {
                "MODEL_FAMILY": "SEDD",
                "GENERATION_CONFIG_NAME": "sedd_generation_config",
                "SAMPLING_STRATEGY": "analytic",
                "NOISE_REMOVAL": "true",
            }
        )
    return env


def build_throughput_owt_env(target: Target) -> dict[str, str]:
    env = build_mauve_env(target)
    env.update(
        {
            "THROUGHPUT_RUN": "true",
            "THROUGHPUT_WARMUP": THROUGHPUT_WARMUP_EXAMPLES_PER_RANK,
            "THROUGHPUT_MEASUREMENTS": THROUGHPUT_MEASURED_EXAMPLES,
            "THROUGHPUT_GLOBAL_MEASUREMENTS": "true",
            "THROUGHPUT_SAMPLES_PER_RANK": THROUGHPUT_SAMPLES_PER_RANK,
            "STOPPING_CONFIDENCE_THRESHOLD": "null",
            "NUM_VISIBLE_DEVICES": THROUGHPUT_NUM_GPUS,
            "NUM_SAMPLES": THROUGHPUT_MEASURED_EXAMPLES,
            "OUTPUT_NUM_SAMPLES": THROUGHPUT_MEASURED_EXAMPLES,
        }
    )
    env.update(throughput_compile_env(target))
    env.update(setdlm_throughput_env(target))
    return env


def table4_decode_overrides(target: Target) -> dict[str, str]:
    if target.variant == "AR":
        return {
            "NUCLEUS_P": "0.90",
            "REPETITION_PENALTY": "1.00",
            "STOPPING_CONFIDENCE_THRESHOLD": "0.005",
        }
    if target.label in {"BD3LM-s16", "SetDLM-smax32"}:
        return {
            "NUCLEUS_P": "0.95",
            "REPETITION_PENALTY": "1.05",
            "STOPPING_CONFIDENCE_THRESHOLD": "0.01",
        }
    return {}


def build_mauve_owt_table4_env(target: Target) -> dict[str, str]:
    env = build_mauve_env(target)
    env.update(table4_decode_overrides(target))
    env.update(rank_invariant_generation_env())
    return env


def build_throughput_owt_table4_env(target: Target) -> dict[str, str]:
    env = build_mauve_owt_table4_env(target)
    env.update(
        {
            "THROUGHPUT_RUN": "true",
            "THROUGHPUT_WARMUP": THROUGHPUT_WARMUP_EXAMPLES_PER_RANK,
            "THROUGHPUT_MEASUREMENTS": THROUGHPUT_MEASURED_EXAMPLES,
            "THROUGHPUT_GLOBAL_MEASUREMENTS": "true",
            "THROUGHPUT_SAMPLES_PER_RANK": THROUGHPUT_SAMPLES_PER_RANK,
            "NUM_VISIBLE_DEVICES": THROUGHPUT_NUM_GPUS,
            "NUM_SAMPLES": THROUGHPUT_MEASURED_EXAMPLES,
            "OUTPUT_NUM_SAMPLES": THROUGHPUT_MEASURED_EXAMPLES,
        }
    )
    env.update(throughput_compile_env(target))
    env.update(setdlm_throughput_env(target))
    return env


def table4_bootstrap_label(target: Target) -> str:
    if target.variant == "AR":
        return "ar_p090_rp100_stop0005"
    if target.variant == "BD3LM":
        return f"bd3lm_s{target.block_size}"
    if target.variant == "SetDLM":
        if target.desired_block_size != "N/A":
            return f"setdlm_smax{2 * int(target.desired_block_size)}"
        return "setdlm_smaxN/A"
    return safe_id(target.label).lower()


def build_mauve_owt_table4_bootstrap_env(target: Target) -> dict[str, str]:
    output_dir = DEFAULT_OUTPUT_ROOT / "owt_table4_bootstrap_mauve"
    return {
        "TABLE4_LABEL": table4_bootstrap_label(target),
        "TABLE4_INPUT": f"{build_mauve_owt_table4_output(target)}/generated_samples.json",
        "TABLE4_OUTPUT_DIR": str(output_dir),
        "TABLE4_DATASET_PATH": OWT_TABLE4_DATASET_PATH,
        "TABLE4_REFERENCE_SUBSETS_JSON": str(
            output_dir
            / f"reference_subsets_seed{OWT_TABLE4_REFERENCE_SEED}_n{OWT_TABLE4_REFERENCE_SIZE}_k{OWT_TABLE4_REFERENCE_SUBSETS}.json"
        ),
        "TABLE4_REFERENCE_SIZE": OWT_TABLE4_REFERENCE_SIZE,
        "TABLE4_REFERENCE_SUBSETS": OWT_TABLE4_REFERENCE_SUBSETS,
        "TABLE4_REFERENCE_SEED": OWT_TABLE4_REFERENCE_SEED,
        "TABLE4_MAUVE_SEED": OWT_TABLE4_MAUVE_SEED,
        "TABLE4_MAX_GENERATED": "1000",
        "NUM_VISIBLE_DEVICES": target.num_visible_devices,
    }


def build_mauve_owt_table4_bootstrap_output(target: Target) -> str:
    return str(DEFAULT_OUTPUT_ROOT / "owt_table4_bootstrap_mauve" / f"{table4_bootstrap_label(target)}_summary.json")


def build_throughput_lm1b_env(target: Target) -> dict[str, str]:
    env = build_lm1b_uncond_env(target)
    env.update(
        {
            "THROUGHPUT_RUN": "true",
            "THROUGHPUT_WARMUP": THROUGHPUT_WARMUP_EXAMPLES_PER_RANK,
            "THROUGHPUT_MEASUREMENTS": THROUGHPUT_MEASURED_EXAMPLES,
            "THROUGHPUT_GLOBAL_MEASUREMENTS": "true",
            "THROUGHPUT_SAMPLES_PER_RANK": THROUGHPUT_SAMPLES_PER_RANK,
            "STOPPING_CONFIDENCE_THRESHOLD": "null",
            "NUM_VISIBLE_DEVICES": THROUGHPUT_NUM_GPUS,
            "NUM_SAMPLES": THROUGHPUT_MEASURED_EXAMPLES,
            "OUTPUT_NUM_SAMPLES": THROUGHPUT_MEASURED_EXAMPLES,
        }
    )
    env.update(throughput_compile_env(target))
    env.update(setdlm_throughput_env(target))
    return env


def build_gsm8k_pareto_throughput_env(target: Target) -> dict[str, str]:
    confidence_threshold = (
        target.confidence_threshold
        if target.confidence_threshold != "N/A"
        else "1e6"
    )
    if target.variant == "SetDLM" and confidence_threshold != "1e6":
        confidence_based_noising = "true"
    else:
        confidence_based_noising = (
            target.confidence_based_noising
            if target.confidence_based_noising != "N/A"
            else "false"
        )
    env = {
        "MODEL_PATH": resolve_model_path_for_eval(target.model_path),
        "BLOCK_SIZE": target.eval_block_size,
        "MAX_WINDOW_SIZE": target.max_window_size,
        "KV_CACHING": target.kv_caching,
        "ALIGN_INPUTS_TO_BLOCKS": target.align_inputs_to_blocks,
        "CONFIDENCE_BASED_NOISING": confidence_based_noising,
        "CONFIDENCE_THRESHOLD": confidence_threshold,
        "THROUGHPUT_RUN": "true",
        "THROUGHPUT_WARMUP": THROUGHPUT_WARMUP_EXAMPLES_PER_RANK,
        "THROUGHPUT_MEASUREMENTS": THROUGHPUT_MEASURED_EXAMPLES,
        "THROUGHPUT_GLOBAL_MEASUREMENTS": "true",
        "THROUGHPUT_SAMPLES_PER_RANK": THROUGHPUT_SAMPLES_PER_RANK,
        "NUM_VISIBLE_DEVICES": THROUGHPUT_NUM_GPUS,
    }
    env.update(throughput_compile_env(target))
    env.update(setdlm_throughput_env(target))
    return env


def build_infill_output_for(
    target: Target,
    num_target_sentences: str,
    repeat_penalty: str = "1.1",
    align_inputs_to_blocks: str | None = None,
    output_tag: str = "",
) -> str:
    align = align_inputs_to_blocks or infill_align_inputs_to_blocks(target)
    tag = f"-{output_tag}" if output_tag else ""
    suffix = (
        f"roc_stories/num_target_sentences{num_target_sentences}/"
        f"L-1024-block_size-{target.eval_block_size}-T{target.eval_block_size}-"
        "do_sample-false-sampling_strategy-predict_and_noise-"
        f"repeat_penalty{repeat_penalty}-first_hitting-false-confidence_based_noising-false-"
        f"align_inputs_to_blocks{align}-"
        f"ckpt-{target.ckpt_file}-ema{target.use_ema}{tag}"
    )
    return output_under_model(target.model_path, suffix)


def build_infill_output(target: Target) -> str:
    return build_infill_output_for(
        target,
        "3",
        repeat_penalty=infill3_repeat_penalty(),
        output_tag=infill_output_tag(),
    )


def build_infill1_output(target: Target) -> str:
    return build_infill_output_for(
        target,
        "1",
        repeat_penalty=infill1_repeat_penalty(),
        output_tag=infill_output_tag(),
    )


def build_infill_legacy_output(target: Target) -> str:
    if target.variant == "SetDLM" and target.desired_block_size == "16":
        setdlm_max_window_size = os.environ.get(
            "INFILL_SETDLM_MAX_WINDOW_SIZE", ""
        ).strip()
        if setdlm_max_window_size:
            output_tag = infill_output_tag(f"legacy_mw{setdlm_max_window_size}")
        else:
            output_tag = infill_output_tag("legacy_mw32")
    else:
        output_tag = infill_output_tag("legacy_qualitative")
    return build_infill_output_for(
        target,
        "3",
        repeat_penalty=infill3_repeat_penalty(),
        align_inputs_to_blocks=legacy_infill_align_inputs_to_blocks(target),
        output_tag=output_tag,
    )


def build_infill1_legacy_output(target: Target) -> str:
    if target.variant == "SetDLM" and target.desired_block_size == "16":
        setdlm_max_window_size = os.environ.get(
            "INFILL_SETDLM_MAX_WINDOW_SIZE", ""
        ).strip()
        if setdlm_max_window_size:
            output_tag = infill_output_tag(f"legacy_mw{setdlm_max_window_size}")
        else:
            output_tag = infill_output_tag("legacy_mw32")
    else:
        output_tag = infill_output_tag("legacy_qualitative")
    return build_infill_output_for(
        target,
        "1",
        repeat_penalty=infill1_repeat_penalty(),
        align_inputs_to_blocks=legacy_infill_align_inputs_to_blocks(target),
        output_tag=output_tag,
    )


def build_setdlm_infill3_first_hitting_diagnostic_output(target: Target) -> str:
    suffix = (
        "roc_stories/num_target_sentences3/"
        f"L-1024-block_size-{target.eval_block_size}-T{target.eval_block_size}-"
        "do_sample-false-sampling_strategy-predict_and_noise-"
        "repeat_penalty1.5-first_hitting-false-"
        "confidence_based_noising-false-align_inputs_to_blocksfalse-"
        f"ckpt-{target.ckpt_file}-ema{target.use_ema}-{_diag_output_tag(target)}"
    )
    return output_under_model(target.model_path, suffix)


def build_cnndm_output(target: Target) -> str:
    generate_target_prompt_enabled = os.environ.get(
        "CNNDM_SETDLM_GENERATE_TARGET_PROMPT", "false"
    ).lower() == "true"
    target_prompt_suffix = (
        "_gen-target-prompttrue"
        if target.variant == "SetDLM" and generate_target_prompt_enabled
        else ""
    )
    suffix = (
        "cnn_dailymail_t1/"
        f"L-null-block_size-{target.eval_block_size}-do_sample-false-"
        "sampling_strategy-predict_and_noise-first_hitting-false-"
        "confidence_based_noising-false-"
        f"align_inputs_to_blocks{target.align_inputs_to_blocks}-"
        "ckptbest-ematruerep-penalty-1.2_len-penalty-1.1_reg-start80"
        f"{target_prompt_suffix}"
    )
    return output_under_model(target.model_path, suffix)


def _build_gsm8k_lm_eval_output(
    target: Target,
    *,
    match_training_context_length: bool,
    stop_on_im_end: bool,
) -> str:
    confidence_threshold = (
        target.confidence_threshold
        if target.confidence_threshold != "N/A"
        else "1e6"
    )
    confidence_based_noising = (
        target.confidence_based_noising
        if target.confidence_based_noising != "N/A"
        else "false"
    )
    suffix = (
        "lm_eval_harness_output/"
        f"ematrue_ckptbest_L1024_block{target.eval_block_size}-"
        "do_samplefalse-sampling_strategypredict_and_noise-"
        f"T{target.eval_block_size}_first_hitfalse-conf_noise{confidence_based_noising}-"
        f"conf_margin_noisefalse-conf_thold{confidence_threshold}-"
        f"align_to_blocks{target.align_inputs_to_blocks}-"
        f"max_window_size{target.max_window_size}"
    )
    if match_training_context_length:
        suffix += "_match_train_len"
    if stop_on_im_end:
        suffix += "_stop_im_end"
    suffix += "_test"
    return output_under_model(target.model_path, suffix)


def build_gsm8k_accuracy_output(target: Target) -> str:
    return _build_gsm8k_lm_eval_output(
        target,
        match_training_context_length=True,
        stop_on_im_end=True,
    )


def build_gsm8k_pareto_accuracy_output(target: Target) -> str:
    return _build_gsm8k_lm_eval_output(
        target,
        match_training_context_length=False,
        stop_on_im_end=False,
    )


def build_gsm8k_pareto_throughput_output(target: Target) -> str:
    confidence_threshold = (
        target.confidence_threshold
        if target.confidence_threshold != "N/A"
        else "1e6"
    )
    confidence_based_noising = (
        target.confidence_based_noising
        if target.confidence_based_noising != "N/A"
        else "false"
    )
    suffix = (
        "lm_eval_harness_output/"
        f"ematrue_ckptbest_0shot_L1024_block{target.eval_block_size}-"
        "do_samplefalse-sampling_strategypredict_and_noise-"
        f"T{target.eval_block_size}_first_hitfalse-conf_noise{confidence_based_noising}-"
        f"conf_margin_noisefalse-conf_thold{confidence_threshold}-"
        f"align_to_blocks{target.align_inputs_to_blocks}"
    )
    return output_under_model(target.model_path, suffix)


def build_mcqa_output(target: Target) -> str:
    suffix = (
        "mcqa_eval_output/"
        f"task-all_ckpt-{target.ckpt_file}_ema-{target.use_ema}_"
        f"block-{target.eval_block_size}_maxlen-1024_maxexamples-null_is-1"
    )
    return output_under_model(target.model_path, suffix)


def build_mauve_output(target: Target) -> str:
    suffix = (
        "owt-L-1024-NUM_SAMPLES1000/"
        f"block_size-{target.eval_block_size}-T{target.eval_block_size}-"
        f"sampling_strategy-{'analytic' if target.variant == 'SEDD' else 'predict_and_noise'}-"
        f"noise_removal-{'true' if target.variant == 'SEDD' else 'false'}-"
        "do_sample-true-first_hitting-false-"
        f"align_inputs_to_blocks{target.align_inputs_to_blocks}-"
        f"ckpt{target.ckpt_file}-ema{target.use_ema}-"
        "nucleus_p0.95-repetition_penalty1.05-conf1e6-"
        f"max_window_size{target.max_window_size}"
    )
    suffix += "-stop_conf0.005-stop_win128-stop_min128-stop_patience4"
    return output_under_model(target.model_path, suffix)


def build_owt_uncond_output_for_env(target: Target, env: dict[str, str]) -> str:
    sampling_strategy = "analytic" if target.variant == "SEDD" else "predict_and_noise"
    noise_removal = "true" if target.variant == "SEDD" else "false"
    output_num_samples = env.get("OUTPUT_NUM_SAMPLES", env.get("NUM_SAMPLES", "1000"))
    suffix = (
        f"owt-L-1024-NUM_SAMPLES{output_num_samples}/"
        f"block_size-{env.get('BLOCK_SIZE', target.eval_block_size)}-"
        f"T{env.get('GENERATION_NUM_STEPS', env.get('BLOCK_SIZE', target.eval_block_size))}-"
        f"sampling_strategy-{env.get('SAMPLING_STRATEGY', sampling_strategy)}-"
        f"noise_removal-{env.get('NOISE_REMOVAL', noise_removal)}-"
        "do_sample-true-first_hitting-false-"
        f"align_inputs_to_blocks{env.get('ALIGN_INPUTS_TO_BLOCKS', target.align_inputs_to_blocks)}-"
        f"ckpt{env.get('CKPT_FILE', target.ckpt_file)}-ema{env.get('USE_EMA', target.use_ema)}-"
        f"nucleus_p{env.get('NUCLEUS_P', '0.95')}-"
        f"repetition_penalty{env.get('REPETITION_PENALTY', '1.05')}-"
        f"conf{env.get('CONFIDENCE_THRESHOLD', '1e6')}-"
        f"max_window_size{env.get('MAX_WINDOW_SIZE', target.max_window_size)}"
    )
    stop_conf = env.get("STOPPING_CONFIDENCE_THRESHOLD", "0.005")
    if stop_conf != "null":
        suffix += (
            f"-stop_conf{stop_conf}-"
            f"stop_win{env.get('STOPPING_CONFIDENCE_WINDOW', '128')}-"
            f"stop_min{env.get('STOPPING_CONFIDENCE_MIN_TOKENS', '128')}-"
            f"stop_patience{env.get('STOPPING_CONFIDENCE_PATIENCE', '4')}"
        )
    if env.get("THROUGHPUT_RUN") == "true":
        suffix += "-throughput_run"
    if env.get("LOW_ENTROPY_TRUNCATION", "legacy") != "legacy":
        suffix += f"-low_entropy_{env['LOW_ENTROPY_TRUNCATION']}"
    if target.variant == "SetDLM":
        max_block_size = env.get("MAX_BLOCK_SIZE")
        if not max_block_size and target.desired_block_size != "N/A":
            max_block_size = str(2 * int(target.desired_block_size))
        if max_block_size:
            suffix += f"-noise_max_block_size{max_block_size}"
    return output_under_model(env.get("MODEL_PATH", target.model_path), suffix)


def build_mauve_owt_table4_output(target: Target) -> str:
    return build_owt_uncond_output_for_env(target, build_mauve_owt_table4_env(target))


def build_throughput_owt_table4_output(target: Target) -> str:
    return build_owt_uncond_output_for_env(target, build_throughput_owt_table4_env(target))


def sedd_owt_target() -> Target:
    return Target(
        variant="SEDD",
        model_path="kuleshov-group/sedd-noeos-owt",
        ckpt_file="none",
        use_ema="false",
        eval_block_size="1024",
        max_window_size="1024",
        kv_caching="false",
        align_inputs_to_blocks="false",
    )


def missing_sedd() -> Target:
    return Target(
        variant="SEDD",
        model_path="N/A",
        ckpt_file="N/A",
        use_ema="N/A",
        eval_block_size="N/A",
        max_window_size="N/A",
        status="missing checkpoint/config",
        reason=(
            "No compatible SEDD checkpoint/config found; configs/model/sedd.yaml is "
            "absent and no sedd-* eval checkpoint was discovered."
        ),
    )


def owt_targets() -> list[Target]:
    return [
        Target(
            "AR",
            str(TEXTDIFFUSION_CKPT_DIR / "mari-owt-ar-noeos-v4-1/20-300000.ckpt"),
            "20-300000.ckpt",
            "true",
            "1",
            "1",
            kv_caching="true",
            align_inputs_to_blocks="true",
        ),
        Target(
            "MDLM",
            str(TEXTDIFFUSION_CKPT_DIR / "mari-owt-mdlm-noeos-v4/18-300000.ckpt"),
            "18-300000.ckpt",
            "true",
            "1024",
            "1024",
            kv_caching="false",
            align_inputs_to_blocks="false",
        ),
        *[
            Target(
                "BD3LM",
                str(
                    BASE_RUN_DIR
                    / (
                        f"owt_block{size}_lr3e-4_bsz512_warm2500ba_layers12_"
                        "hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"
                    )
                ),
                "ep17-ba300000-rank0.pt",
                "false",
                str(size),
                str(size),
                block_size=str(size),
                kv_caching="true",
                align_inputs_to_blocks="false",
            )
            for size in (4, 8, 16)
        ],
        *[
            Target(
                "SetDLM",
                str(
                    BASE_RUN_DIR
                    / (
                        "owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_"
                        f"hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block{size}_vscratch"
                    )
                ),
                "ep17-ba300000-rank0.pt",
                "false",
                "1024",
                str(size),
                desired_block_size=str(size),
                kv_caching="true",
                align_inputs_to_blocks="false",
            )
            for size in (4, 8, 16)
        ],
        sedd_owt_target(),
    ]


def legacy_infill_targets() -> list[Target]:
    keep = {"AR", "MDLM", "BD3LM-s16", "SetDLM-smax32"}
    return [target for target in owt_targets() if target.label in keep]


def owt_table4_targets() -> list[Target]:
    return [
        Target(
            "AR",
            "kuleshov-group/owt-ar",
            "none",
            "false",
            "1",
            "1",
            kv_caching="true",
            align_inputs_to_blocks="true",
        ),
        Target(
            "BD3LM",
            "kuleshov-group/owt-bd3lm-s16",
            "none",
            "false",
            "16",
            "16",
            block_size="16",
            kv_caching="true",
            align_inputs_to_blocks="false",
        ),
        Target(
            "SetDLM",
            "kuleshov-group/owt-setdlm-smax32",
            "none",
            "false",
            "1024",
            "16",
            desired_block_size="16",
            kv_caching="true",
            align_inputs_to_blocks="false",
        ),
    ]


def setdlm_infill3_first_hitting_diagnostic_targets() -> list[Target]:
    base = (
        "owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_"
        "hidden768_inter3072_aoarm_normlayernorm_adalnfalse"
    )
    specs = [
        (
            "old",
            str(BASE_RUN_DIR / f"{base}_block16_ft_v5"),
            "best-rank0.pt",
            "true",
            "true",
        ),
        (
            "current",
            str(BASE_RUN_DIR / f"{base}_block16_vscratch"),
            "ep17-ba300000-rank0.pt",
            "false",
            "false",
        ),
    ]
    targets: list[Target] = []
    for family, model_path, ckpt_file, use_ema, use_eos in specs:
        for cache_promotion_order in ("l2r", "first_hitting"):
            targets.append(
                Target(
                    f"SetDLM-{family}-cache-{cache_promotion_order}-eos{use_eos}",
                    model_path,
                    ckpt_file,
                    use_ema,
                    "1024",
                    "32",
                    desired_block_size="16",
                    kv_caching="true",
                    align_inputs_to_blocks="false",
                )
            )
    return targets


def lm1b_targets() -> list[Target]:
    return [
        Target(
            "AR",
            str(BASE_RUN_DIR / "lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter_ar_hparam_v1"),
            "best-rank0.pt",
            "true",
            "1",
            "1",
        ),
        Target(
            "MDLM",
            str(BASE_RUN_DIR / "lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_mdlm_adaln_cleanbos_antithetic_hparam_v2"),
            "ep72-ba1000000-rank0.pt",
            "false",
            "128",
            "128",
            kv_caching="false",
        ),
        *[
            Target(
                "BD3LM",
                str(
                    BASE_RUN_DIR
                    / (
                        f"lm1b_block{size}_lr3e-4_bsz512_warm2500ba_layers12_"
                        "hidden768_inter3072_bd3lm_dropout0.1_normlayernorm_hparam_v3"
                    )
                ),
                "best-rank0.pt" if size == 8 else "ep72-ba1000000-rank0.pt",
                "true" if size == 8 else "false",
                str(size),
                str(size),
                block_size=str(size),
            )
            for size in (4, 8, 16)
        ],
        *[
            Target(
                "SetDLM",
                str(
                    BASE_RUN_DIR
                    / (
                        "lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_"
                        f"hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired{size}"
                        + ("_vlambda" if size == 16 else "_max128_v5")
                    )
                ),
                "best-rank0.pt" if size == 16 else "ep72-ba1000000-rank0.pt",
                "true" if size == 16 else "false",
                "128",
                str(size),
                desired_block_size=str(size),
                align_inputs_to_blocks="false",
            )
            for size in (4, 8, 16)
        ],
        missing_sedd(),
    ]


def cnndm_targets() -> list[Target]:
    return [
        Target(
            "AR",
            str(BASE_RUN_DIR / "cnn_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_ar_len768_v1"),
            "best-rank0.pt",
            "true",
            "1",
            "1",
        ),
        Target(
            "MDLM",
            str(BASE_RUN_DIR / "cnn_block_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_mdlm_len768_v1"),
            "best-rank0.pt",
            "true",
            "32",
            "32",
        ),
        *[
            Target(
                "BD3LM",
                str(
                    BASE_RUN_DIR
                    / (
                        f"cnn_block{size}_lr3e-4_bsz128_warm1000ba_layers28_"
                        "hidden256_inter768_bd3lm_len768_v1"
                    )
                ),
                "best-rank0.pt",
                "true",
                str(size),
                str(size),
                block_size=str(size),
            )
            for size in (4, 8, 16)
        ],
        *[
            Target(
                "SetDLM",
                str(
                    BASE_RUN_DIR
                    / (
                        "cnn_block768_lr3e-4_bsz128_warm1000ba_layers28_"
                        f"hidden256_inter768_aoarm_tgt{size}_len768_v1"
                    )
                ),
                "best-rank0.pt",
                "true",
                "768",
                str(size),
                desired_block_size=str(size),
                align_inputs_to_blocks="false",
            )
            for size in (4, 8, 16)
        ],
        missing_sedd(),
    ]


def cnndm_throughput_targets() -> list[Target]:
    return [target for target in cnndm_targets() if target.status == "ready"]


def gsm8k_targets() -> list[Target]:
    return [
        Target(
            "AR",
            "gsm8k:ar",
            "best-rank0.pt",
            "true",
            "1",
            "1",
        ),
        Target(
            "MDLM",
            "gsm8k:mdlm",
            "best-rank0.pt",
            "true",
            "32",
            "32",
            kv_caching="false",
        ),
        *[
            Target(
                "BD3LM",
                f"gsm8k:bd3lm-s{size}",
                "best-rank0.pt",
                "true",
                str(size),
                str(size),
                block_size=str(size),
            )
            for size in (4, 8, 16)
        ],
        *[
            Target(
                "SetDLM",
                f"gsm8k:setdlm-smax{2 * size}",
                "best-rank0.pt",
                "true",
                "1024",
                str(size),
                desired_block_size=str(size),
                align_inputs_to_blocks="false",
            )
            for size in (4, 8, 16)
        ],
        missing_sedd(),
    ]


def _gsm8k_pareto_group(target: Target) -> str:
    if target.variant == "BD3LM":
        return f"Block Diffusion (S={target.block_size})"
    if target.variant == "SetDLM":
        return f"Set Diffusion (match w/ S={target.desired_block_size})"
    return target.variant


def gsm8k_pareto_targets() -> list[Target]:
    targets: list[Target] = []
    for target in gsm8k_targets():
        if target.status != "ready":
            targets.append(
                replace(
                    target,
                    confidence_threshold="N/A",
                    confidence_based_noising="N/A",
                    pareto_group=_gsm8k_pareto_group(target),
                )
            )
            continue
        if target.variant == "AR":
            thresholds = ("1e6",)
        elif target.variant in {"BD3LM", "SetDLM"}:
            thresholds = GSM8K_PARETO_THRESHOLDS
        elif target.variant == "MDLM":
            targets.append(
                replace(
                    target,
                    status="unsupported",
                    reason=(
                        "GSM8K Pareto plotting script defines an AR baseline and "
                        "BD3LM/SetDLM block frontiers; no MDLM threshold frontier "
                        "mapping is defined."
                    ),
                    confidence_threshold="N/A",
                    confidence_based_noising="N/A",
                    pareto_group=_gsm8k_pareto_group(target),
                )
            )
            continue
        else:
            continue
        for threshold in thresholds:
            targets.append(
                replace(
                    target,
                    confidence_threshold=threshold,
                    confidence_based_noising="false",
                    pareto_group=_gsm8k_pareto_group(target),
                )
            )
    return targets


TASKS: list[TaskSpec] = [
    TaskSpec(
        "owt_ppl",
        "bash_scripts/run_likelihood_eval_owt.sh",
        "owt",
        build_owt_ppl_env,
        likelihood_stdout_only,
    ),
    TaskSpec(
        "lm1b_ppl",
        "bash_scripts/run_likelihood_eval_lm1b.sh",
        "lm1b",
        build_lm1b_ppl_env,
        likelihood_stdout_only,
    ),
    TaskSpec(
        "infilling1",
        "bash_scripts/run_seq2seq_eval_infill_nlp.sh",
        "owt",
        build_infill1_env,
        build_infill1_output,
    ),
    TaskSpec(
        "infilling1_legacy",
        "bash_scripts/run_seq2seq_eval_infill_nlp.sh",
        "owt_infill_legacy",
        build_infill1_legacy_env,
        build_infill1_legacy_output,
    ),
    TaskSpec(
        "infilling",
        "bash_scripts/run_seq2seq_eval_infill_nlp.sh",
        "owt",
        build_infill_env,
        build_infill_output,
    ),
    TaskSpec(
        "infilling_legacy",
        "bash_scripts/run_seq2seq_eval_infill_nlp.sh",
        "owt_infill_legacy",
        build_infill_legacy_env,
        build_infill_legacy_output,
    ),
    TaskSpec(
        "setdlm_infill3_first_hitting_diagnostic",
        "bash_scripts/run_seq2seq_eval_infill_nlp.sh",
        "setdlm_infill3_first_hitting_diagnostic",
        build_setdlm_infill3_first_hitting_diagnostic_env,
        build_setdlm_infill3_first_hitting_diagnostic_output,
    ),
    TaskSpec(
        "infilling1_throughput",
        "bash_scripts/run_seq2seq_eval_infill_nlp.sh",
        "owt_infill_legacy",
        build_infill1_throughput_env,
        build_infill1_legacy_output,
    ),
    TaskSpec(
        "infilling_throughput",
        "bash_scripts/run_seq2seq_eval_infill_nlp.sh",
        "owt_infill_legacy",
        build_infill_throughput_env,
        build_infill_legacy_output,
    ),
    TaskSpec(
        "cnndm_summarization",
        "bash_scripts/run_seq2seq_eval_cnndm.sh",
        "cnndm",
        build_cnndm_env,
        build_cnndm_output,
    ),
    TaskSpec(
        "cnndm_summarization_throughput",
        "bash_scripts/run_seq2seq_eval_cnndm.sh",
        "cnndm_throughput",
        build_cnndm_throughput_env,
        build_cnndm_output,
    ),
    TaskSpec(
        "gsm8k_accuracy",
        "bash_scripts/run_lm_eval_harness.sh",
        "gsm8k",
        build_gsm8k_accuracy_env,
        build_gsm8k_accuracy_output,
    ),
    TaskSpec(
        "gsm8k_pareto_accuracy",
        "bash_scripts/run_lm_eval_harness.sh",
        "gsm8k_pareto",
        build_gsm8k_pareto_accuracy_env,
        build_gsm8k_pareto_accuracy_output,
    ),
    TaskSpec(
        "gsm8k_pareto_throughput",
        "bash_scripts/run_lm_eval_harness_tput.sh",
        "gsm8k_pareto",
        build_gsm8k_pareto_throughput_env,
        build_gsm8k_pareto_throughput_output,
    ),
    TaskSpec(
        "gsm8k_ppl",
        "bash_scripts/run_likelihood_eval_gsm8k.sh",
        "gsm8k",
        build_gsm8k_ppl_env,
        likelihood_stdout_only,
    ),
    TaskSpec(
        "commonsense",
        "bash_scripts/run_mcqa_eval_owt.sh",
        "owt",
        build_mcqa_env,
        build_mcqa_output,
    ),
    TaskSpec(
        "mauve",
        "bash_scripts/run_uncond_gen_ppl_owt.sh",
        "owt",
        build_mauve_env,
        build_mauve_output,
    ),
    TaskSpec(
        "mauve_owt_table4",
        "bash_scripts/run_uncond_gen_ppl_owt.sh",
        "owt_table4",
        build_mauve_owt_table4_env,
        build_mauve_owt_table4_output,
    ),
    TaskSpec(
        "mauve_owt_table4_bootstrap",
        "bash_scripts/run_mauve_owt_table4_bootstrap.sh",
        "owt_table4",
        build_mauve_owt_table4_bootstrap_env,
        build_mauve_owt_table4_bootstrap_output,
    ),
    TaskSpec(
        "throughput_owt",
        "bash_scripts/run_uncond_gen_ppl_owt.sh",
        "owt",
        build_throughput_owt_env,
        build_mauve_output,
    ),
    TaskSpec(
        "throughput_owt_table4",
        "bash_scripts/run_uncond_gen_ppl_owt.sh",
        "owt_table4",
        build_throughput_owt_table4_env,
        build_throughput_owt_table4_output,
    ),
    TaskSpec(
        "throughput_lm1b",
        "bash_scripts/run_uncond_gen_ppl_lm1b.sh",
        "lm1b",
        build_throughput_lm1b_env,
        build_mauve_output,
    ),
]


GROUP_TARGETS = {
    "owt": owt_targets,
    "owt_table4": owt_table4_targets,
    "owt_infill_legacy": legacy_infill_targets,
    "setdlm_infill3_first_hitting_diagnostic": setdlm_infill3_first_hitting_diagnostic_targets,
    "lm1b": lm1b_targets,
    "cnndm": cnndm_targets,
    "cnndm_throughput": cnndm_throughput_targets,
    "gsm8k": gsm8k_targets,
    "gsm8k_pareto": gsm8k_pareto_targets,
}


def verify_target(target: Target) -> tuple[str, str]:
    if target.status != "ready":
        return target.status, target.reason
    if is_remote_or_known_eval_model(target.model_path):
        return "ready", ""
    model_path = Path(target.model_path)
    if model_path.suffix == ".ckpt":
        if not model_path.exists():
            return "missing checkpoint/config", f"checkpoint file not found: {model_path}"
        return "ready", ""
    if not model_path.exists():
        return "missing checkpoint/config", f"model directory not found: {model_path}"
    if not (model_path / "config.yaml").exists():
        return "missing checkpoint/config", f"config.yaml not found in {model_path}"
    ckpt_candidates = [
        model_path / "checkpoints" / target.ckpt_file,
        model_path / target.ckpt_file,
    ]
    if not any(path.exists() for path in ckpt_candidates):
        return (
            "missing checkpoint/config",
            f"checkpoint {target.ckpt_file} not found in {model_path}/checkpoints or model dir",
        )
    return "ready", ""


def build_command(mode: str, task: TaskSpec, env: dict[str, str]) -> str:
    if mode == "sbatch":
        command = f"bash bash_scripts/sbatch_wrapper.sh {q(Path(task.script).name)}"
    else:
        command = f"bash {q(task.script)}"
    return f"cd {q(str(REPO_ROOT))} && {env_prefix(env)} {command}"


def build_local_command(task: TaskSpec, env: dict[str, str]) -> str:
    return f"cd {q(str(REPO_ROOT))} && {env_prefix(env)} bash {q(task.script)}"


def build_row(
    task: TaskSpec,
    target: Target,
    suite_dir: Path,
    mode: str,
    repeat_idx: int | None = None,
) -> MatrixRow:
    status, reason = verify_target(target)
    if status == "ready" and task.name == "lm1b_ppl" and not LM1B_EVAL_DATASET.exists():
        status = "missing checkpoint/config"
        reason = f"required LM1B eval dataset not found: {LM1B_EVAL_DATASET}"
    env = task.env_builder(target) if status == "ready" else {}
    if status == "ready" and os.environ.get("GPU_EXCLUDE_EXTRA"):
        env["GPU_EXCLUDE_EXTRA"] = os.environ["GPU_EXCLUDE_EXTRA"]
    num_gpus = "N/A"
    gpu_constraint = "N/A"
    if status == "ready":
        num_gpus, gpu_constraint = apply_resource_policy(env, task, target)
        if task.name in {
            "infilling",
            "infilling1",
            "infilling_legacy",
            "infilling1_legacy",
        }:
            forced_num_gpus = os.environ.get("INFILL_NUM_VISIBLE_DEVICES", "").strip()
            forced_constraint = os.environ.get("INFILL_GPU_CONSTRAINT", "").strip()
            forced_partition = os.environ.get("INFILL_GPU_PARTITION", "").strip()
            forced_nodelist = os.environ.get("INFILL_GPU_NODELIST", "").strip()
            if forced_num_gpus:
                env["NUM_VISIBLE_DEVICES"] = forced_num_gpus
                num_gpus = forced_num_gpus
            if forced_constraint:
                env["GPU_CONSTRAINT"] = forced_constraint
                gpu_constraint = forced_constraint
            if forced_partition:
                env["GPU_PARTITION"] = forced_partition
            if forced_nodelist:
                env["GPU_NODELIST"] = forced_nodelist
    if (
        status == "ready"
        and task.name in THROUGHPUT_TASKS
        and task.name not in INFILL_THROUGHPUT_TASKS
        and target.variant == "SetDLM"
    ):
        row_id = safe_id(f"{task.name}_{setdlm_output_id(target, env)}")
    else:
        row_id = safe_id(f"{task.name}_{target.label}")
    if repeat_idx is not None:
        row_id = safe_id(f"{row_id}_rep{repeat_idx}")
    output_path = task.output_builder(target) if status == "ready" else "N/A"
    if status == "ready" and task.name in {
        "mauve",
        "infilling",
        "infilling1",
        "infilling_legacy",
        "infilling1_legacy",
        *THROUGHPUT_TASKS,
    }:
        output_path = str(suite_dir / "outputs" / row_id)
        env["OUTPUT_PATH_OVERRIDE"] = output_path
    if mode == "sbatch":
        log_path = str(REPO_ROOT / "watch_folder" / f"{script_job_name(task.script)}_<jobid>.log")
    else:
        log_path = str(suite_dir / "logs" / f"{row_id}.log")
    metadata_path = str(suite_dir / "metadata" / f"{row_id}.json") if status == "ready" else "N/A"
    if status == "ready":
        env["REPRO_ROW_ID"] = row_id
        env["REPRO_METADATA_PATH"] = metadata_path
        env["REPRO_OUTPUT_ROOT"] = str(suite_dir)
        env["REPRO_LOG_PATH_TEMPLATE"] = log_path
        if repeat_idx is not None:
            env["REPRO_REPEAT_ID"] = str(repeat_idx)
            if task.name in INFILL_THROUGHPUT_TASKS:
                base_output_tag = env.get("OUTPUT_TAG", "infill_throughput")
                env["OUTPUT_TAG"] = f"{base_output_tag}_rep{repeat_idx}"
    git_sha = current_git_sha()
    return MatrixRow(
        row_id=row_id,
        task=task.name,
        model_variant=target.variant,
        block_size=target.block_size,
        desired_block_size=target.desired_block_size,
        eval_block_size=target.eval_block_size,
        max_window_size=target.max_window_size,
        confidence_threshold=env.get(
            "CONFIDENCE_THRESHOLD", target.confidence_threshold
        ),
        confidence_based_noising=env.get(
            "CONFIDENCE_BASED_NOISING", target.confidence_based_noising
        ),
        pareto_group=target.pareto_group,
        checkpoint_path=target.checkpoint_path,
        model_path=target.model_path,
        ckpt_file=target.ckpt_file,
        use_ema=target.use_ema,
        command=build_command(mode, task, env) if status == "ready" else "N/A",
        local_command=build_local_command(task, env) if status == "ready" else "N/A",
        script=task.script,
        output_path=output_path,
        log_path=log_path if status == "ready" else "N/A",
        group_log_path="",
        metadata_path=metadata_path,
        job_id="",
        output_root=str(suite_dir),
        git_sha=git_sha,
        node_name="N/A until execution",
        gpu_name="N/A until execution",
        num_gpus=num_gpus,
        gpu_constraint=gpu_constraint,
        gpu_partition=env.get("GPU_PARTITION", GPU_PARTITION_LIST) if status == "ready" else "N/A",
        throughput_measured_examples=(
            THROUGHPUT_MEASURED_EXAMPLES if task.name in THROUGHPUT_TASKS else "N/A"
        ),
        throughput_warmup_examples_per_rank=(
            THROUGHPUT_WARMUP_EXAMPLES_PER_RANK if task.name in THROUGHPUT_TASKS else "N/A"
        ),
        throughput_global_measurements=(
            "true" if task.name in THROUGHPUT_TASKS else "N/A"
        ),
        throughput_num_gpus=num_gpus if task.name in THROUGHPUT_TASKS else "N/A",
        compile_backbone=env.get("COMPILE_BACKBONE", "N/A"),
        compile_mode=env.get("COMPILE_MODE", "N/A"),
        compile_dynamic=env.get("COMPILE_DYNAMIC", "N/A"),
        compile_supported=env.get("COMPILE_SUPPORTED", "N/A"),
        cnndm_generate_target_prompt=env.get("CNNDM_GENERATE_TARGET_PROMPT", "N/A"),
        setdlm_throughput_run_name=env.get("SETDLM_THROUGHPUT_RUN_NAME", "N/A"),
        status=status,
        reason=reason,
    )


def build_matrix(args: argparse.Namespace, suite_dir: Path) -> list[MatrixRow]:
    task_filter = set(args.tasks or [])
    variant_filter = set(args.variants or [])
    row_id_filter = set(args.row_ids or [])
    rows: list[MatrixRow] = []
    for task in TASKS:
        if task_filter and task.name not in task_filter:
            continue
        repeat_count = (
            infill_throughput_repeat_count()
            if task.name in INFILL_THROUGHPUT_TASKS
            else 1
        )
        for target in GROUP_TARGETS[task.group]():
            if variant_filter and target.variant not in variant_filter:
                continue
            for repeat_idx in range(1, repeat_count + 1):
                row = build_row(
                    task,
                    target,
                    suite_dir,
                    args.mode,
                    repeat_idx=repeat_idx if repeat_count > 1 else None,
                )
                if row_id_filter and row.row_id not in row_id_filter:
                    continue
                rows.append(row)
    return rows


def write_matrix(rows: list[MatrixRow], suite_dir: Path) -> None:
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "logs").mkdir(parents=True, exist_ok=True)
    jsonl_path = suite_dir / "command_matrix.jsonl"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    tsv_path = suite_dir / "command_matrix.tsv"
    with open(tsv_path, "w", newline="") as f:
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(MatrixRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_tsv(rows: list[MatrixRow]) -> None:
    fieldnames = [
        "row_id",
        "task",
        "model_variant",
        "block_size",
        "desired_block_size",
        "eval_block_size",
        "max_window_size",
        "confidence_threshold",
        "confidence_based_noising",
        "pareto_group",
        "status",
        "reason",
        "checkpoint_path",
        "output_path",
        "log_path",
        "group_log_path",
        "metadata_path",
        "job_id",
        "git_sha",
        "node_name",
        "gpu_name",
        "num_gpus",
        "gpu_constraint",
        "gpu_partition",
        "throughput_measured_examples",
        "throughput_warmup_examples_per_rank",
        "throughput_global_measurements",
        "throughput_num_gpus",
        "compile_backbone",
        "compile_mode",
        "compile_dynamic",
        "compile_supported",
        "cnndm_generate_target_prompt",
        "setdlm_throughput_run_name",
        "command",
        "local_command",
    ]
    print("\t".join(fieldnames))
    for row in rows:
        data = asdict(row)
        print("\t".join(str(data[name]).replace("\t", " ") for name in fieldnames))


def submit_row(row: MatrixRow) -> MatrixRow:
    if row.status != "ready":
        return row
    env = os.environ.copy()
    # Extract env assignments from the command rather than rebuilding, so the
    # submitted command exactly matches the dry-run row.
    command = row.command
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        row.status = "submit_failed"
        row.reason = output
        return row
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    if match:
        row.job_id = match.group(1)
        row.log_path = row.log_path.replace("<jobid>", row.job_id)
        row.status = "submitted"
        row.reason = output
    else:
        row.status = "completed" if result.returncode == 0 else "failed"
        row.reason = output
    return row


def run_local_row(row: MatrixRow) -> MatrixRow:
    if row.status != "ready":
        return row
    log_path = Path(row.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            row.command,
            cwd=REPO_ROOT,
            shell=True,
            text=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    row.status = "completed" if result.returncode == 0 else "failed"
    row.reason = f"exit_code={result.returncode}"
    return row


def write_group_manifest(rows: list[MatrixRow], suite_dir: Path) -> Path:
    manifest = suite_dir / "throughput_group_manifest.jsonl"
    with open(manifest, "w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    return manifest


def submit_throughput_group(rows: list[MatrixRow], suite_dir: Path) -> list[MatrixRow]:
    if not rows:
        return rows
    manifest = write_group_manifest(rows, suite_dir)
    env = {
        "REPRO_THROUGHPUT_MATRIX": str(manifest),
        "REPRO_THROUGHPUT_GROUP_ID": suite_dir.name,
        "NUM_VISIBLE_DEVICES": THROUGHPUT_NUM_GPUS,
        "GPU_CONSTRAINT": A6000_CONSTRAINT,
        "GPU_PARTITION": THROUGHPUT_GPU_PARTITION,
        "GPU_NODELIST": THROUGHPUT_NODELIST,
        "THROUGHPUT_NODELIST": THROUGHPUT_NODELIST,
    }
    command = (
        f"cd {q(str(REPO_ROOT))} && {env_prefix(env)} "
        "bash bash_scripts/sbatch_wrapper.sh run_repro_throughput_group.sh"
    )
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        check=False,
    )
    output = result.stdout.strip()
    group_log_path = ""
    job_id = ""
    status = "submitted"
    reason = output
    if result.returncode != 0:
        status = "submit_failed"
    else:
        match = re.search(r"Submitted batch job\s+(\d+)", output)
        if match:
            job_id = match.group(1)
            group_log_path = str(
                REPO_ROOT / "watch_folder" / f"repro_throughput_group_{job_id}.log"
            )
            reason = f"{output}; grouped throughput manifest={manifest}"
        else:
            status = "completed"
    for row in rows:
        row.command = command
        row.job_id = job_id
        row.group_log_path = group_log_path
        row.status = status
        row.reason = reason
    return rows


def execute(rows: list[MatrixRow], args: argparse.Namespace, suite_dir: Path) -> list[MatrixRow]:
    ready_rows = [row for row in rows if row.status == "ready"]
    if args.max_jobs is not None:
        ready_rows = ready_rows[: args.max_jobs]
    grouped_rows: list[MatrixRow] = []
    if args.mode == "sbatch" and args.group_throughput:
        grouped_rows = [row for row in ready_rows if row.task in THROUGHPUT_TASKS]
        if grouped_rows:
            grouped_rows = submit_throughput_group(grouped_rows, suite_dir)
    grouped_ids = {row.row_id for row in grouped_rows}
    grouped_by_id = {row.row_id: row for row in grouped_rows}
    ready_ids = {row.row_id for row in ready_rows}
    executed: list[MatrixRow] = []
    stop_reason = ""
    if any(row.status == "submit_failed" for row in grouped_rows):
        stop_reason = "Stopped after throughput group submit failure."
    for row in rows:
        if row.row_id in grouped_ids:
            executed.append(grouped_by_id[row.row_id])
            continue
        if row.row_id not in ready_ids:
            executed.append(row)
            continue
        if stop_reason:
            row.status = "not_submitted_after_failure"
            row.reason = stop_reason
            executed.append(row)
            continue
        if args.mode == "sbatch":
            submitted = submit_row(row)
        else:
            submitted = run_local_row(row)
        if submitted.status in {"submit_failed", "failed"}:
            stop_reason = f"Stopped after {submitted.row_id} {submitted.status}: {submitted.reason}"
        executed.append(submitted)
    return executed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sbatch", "local"], default="sbatch")
    parser.add_argument("--suite-id", default=dt.datetime.now().strftime("repro_%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tasks", nargs="*", choices=[task.name for task in TASKS])
    parser.add_argument("--variants", nargs="*", choices=["AR", "MDLM", "BD3LM", "SetDLM", "SEDD"])
    parser.add_argument("--row-ids", nargs="*", help="Limit matrix to exact row_id values after task/variant filters.")
    parser.add_argument("--execute", action="store_true", help="Submit or run ready rows. Default is dry-run only.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limit launched ready rows when --execute is set.")
    parser.add_argument(
        "--group-throughput",
        dest="group_throughput",
        action="store_true",
        default=False,
        help=(
            "Deprecated/disabled. Throughput rows must submit one row per Slurm job "
            "for reproducible final or diagnostic reporting."
        ),
    )
    parser.add_argument(
        "--no-group-throughput",
        dest="group_throughput",
        action="store_false",
        help="Submit throughput rows one row per job.",
    )
    parser.add_argument("--no-print", action="store_true", help="Write matrix files without printing TSV to stdout.")
    args = parser.parse_args()
    if args.group_throughput:
        parser.error(
            "--group-throughput is disabled for reproducibility throughput runs; "
            "submit throughput rows one row per Slurm job."
        )
    return args


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.output_root) / args.suite_id
    rows = build_matrix(args, suite_dir)
    if args.execute:
        rows = execute(rows, args, suite_dir)
    write_matrix(rows, suite_dir)
    print(f"matrix_jsonl={suite_dir / 'command_matrix.jsonl'}")
    print(f"matrix_tsv={suite_dir / 'command_matrix.tsv'}")
    print(f"suite_dir={suite_dir}")
    if not args.no_print:
        print_tsv(rows)


if __name__ == "__main__":
    main()
