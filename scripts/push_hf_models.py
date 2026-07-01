#!/usr/bin/env python3
"""Push evaluation checkpoints to Hugging Face Hub.

This is the Python-script version of notebooks/push_to_hub.ipynb, expanded to cover
all local checkpoint-backed models in the reproducibility evaluation target matrix.
By default it lists what would be pushed. Pass --yes to perform uploads.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASE_RUN_DIR = Path("/share/kuleshov/ma2238/runs/dllm-dev")
TEXTDIFFUSION_CKPT_DIR = Path("/share/kuleshov/ma2238/textdiffusion/checkpoints")
DEFAULT_REPO_PREFIX = "kuleshov-group"


@dataclass(frozen=True)
class ModelSpec:
    task_group: str
    label: str
    checkpoint_dir: Path
    ckpt_file: str
    use_ema: bool
    repo_suffix: str

    @property
    def checkpoint_path(self) -> Path:
        in_checkpoints = self.checkpoint_dir / "checkpoints" / self.ckpt_file
        if in_checkpoints.exists():
            return in_checkpoints
        return self.checkpoint_dir / self.ckpt_file

    @property
    def config_path(self) -> Path:
        return self.checkpoint_dir / "config.yaml"

    @property
    def is_legacy_checkpoint(self) -> bool:
        return self.checkpoint_path.suffix == ".ckpt" and not self.config_path.exists()


def run_dir(name: str) -> Path:
    return BASE_RUN_DIR / name


def td_dir(name: str) -> Path:
    return TEXTDIFFUSION_CKPT_DIR / name


EVAL_MODELS: tuple[ModelSpec, ...] = (
    # OWT likelihood / infilling / generation targets.
    ModelSpec("owt", "AR", td_dir("mari-owt-ar-noeos-v4-1"), "20-300000.ckpt", True, "owt-ar"),
    ModelSpec("owt", "MDLM", td_dir("mari-owt-mdlm-noeos-v4"), "18-300000.ckpt", True, "owt-mdlm"),
    ModelSpec("owt", "BD3LM-s4", run_dir("owt_block4_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"), "ep17-ba300000-rank0.pt", False, "owt-bd3lm-s4"),
    ModelSpec("owt", "BD3LM-s8", run_dir("owt_block8_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"), "ep17-ba300000-rank0.pt", False, "owt-bd3lm-s8"),
    ModelSpec("owt", "BD3LM-s16", run_dir("owt_block16_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_normlayernorm_adalnfalse_vscratch2"), "ep17-ba300000-rank0.pt", False, "owt-bd3lm-s16"),
    ModelSpec("owt", "SetDLM-d4", run_dir("owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block4_vscratch"), "ep17-ba300000-rank0.pt", False, "owt-setdlm-d4"),
    ModelSpec("owt", "SetDLM-d8", run_dir("owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block8_vscratch"), "ep17-ba300000-rank0.pt", False, "owt-setdlm-d8"),
    ModelSpec("owt", "SetDLM-d16", run_dir("owt_block1024_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_normlayernorm_adalnfalse_block16_vscratch"), "ep17-ba300000-rank0.pt", False, "owt-setdlm-d16"),
    # LM1B likelihood / throughput targets.
    ModelSpec("lm1b", "AR", run_dir("lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter_ar_hparam_v1"), "best-rank0.pt", True, "lm1b-ar"),
    ModelSpec("lm1b", "MDLM", run_dir("lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_mdlm_adaln_cleanbos_antithetic_hparam_v2"), "ep72-ba1000000-rank0.pt", False, "lm1b-mdlm"),
    ModelSpec("lm1b", "BD3LM-s4", run_dir("lm1b_block4_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_dropout0.1_normlayernorm_hparam_v3"), "ep72-ba1000000-rank0.pt", False, "lm1b-bd3lm-s4"),
    ModelSpec("lm1b", "BD3LM-s8", run_dir("lm1b_block8_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_dropout0.1_normlayernorm_hparam_v3"), "best-rank0.pt", True, "lm1b-bd3lm-s8"),
    ModelSpec("lm1b", "BD3LM-s16", run_dir("lm1b_block16_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_bd3lm_dropout0.1_normlayernorm_hparam_v3"), "ep72-ba1000000-rank0.pt", False, "lm1b-bd3lm-s16"),
    ModelSpec("lm1b", "SetDLM-d4", run_dir("lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired4_max128_v5"), "ep72-ba1000000-rank0.pt", False, "lm1b-setdlm-d4"),
    ModelSpec("lm1b", "SetDLM-d8", run_dir("lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired8_max128_v5"), "ep72-ba1000000-rank0.pt", False, "lm1b-setdlm-d8"),
    ModelSpec("lm1b", "SetDLM-d16", run_dir("lm1b_block128_lr3e-4_bsz512_warm2500ba_layers12_hidden768_inter3072_aoarm_dropout0.1_normlayernorm_hparam_desired16_vlambda"), "best-rank0.pt", True, "lm1b-setdlm-d16"),
    # CNN/DM summarization targets.
    ModelSpec("cnndm", "AR", run_dir("cnn_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_ar_len768_v1"), "best-rank0.pt", True, "cnndm-ar"),
    ModelSpec("cnndm", "MDLM", run_dir("cnn_block_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_mdlm_len768_v1"), "best-rank0.pt", True, "cnndm-mdlm"),
    ModelSpec("cnndm", "BD3LM-s4", run_dir("cnn_block4_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_bd3lm_len768_v1"), "best-rank0.pt", True, "cnndm-bd3lm-s4"),
    ModelSpec("cnndm", "BD3LM-s8", run_dir("cnn_block8_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_bd3lm_len768_v1"), "best-rank0.pt", True, "cnndm-bd3lm-s8"),
    ModelSpec("cnndm", "BD3LM-s16", run_dir("cnn_block16_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_bd3lm_len768_v1"), "best-rank0.pt", True, "cnndm-bd3lm-s16"),
    ModelSpec("cnndm", "SetDLM-d4", run_dir("cnn_block768_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_aoarm_tgt4_len768_v1"), "best-rank0.pt", True, "cnndm-setdlm-d4"),
    ModelSpec("cnndm", "SetDLM-d8", run_dir("cnn_block768_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_aoarm_tgt8_len768_v1"), "best-rank0.pt", True, "cnndm-setdlm-d8"),
    ModelSpec("cnndm", "SetDLM-d16", run_dir("cnn_block768_lr3e-4_bsz128_warm1000ba_layers28_hidden256_inter768_aoarm_tgt16_len768_v1"), "best-rank0.pt", True, "cnndm-setdlm-d16"),
    # GSM8K accuracy / likelihood / throughput targets.
    ModelSpec("gsm8k", "AR", run_dir("gsm8k-0shot_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_ar_distill_v6"), "best-rank0.pt", True, "gsm8k-ar"),
    ModelSpec("gsm8k", "MDLM", run_dir("gsm8k-0shot_block_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_mdlm_distill_v5"), "best-rank0.pt", True, "gsm8k-mdlm"),
    ModelSpec("gsm8k", "BD3LM-s4", run_dir("gsm8k-shot_block4_lr1e-5_bsz1_warm100ba_max-dur75000ba_amp_bf16_layers28_bd3lm_distill_anneal0ba_maxbs4_v10"), "best-rank0.pt", True, "gsm8k-bd3lm-s4"),
    ModelSpec("gsm8k", "BD3LM-s8", run_dir("gsm8k-shot_block8_lr1e-5_bsz1_warm100ba_max-dur75000ba_amp_bf16_layers28_bd3lm_distill_anneal0ba_maxbs8_v10"), "best-rank0.pt", True, "gsm8k-bd3lm-s8"),
    ModelSpec("gsm8k", "BD3LM-s16", run_dir("gsm8k-shot_block16_lr1e-5_bsz1_warm100ba_max-dur75000ba_amp_bf16_layers28_bd3lm_distill_anneal0ba_maxbs16_v10"), "best-rank0.pt", True, "gsm8k-bd3lm-s16"),
    ModelSpec("gsm8k", "SetDLM-d4", run_dir("gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt4_max1024_distill_again_v2"), "best-rank0.pt", True, "setdlm-gsm8k-smax8"),
    ModelSpec("gsm8k", "SetDLM-d8", run_dir("gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt8_max1024_distill_v23"), "best-rank0.pt", True, "setdlm-gsm8k-smax16"),
    ModelSpec("gsm8k", "SetDLM-d16", run_dir("gsm8k-0shot_block1024_lr1e-5_bsz1_warm100ba_alphaf0.5_max-dur75000ba_amp_bf16_layers28_aoarm_tgt16_max1024_distill_again_v2"), "best-rank0.pt", True, "setdlm-gsm8k-smax32"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Actually push models. Without this, only list planned pushes.")
    parser.add_argument("--only", action="append", default=[], help="Only include specs containing this substring. Can be repeated.")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude specs containing this substring. Can be repeated.")
    parser.add_argument("--repo-prefix", default=DEFAULT_REPO_PREFIX, help="HF namespace/org for generated repo IDs.")
    parser.add_argument("--public", action="store_true", help="Create/update public repos instead of private repos.")
    parser.add_argument("--device", default="cpu", help="Device used to load checkpoints before pushing.")
    parser.add_argument("--commit-message", default="Upload evaluation checkpoint", help="Commit message for model/code upload.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip specs whose checkpoint or config.yaml is missing.")
    parser.add_argument("--local-dir", type=Path, default=None, help="Save converted HF repos under this directory instead of pushing to Hub.")
    parser.add_argument("--resolve", default=None, help="Resolve a model key/path for eval scripts and print shell assignments.")
    parser.add_argument("--prefer-local", action="store_true", help="Prefer the local checkpoint when resolving instead of HF.")
    parser.add_argument("--no-hf-check", action="store_true", help="Treat known HF repo IDs as available without querying the Hub.")
    return parser.parse_args()


def matches_filters(spec: ModelSpec, only: list[str], exclude: list[str]) -> bool:
    haystack = " ".join(
        [
            spec.task_group,
            spec.label,
            spec.repo_suffix,
            str(spec.checkpoint_dir),
            spec.ckpt_file,
        ]
    ).lower()
    if only and not any(pattern.lower() in haystack for pattern in only):
        return False
    if any(pattern.lower() in haystack for pattern in exclude):
        return False
    return True


def repo_id_for(spec: ModelSpec, repo_prefix: str) -> str:
    return f"{repo_prefix}/{spec.repo_suffix}"


def status_for(spec: ModelSpec) -> str:
    missing = []
    if not spec.checkpoint_path.exists():
        missing.append("checkpoint")
    if not spec.config_path.exists() and not spec.is_legacy_checkpoint:
        missing.append("config.yaml")
    return "ok" if not missing else "missing " + ", ".join(missing)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def local_model_path_for(spec: ModelSpec) -> Path:
    if spec.is_legacy_checkpoint:
        return spec.checkpoint_path
    return spec.checkpoint_dir


def spec_aliases(spec: ModelSpec, repo_prefix: str) -> set[str]:
    repo_id = repo_id_for(spec, repo_prefix)
    label_slug = slug(spec.label)
    aliases = {
        repo_id,
        spec.repo_suffix,
        f"{spec.task_group}:{label_slug}",
        f"{spec.task_group}-{label_slug}",
        str(spec.checkpoint_dir),
        str(spec.checkpoint_path),
        str(local_model_path_for(spec)),
    }
    return {alias.lower() for alias in aliases if alias}


def find_spec(requested: str, repo_prefix: str) -> ModelSpec | None:
    requested_key = requested.strip().lower()
    for spec in EVAL_MODELS:
        if requested_key in spec_aliases(spec, repo_prefix):
            return spec
    return None


def looks_like_hf_model_id(value: str) -> bool:
    return bool(value) and not value.startswith("/") and "/" in value


def hf_model_available(repo_id: str) -> bool:
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return False
    try:
        from huggingface_hub import HfApi
    except Exception:
        return False
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        HfApi(token=token).model_info(repo_id)
    except Exception:
        return False
    return True


def local_model_available_for_spec(spec: ModelSpec) -> bool:
    if spec.is_legacy_checkpoint:
        return spec.checkpoint_path.exists()
    return spec.config_path.exists() and spec.checkpoint_path.exists()


def local_model_available(path: Path, ckpt_file: str | None = None) -> bool:
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    if not ckpt_file:
        return True
    return (path / "checkpoints" / ckpt_file).exists() or (path / ckpt_file).exists()


def resolve_requested_model(
    requested: str,
    repo_prefix: str,
    prefer_hf: bool,
    check_hf: bool,
) -> dict[str, str]:
    if not requested:
        raise ValueError("Set MODEL_PATH or EVAL_MODEL_KEY to a HF repo id, local path, or known model key.")

    spec = find_spec(requested, repo_prefix)
    if spec is None:
        if looks_like_hf_model_id(requested):
            if not check_hf or hf_model_available(requested):
                return {
                    "RESOLVED_MODEL_PATH": requested,
                    "RESOLVED_MODEL_SOURCE": "hf",
                    "RESOLVED_MODEL_REPO_ID": requested,
                    "RESOLVED_LOCAL_MODEL_PATH": "",
                    "RESOLVED_CKPT_FILE": os.environ.get("CKPT_FILE", ""),
                    "RESOLVED_USE_EMA": os.environ.get("USE_EMA", ""),
                    "RESOLVED_TASK_GROUP": "",
                    "RESOLVED_MODEL_LABEL": "",
                }
            raise FileNotFoundError(f"HF model is unavailable and no local fallback is known: {requested}")

        local_path = Path(requested)
        if local_model_available(local_path, os.environ.get("CKPT_FILE")):
            return {
                "RESOLVED_MODEL_PATH": str(local_path),
                "RESOLVED_MODEL_SOURCE": "local",
                "RESOLVED_MODEL_REPO_ID": "",
                "RESOLVED_LOCAL_MODEL_PATH": str(local_path),
                "RESOLVED_CKPT_FILE": os.environ.get("CKPT_FILE", ""),
                "RESOLVED_USE_EMA": os.environ.get("USE_EMA", ""),
                "RESOLVED_TASK_GROUP": "",
                "RESOLVED_MODEL_LABEL": "",
            }
        raise FileNotFoundError(f"No available HF model or local model path found for: {requested}")

    repo_id = repo_id_for(spec, repo_prefix)
    local_path = str(local_model_path_for(spec))
    hf_available = (not check_hf) or hf_model_available(repo_id)
    local_available = local_model_available_for_spec(spec)

    if prefer_hf and hf_available:
        resolved_path = repo_id
        source = "hf"
    elif local_available:
        resolved_path = local_path
        source = "local"
    elif hf_available:
        resolved_path = repo_id
        source = "hf"
    else:
        raise FileNotFoundError(
            "No available HF model or local checkpoint found for "
            f"{requested} (HF: {repo_id}, local: {local_path})."
        )

    return {
        "RESOLVED_MODEL_PATH": resolved_path,
        "RESOLVED_MODEL_SOURCE": source,
        "RESOLVED_MODEL_REPO_ID": repo_id,
        "RESOLVED_LOCAL_MODEL_PATH": local_path,
        "RESOLVED_CKPT_FILE": spec.ckpt_file,
        "RESOLVED_USE_EMA": str(spec.use_ema).lower(),
        "RESOLVED_TASK_GROUP": spec.task_group,
        "RESOLVED_MODEL_LABEL": spec.label,
    }


def print_shell_assignments(values: dict[str, str]) -> None:
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


def load_tokenizer(config_path: Path):
    import hydra
    import yaml
    from omegaconf import OmegaConf

    with config_path.open("rb") as f:
        config = OmegaConf.create(yaml.safe_load(f))
    return hydra.utils.instantiate(config.tokenizer)


def push_one(spec: ModelSpec, args: argparse.Namespace) -> None:
    import torch

    from scripts.utils import load_model_from_ckpt_dir_path
    from src.utils import save_pretrained_or_push_to_hub

    destination = repo_id_for(spec, args.repo_prefix)
    local = args.local_dir is not None
    if local:
        destination = str(args.local_dir / spec.repo_suffix)

    if spec.config_path.exists():
        tokenizer = load_tokenizer(spec.config_path)
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=str(spec.checkpoint_dir),
            ckpt_file=spec.ckpt_file,
            load_ema_weights=spec.use_ema,
            device=torch.device(args.device),
        )
    elif spec.is_legacy_checkpoint:
        from scripts.eval.model_loading import (
            load_eval_model,
            maybe_load_legacy_checkpoint_tokenizer,
        )

        tokenizer = maybe_load_legacy_checkpoint_tokenizer(str(spec.checkpoint_path))
        if tokenizer is None:
            raise ValueError(f"Could not load tokenizer from {spec.checkpoint_path}")
        model = load_eval_model(
            str(spec.checkpoint_path),
            tokenizer=tokenizer,
            device=torch.device(args.device),
            load_ema_weights=spec.use_ema,
        )
    else:
        raise FileNotFoundError(f"Missing config.yaml for {spec.checkpoint_dir}")
    save_pretrained_or_push_to_hub(
        model=model,
        tokenizer=tokenizer,
        repo_id=destination,
        private=not args.public,
        local=local,
        project_root=str(REPO_ROOT),
        commit_message=args.commit_message,
    )


def main() -> None:
    args = parse_args()
    if args.resolve is not None:
        try:
            resolved = resolve_requested_model(
                requested=args.resolve,
                repo_prefix=args.repo_prefix,
                prefer_hf=not args.prefer_local,
                check_hf=not args.no_hf_check,
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print_shell_assignments(resolved)
        return

    specs = [
        spec
        for spec in EVAL_MODELS
        if matches_filters(spec, only=args.only, exclude=args.exclude)
    ]
    if not specs:
        raise SystemExit("No model specs matched the requested filters.")

    print(f"Found {len(specs)} model spec(s).")
    for idx, spec in enumerate(specs, start=1):
        repo_id = repo_id_for(spec, args.repo_prefix)
        status = status_for(spec)
        print(
            f"{idx:02d}. {repo_id} | {spec.task_group} {spec.label} | "
            f"ckpt={spec.checkpoint_path} | ema={str(spec.use_ema).lower()} | {status}"
        )

    if not args.yes:
        print("Dry run only. Re-run with --yes to push, or --local-dir PATH --yes to test local saving.")
        return

    for spec in specs:
        status = status_for(spec)
        if status != "ok":
            message = f"Cannot push {repo_id_for(spec, args.repo_prefix)}: {status}."
            if args.skip_missing:
                print(f"Skipping: {message}")
                continue
            raise FileNotFoundError(message)
        print(f"Pushing {repo_id_for(spec, args.repo_prefix)}")
        push_one(spec, args)


if __name__ == "__main__":
    main()
