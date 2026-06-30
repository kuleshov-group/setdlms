import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from scripts.utils import load_model_from_ckpt_dir_path
from src.denoiser.ar import AR, ARConfig
from src.denoiser.bd3lm import BD3LM, BD3LMConfig
from src.denoiser.mdlm import MDLM, SEDD, MDLMConfig
from src.noise_schedule.noise_schedules import LinearNoise
from src.utils import fsspec_exists


def configure_rank_local_torchinductor_cache() -> str | None:
    """Avoid multi-rank torch.compile cache rename races."""
    enabled = os.environ.get("REPRO_RANK_LOCAL_TORCHINDUCTOR_CACHE", "true")
    if enabled.lower() in {"0", "false", "no"}:
        return None
    rank = (
        os.environ.get("LOCAL_RANK")
        or os.environ.get("RANK")
        or os.environ.get("SLURM_PROCID")
        or "0"
    )
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not base:
        base = str(Path.cwd() / ".torchinductor_cache" / os.environ.get("USER", "user"))
    cache_dir = Path(base) / "rank_local" / f"job{job_id}" / f"rank{rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    return str(cache_dir)


def _maybe_to_plain_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=False)
    return value if isinstance(value, dict) else None


def _extract_legacy_hyper_parameters(ckpt: dict[str, Any]) -> dict[str, Any]:
    hyper_parameters = _maybe_to_plain_dict(ckpt.get("hyper_parameters"))
    return hyper_parameters if hyper_parameters is not None else {}


def _extract_legacy_checkpoint_tokenizer(ckpt: dict[str, Any]) -> Any | None:
    return _extract_legacy_hyper_parameters(ckpt).get("tokenizer")


def maybe_load_legacy_checkpoint_tokenizer(
    pretrained_model_name_or_path: str,
) -> Any | None:
    if not (
        os.path.isfile(pretrained_model_name_or_path)
        and pretrained_model_name_or_path.endswith(".ckpt")
    ):
        return None
    try:
        ckpt = torch.load(
            pretrained_model_name_or_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception:
        return None
    tokenizer = _extract_legacy_checkpoint_tokenizer(ckpt)
    if tokenizer is None:
        return None
    if getattr(tokenizer, "padding_side", None) is None:
        tokenizer.padding_side = "right"
    return tokenizer


def _normalize_legacy_state_dict_keys(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    normalized_state_dict = dict(state_dict)
    for key in list(normalized_state_dict.keys()):
        new_key = key
        if "backbone." in new_key:
            new_key = new_key.replace("backbone.", "")
        if "_orig_mod." in new_key:
            new_key = new_key.replace("_orig_mod.", "")
        if new_key != key:
            normalized_state_dict[new_key] = normalized_state_dict.pop(key)
    normalized_state_dict.pop("sampling_eps_min", None)
    normalized_state_dict.pop("sampling_eps_max", None)
    return normalized_state_dict


def _maybe_load_legacy_ema_state_dict(
    *,
    ckpt: dict[str, Any],
    denoiser,
    load_ema_weights: bool,
) -> dict[str, Any]:
    state_dict = _normalize_legacy_state_dict_keys(ckpt["state_dict"])
    if not load_ema_weights:
        return state_dict

    ema_state = ckpt.get("ema")
    if not isinstance(ema_state, dict):
        raise ValueError(
            "EMA weights requested, but legacy checkpoint has no `ema` state."
        )
    shadow_params = ema_state.get("shadow_params")
    if not isinstance(shadow_params, list):
        raise ValueError(
            "EMA weights requested, but checkpoint `ema.shadow_params` "
            "is missing or malformed."
        )

    named_parameters = list(denoiser.backbone.named_parameters())
    if len(named_parameters) != len(shadow_params):
        raise ValueError(
            "EMA weights requested, but the checkpoint parameter count "
            f"({len(shadow_params)}) does not match the instantiated backbone "
            f"parameter count ({len(named_parameters)})."
        )

    ema_backbone_state_dict = dict(state_dict)
    for (param_name, _), shadow_param in zip(
        named_parameters, shadow_params, strict=True
    ):
        ema_backbone_state_dict[param_name] = shadow_param
    return ema_backbone_state_dict


def normalize_model_config_overrides(
    model_config_overrides: dict[str, Any] | DictConfig | str | None,
) -> dict[str, Any]:
    if model_config_overrides is None:
        return {}
    if isinstance(model_config_overrides, str):
        model_config_overrides = json.loads(model_config_overrides)
    if isinstance(model_config_overrides, DictConfig):
        model_config_overrides = OmegaConf.to_container(  # type: ignore[assignment]
            model_config_overrides,
            resolve=True,
        )
    return dict(model_config_overrides)


def _load_hf_model(
    pretrained_model_name_or_path: str,
    pretrained_model_revision: str | None = None,
    allow_masked_lm: bool = True,
):
    pretrained_kwargs = {
        "trust_remote_code": True,
        "revision": pretrained_model_revision,
    }
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is not None:
        pretrained_kwargs["token"] = hf_token
    try:
        return (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                **pretrained_kwargs,
            ),
            "causal_lm",
        )
    except Exception:
        if not allow_masked_lm:
            return None, "causal_lm_unavailable"
        try:
            return (
                AutoModelForMaskedLM.from_pretrained(
                    pretrained_model_name_or_path,
                    **pretrained_kwargs,
                ),
                "masked_lm",
            )
        except Exception:
            return None, None


def _normalize_hf_model_load_result(
    load_result: Any,
) -> tuple[Any | None, str | None]:
    if isinstance(load_result, tuple) and len(load_result) == 2:
        return load_result
    return load_result, None


def _is_sedd_model_path(pretrained_model_name_or_path: str) -> bool:
    return "sedd" in str(pretrained_model_name_or_path).lower()


def _legacy_backbone_config_name(pretrained_model_name_or_path: str) -> str:
    return "dit_legacy.yaml"


def _load_legacy_denoiser(
    pretrained_model_name_or_path: str,
    tokenizer,
    device: torch.device | str,
    model: Any | None = None,
    model_config_overrides: dict[str, Any] | None = None,
    load_ema_weights: bool = False,
):
    model_config_overrides = (
        {} if model_config_overrides is None else model_config_overrides
    )
    backbone_overrides = model_config_overrides.get("backbone_config", {})
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    backbone_config_path = os.path.join(
        repo_root,
        "configs",
        "model",
        "backbone",
        _legacy_backbone_config_name(pretrained_model_name_or_path),
    )
    backbone_config = OmegaConf.load(backbone_config_path)

    length = model_config_overrides.get("length") or 1024
    block_size = model_config_overrides.get("block_size")
    backbone_config.length = length
    backbone_config.vocab_size = len(tokenizer)
    backbone_config.block_size = block_size
    backbone_config.pretrained_model_name_or_path = (
        pretrained_model_name_or_path if model is not None else None
    )
    backbone_config.num_layers = 12
    backbone_config.n_heads = 12
    backbone_config.hidden_size = 768
    if "-ar-" in pretrained_model_name_or_path:
        backbone_config.adaln = False
        backbone_config.causal_attention = True
        backbone_config.attn_backend = "flash_attn"
    else:
        backbone_config.adaln = True

    for key, value in backbone_overrides.items():
        setattr(backbone_config, key, value)

    if not isinstance(backbone_config, DictConfig):
        backbone_config = OmegaConf.create(
            OmegaConf.to_container(backbone_config, resolve=False)
        )

    common_config_kwargs = {
        "length": length,
        "backbone_config": OmegaConf.to_container(backbone_config, resolve=True),
    }
    if _is_sedd_model_path(pretrained_model_name_or_path):
        denoiser_config = MDLMConfig(**common_config_kwargs)
        denoiser_cls = SEDD
    elif "mdlm-" in pretrained_model_name_or_path:
        denoiser_config = MDLMConfig(**common_config_kwargs)
        denoiser_cls = MDLM
    elif "ar-" in pretrained_model_name_or_path:
        denoiser_config = ARConfig(**common_config_kwargs)
        denoiser_cls = AR
    else:
        denoiser_config = BD3LMConfig(
            **common_config_kwargs,
            block_size=block_size,
        )
        denoiser_cls = BD3LM

    denoiser_config.keep_clean_bos = True
    denoiser_config.mask_token_id = tokenizer.mask_token_id
    denoiser_config.vocab_size = len(tokenizer)
    denoiser = denoiser_cls(denoiser_config, tokenizer=tokenizer)

    for key, value in model_config_overrides.items():
        if key == "backbone_config":
            continue
        setattr(denoiser.config, key, value)
    if backbone_overrides:
        backbone_runtime_config = getattr(denoiser.backbone, "config", None)
        target = backbone_runtime_config if backbone_runtime_config is not None else denoiser.backbone
        for key, value in backbone_overrides.items():
            setattr(target, key, value)

    if model is not None:
        denoiser.backbone = model.backbone if hasattr(model, "backbone") else model
    else:
        ckpt = torch.load(
            pretrained_model_name_or_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = _maybe_load_legacy_ema_state_dict(
            ckpt=ckpt,
            denoiser=denoiser,
            load_ema_weights=load_ema_weights,
        )
        denoiser.backbone.load_state_dict(state_dict)

    denoiser = denoiser.to(device)
    denoiser.noise_schedule = LinearNoise()
    return denoiser


def load_eval_model(
    pretrained_model_name_or_path: str,
    tokenizer,
    device: torch.device | str,
    pretrained_model_revision: str | None = None,
    load_ema_weights: bool = False,
    ckpt_file: str = "best-rank0.pt",
    model_config_overrides: dict[str, Any] | DictConfig | str | None = None,
    verbose: bool = False,
    force_legacy_if_no_generate: bool = False,
):
    model_config_overrides = normalize_model_config_overrides(model_config_overrides)
    ckpt_config_path = os.path.join(pretrained_model_name_or_path, "config.yaml")
    has_ckpt_config = fsspec_exists(ckpt_config_path)

    if has_ckpt_config:
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=pretrained_model_name_or_path,
            load_ema_weights=load_ema_weights,
            ckpt_file=ckpt_file,
            verbose=verbose,
            device=device,
            **model_config_overrides,
        )
    else:
        model, _ = _normalize_hf_model_load_result(
            _load_hf_model(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                pretrained_model_revision=pretrained_model_revision,
            )
        )

    is_sedd_path = _is_sedd_model_path(pretrained_model_name_or_path)
    if is_sedd_path and isinstance(model, SEDD):
        return model.to(device)

    if model is None or (
        is_sedd_path
        and not isinstance(model, SEDD)
    ) or (
        force_legacy_if_no_generate and not hasattr(model, "generate")
    ):
        model = _load_legacy_denoiser(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            device=device,
            model=model,
            model_config_overrides=model_config_overrides,
            load_ema_weights=load_ema_weights,
        )
    return model.to(device)
