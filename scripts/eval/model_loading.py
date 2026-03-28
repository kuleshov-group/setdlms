import inspect
import json
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from scripts.utils import load_model_from_ckpt_dir_path
from src.denoiser.ar import AR, ARConfig
from src.denoiser.bd3lm import BD3LM, BD3LMConfig
from src.denoiser.esolm import EsoLM, EsoLMConfig
from src.denoiser.mdlm import MDLM, MDLMConfig, SEDD
from src.denoiser.refusion import ReFusion, ReFusionConfig
from src.noise_schedule.noise_schedules import LinearNoise
from src.utils import fsspec_exists

REFUSION_SPECIAL_TOKEN_ATTRS = (
    "mask_token_id",
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
)


def _resize_model_embeddings_to_match_tokenizer(
    model: Any | None,
    tokenizer,
) -> None:
    if model is None:
        return
    resize_target = model
    resize_fn = getattr(resize_target, "resize_token_embeddings", None)
    if not callable(resize_fn) and hasattr(model, "backbone"):
        resize_target = getattr(model.backbone, "model", model.backbone)
        resize_fn = getattr(resize_target, "resize_token_embeddings", None)
    config = getattr(resize_target, "config", None)
    if not callable(resize_fn) or config is None:
        return
    target_vocab_size = len(tokenizer)
    current_vocab_size = getattr(config, "vocab_size", None)
    if isinstance(current_vocab_size, int) and current_vocab_size != target_vocab_size:
        resize_fn(target_vocab_size)
    config.vocab_size = target_vocab_size
    if hasattr(resize_target, "vocab_size"):
        resize_target.vocab_size = target_vocab_size


def _normalize_requested_model_type(model_type: Any) -> str | None:
    if not isinstance(model_type, str):
        return None
    normalized = model_type.strip().lower().rsplit(".", maxsplit=1)[-1]
    if normalized in {"refusion", "refusionconfig"}:
        return ReFusionConfig.model_type
    return None


def _get_explicit_requested_model_type(
    model_config_overrides: dict[str, Any],
) -> str | None:
    candidates = [
        model_config_overrides.get("model_type"),
        model_config_overrides.get("_target_"),
        model_config_overrides.get("denoiser_type"),
    ]
    nested_config = model_config_overrides.get("config")
    if isinstance(nested_config, dict):
        candidates.extend(
            [
                nested_config.get("model_type"),
                nested_config.get("_target_"),
            ]
        )
    for candidate in candidates:
        normalized = _normalize_requested_model_type(candidate)
        if normalized is not None:
            return normalized
    return None


def _is_refusion_model(model: Any | None) -> bool:
    if model is None:
        return False
    if isinstance(model, ReFusion):
        return True
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return isinstance(model_type, str) and model_type.lower() == ReFusionConfig.model_type


def _has_refusion_forward_contract(model: Any | None) -> bool:
    """Heuristic for official upstream ReFusion HF models before local wrapping.

    The upstream implementation keeps ReFusion semantics inside a Qwen wrapper and
    exposes `prompt_lengths` in `forward`, which plain causal LMs do not. We use
    that contract to auto-wrap official-style checkpoints in the local ReFusion
    adapter instead of silently falling back to autoregressive generation.
    """

    if model is None or _is_refusion_model(model):
        return False
    forward = getattr(model, "forward", None)
    if forward is None or not callable(forward):
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    return "prompt_lengths" in parameters


def _normalize_sequence_length_candidate(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, int):
        return None
    if value <= 0 or value >= 1_000_000:
        return None
    return int(value)


def _infer_refusion_rebuild_length(
    *,
    model: Any | None,
    tokenizer,
    model_config_overrides: dict[str, Any],
) -> int | None:
    explicit_length = _normalize_sequence_length_candidate(
        model_config_overrides.get("length")
    )
    if explicit_length is not None:
        return explicit_length

    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "length", None),
        getattr(config, "max_position_embeddings", None),
        getattr(config, "max_seq_len", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    for candidate in candidates:
        normalized = _normalize_sequence_length_candidate(candidate)
        if normalized is not None:
            return normalized
    return None


def _get_required_refusion_special_token_ids(tokenizer) -> dict[str, int]:
    ReFusion.prepare_tokenizer_for_refusion(tokenizer)
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError(
            "ReFusion requires `tokenizer.mask_token_id`; provide a tokenizer with "
            "an explicit mask token instead of relying on generic fallback behavior."
        )
    special_token_ids = {"mask_token_id": int(mask_token_id)}
    for token_attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, token_attr, None)
        if token_id is not None:
            special_token_ids[token_attr] = int(token_id)
    return special_token_ids


def _validate_refusion_special_token_ids(
    model: Any,
    tokenizer,
    *,
    model_label: str = "model",
) -> dict[str, int]:
    special_token_ids = _get_required_refusion_special_token_ids(tokenizer)
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("ReFusion loading requires the model to expose `config`.")

    model_vocab_size = getattr(config, "vocab_size", None)
    if model_vocab_size is not None:
        model_vocab_size = int(model_vocab_size)
        for token_attr, token_id in special_token_ids.items():
            if token_id >= model_vocab_size:
                raise ValueError(
                    "ReFusion loading requires tokenizer special tokens to fall within "
                    f"`{model_label}.config.vocab_size`; got "
                    f"`tokenizer.{token_attr}={token_id}` and "
                    f"`{model_label}.config.vocab_size={model_vocab_size}`."
                )

    for token_attr in REFUSION_SPECIAL_TOKEN_ATTRS:
        tokenizer_token_id = getattr(tokenizer, token_attr, None)
        config_token_id = getattr(config, token_attr, None)
        model_token_id = getattr(model, token_attr, None)
        if tokenizer_token_id is not None:
            expected_token_id = int(tokenizer_token_id)
            for source_name, source_token_id in (
                ("config", config_token_id),
                (model_label, model_token_id),
            ):
                if (
                    source_token_id is not None
                    and int(source_token_id) != expected_token_id
                ):
                    raise ValueError(
                        "ReFusion loading requires "
                        f"`{source_name}.{token_attr}` to match "
                        f"`tokenizer.{token_attr}`."
                    )
        elif (
            config_token_id is not None
            and model_token_id is not None
            and int(config_token_id) != int(model_token_id)
        ):
            raise ValueError(
                "ReFusion loading requires "
                f"`config.{token_attr}` and `{model_label}.{token_attr}` to agree "
                "when the tokenizer does not define that special token."
            )

    return special_token_ids


def _configure_refusion_special_token_ids(model: Any, tokenizer) -> None:
    ReFusion.prepare_tokenizer_for_refusion(tokenizer)
    if isinstance(model, ReFusion):
        model.sync_tokenizer_and_backbone(tokenizer=tokenizer)
    else:
        _resize_model_embeddings_to_match_tokenizer(model, tokenizer)
    special_token_ids = _validate_refusion_special_token_ids(model, tokenizer)
    for token_attr, token_id in special_token_ids.items():
        setattr(model.config, token_attr, token_id)
        setattr(model, token_attr, token_id)


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


def _rebuild_as_refusion_or_raise(
    *,
    pretrained_model_name_or_path: str,
    tokenizer,
    device: torch.device | str,
    model: Any | None,
    model_config_overrides: dict[str, Any],
    requested_model_type: str | None,
    error_prefix: str,
) -> ReFusion:
    if model is not None:
        ReFusion.prepare_tokenizer_for_refusion(tokenizer)
        _resize_model_embeddings_to_match_tokenizer(model, tokenizer)
        _validate_refusion_special_token_ids(model, tokenizer, model_label="backbone")
    try:
        wrapped_model = _load_legacy_denoiser(
            pretrained_model_name_or_path,
            tokenizer=tokenizer,
            device=device,
            model=model,
            model_config_overrides=model_config_overrides,
            requested_model_type=requested_model_type or ReFusionConfig.model_type,
        )
    except Exception as exc:
        raise ValueError(f"{error_prefix}: {exc}") from exc
    if not _is_refusion_model(wrapped_model):
        raise ValueError(
            f"{error_prefix} because loading produced a non-ReFusion model."
        )
    _configure_refusion_special_token_ids(wrapped_model, tokenizer)
    return wrapped_model.to(device)


def _load_legacy_denoiser(
    pretrained_model_name_or_path: str,
    tokenizer,
    device: torch.device | str,
    model: Any | None = None,
    model_config_overrides: dict[str, Any] | None = None,
    requested_model_type: str | None = None,
):
    model_config_overrides = (
        {} if model_config_overrides is None else model_config_overrides
    )
    backbone_overrides = model_config_overrides.get("backbone_config", {})
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    is_refusion = requested_model_type == ReFusionConfig.model_type
    backbone_config_path = os.path.join(
        repo_root,
        "configs",
        "model",
        "backbone",
        (
            "automodel_for_causal_lm.yaml"
            if is_refusion
            else "dit_legacy.yaml"
        ),
    )
    backbone_config = OmegaConf.load(backbone_config_path)

    length = model_config_overrides.get("length", 1024)
    block_size = model_config_overrides.get("block_size")
    backbone_config.length = length
    backbone_config.vocab_size = len(tokenizer)
    if is_refusion:
        special_token_ids = _get_required_refusion_special_token_ids(tokenizer)
        if model is not None:
            _resize_model_embeddings_to_match_tokenizer(model, tokenizer)
            _validate_refusion_special_token_ids(
                model,
                tokenizer,
                model_label="backbone",
            )
        override_backbone = model_config_overrides.get("backbone_config", {})
        pretrained_backbone_path = override_backbone.get(
            "pretrained_model_name_or_path"
        )
        if pretrained_backbone_path is None:
            pretrained_backbone_path = model_config_overrides.get(
                "backbone_pretrained_model_name_or_path"
            )
        if pretrained_backbone_path is None and model is None:
            raise ValueError(
                "Legacy ReFusion loading requires "
                "`model_config_overrides.backbone_config.pretrained_model_name_or_path` "
                "or a checkpoint directory with `config.yaml`."
            )
        if pretrained_backbone_path is not None:
            backbone_config.pretrained_model_name_or_path = pretrained_backbone_path
        backbone_config.use_causal_mask = True
    else:
        backbone_config.block_size = block_size
        backbone_config.pretrained_model_name_or_path = pretrained_model_name_or_path
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

    if is_refusion:
        denoiser_config = ReFusionConfig(length=length)
        denoiser_config.backbone_config = OmegaConf.to_container(
            backbone_config, resolve=True
        )
        for token_attr, token_id in special_token_ids.items():
            setattr(denoiser_config, token_attr, token_id)
        denoiser_config.vocab_size = len(tokenizer)
        denoiser = ReFusion(
            denoiser_config,
            tokenizer=tokenizer,
        )
    elif "esolm-" in pretrained_model_name_or_path:
        denoiser_config = EsoLMConfig(length=length)
        denoiser_config.backbone_config = OmegaConf.to_container(
            backbone_config, resolve=True
        )
        denoiser_config.keep_clean_bos = True
        denoiser_config.mask_token_id = tokenizer.mask_token_id
        denoiser_config.vocab_size = len(tokenizer)
        denoiser = EsoLM(
            denoiser_config,
            tokenizer=tokenizer,
        )
    elif "mdlm-" in pretrained_model_name_or_path:
        denoiser_config = MDLMConfig(length=length)
        denoiser_config.backbone_config = OmegaConf.to_container(
            backbone_config, resolve=True
        )
        denoiser_config.keep_clean_bos = True
        denoiser_config.mask_token_id = tokenizer.mask_token_id
        denoiser_config.vocab_size = len(tokenizer)
        denoiser = MDLM(
            denoiser_config,
            tokenizer=tokenizer,
        )
    elif "sedd-" in pretrained_model_name_or_path:
        denoiser_config = MDLMConfig(length=length)
        denoiser_config.backbone_config = OmegaConf.to_container(
            backbone_config, resolve=True
        )
        denoiser_config.keep_clean_bos = True
        denoiser_config.mask_token_id = tokenizer.mask_token_id
        denoiser_config.vocab_size = len(tokenizer)
        denoiser = SEDD(
            denoiser_config,
            tokenizer=tokenizer,
        )
    elif "ar-" in pretrained_model_name_or_path:
        denoiser_config = ARConfig(length=length, backbone_config=backbone_config)
        denoiser_config.backbone_config = OmegaConf.to_container(
            backbone_config, resolve=True
        )
        denoiser_config.keep_clean_bos = True
        denoiser_config.mask_token_id = tokenizer.mask_token_id
        denoiser_config.vocab_size = len(tokenizer)
        denoiser = AR(
            denoiser_config,
            tokenizer=tokenizer,
        )
    else:
        denoiser_config = BD3LMConfig(
            length=length,
            backbone_config=backbone_config,
            block_size=block_size,
        )
        denoiser_config.backbone_config = OmegaConf.to_container(
            backbone_config, resolve=True
        )
        denoiser_config.keep_clean_bos = True
        denoiser_config.mask_token_id = tokenizer.mask_token_id
        denoiser_config.vocab_size = len(tokenizer)
        denoiser = BD3LM(
            denoiser_config,
            tokenizer=tokenizer,
        )

    for key, value in model_config_overrides.items():
        if key == "backbone_config":
            continue
        setattr(denoiser.config, key, value)
    if backbone_overrides:
        backbone_runtime_config = getattr(denoiser.backbone, "config", None)
        if backbone_runtime_config is not None:
            for key, value in backbone_overrides.items():
                setattr(backbone_runtime_config, key, value)
        else:
            for key, value in backbone_overrides.items():
                setattr(denoiser.backbone, key, value)
    if is_refusion:
        _configure_refusion_special_token_ids(denoiser, tokenizer)

    if model is not None:
        denoiser.backbone = model.backbone if hasattr(model, "backbone") else model
    else:
        state_dict = torch.load(
            pretrained_model_name_or_path,
            map_location="cpu",
            weights_only=False,
        )["state_dict"]

        for key in list(state_dict.keys()):
            new_key = key
            if "backbone." in new_key:
                new_key = new_key.replace("backbone.", "")
            if "_orig_mod." in new_key:
                new_key = new_key.replace("_orig_mod.", "")
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)

        state_dict.pop("sampling_eps_min", None)
        state_dict.pop("sampling_eps_max", None)
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
    require_explicit_refusion_length: bool = False,
):
    model_config_overrides = normalize_model_config_overrides(model_config_overrides)
    requested_model_type = _get_explicit_requested_model_type(model_config_overrides)
    requested_refusion = requested_model_type == ReFusionConfig.model_type
    ckpt_config_path = os.path.join(pretrained_model_name_or_path, "config.yaml")
    has_ckpt_config = fsspec_exists(ckpt_config_path)
    hf_model_source = None

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
        model = _load_hf_model(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pretrained_model_revision=pretrained_model_revision,
            allow_masked_lm=not requested_refusion,
        )
        model, hf_model_source = _normalize_hf_model_load_result(model)

    if _is_refusion_model(model):
        _configure_refusion_special_token_ids(model, tokenizer)
        return model.to(device)

    if _has_refusion_forward_contract(model):
        inferred_length = _infer_refusion_rebuild_length(
            model=model,
            tokenizer=tokenizer,
            model_config_overrides=model_config_overrides,
        )
        if inferred_length is None:
            raise ValueError(
                "Official-style upstream ReFusion loading could not infer a usable "
                "sequence length for the local wrapper rebuild."
            )
        rebuild_overrides = dict(model_config_overrides)
        rebuild_overrides["model_type"] = ReFusionConfig.model_type
        rebuild_overrides["length"] = inferred_length
        return _rebuild_as_refusion_or_raise(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            device=device,
            model=model,
            model_config_overrides=rebuild_overrides,
            requested_model_type=ReFusionConfig.model_type,
            error_prefix=(
                "Official-style upstream ReFusion checkpoint could not be rebuilt "
                "as the local ReFusion wrapper"
            ),
        )

    if requested_refusion:
        error_prefix = "Explicit ReFusion request could not be satisfied"
        if hf_model_source == "masked_lm":
            raise ValueError(
                f"{error_prefix} because ReFusion requires a causal LM backbone; "
                "masked-LM fallback is not supported."
            )
        if requested_refusion and has_ckpt_config:
            raise ValueError(
                "Explicit ReFusion request could not be satisfied because the checkpoint "
                "directory did not load a ReFusion wrapper."
            )
        if requested_refusion and model is None and not has_ckpt_config:
            raise ValueError(
                f"{error_prefix} because no causal LM backbone could be loaded."
            )
        if model_config_overrides.get("length") is None and not has_ckpt_config:
            raise ValueError(
                f"{error_prefix} without an explicit `length`; rebuilding from a plain "
                "causal LM would otherwise fall back to non-ReFusion behavior and the "
                "loader's implicit default length."
            )
        return _rebuild_as_refusion_or_raise(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            device=device,
            model=model,
            model_config_overrides=model_config_overrides,
            requested_model_type=requested_model_type,
            error_prefix=error_prefix,
        )

    if model is None or (
        force_legacy_if_no_generate and not hasattr(model, "generate")
    ):
        model = _load_legacy_denoiser(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            device=device,
            model=model,
            model_config_overrides=model_config_overrides,
            requested_model_type=requested_model_type,
        )
    if _is_refusion_model(model):
        _configure_refusion_special_token_ids(model, tokenizer)
    return model.to(device)
