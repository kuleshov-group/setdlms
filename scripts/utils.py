import os
import random
from typing import Any

import fsspec
import hydra
import numpy as np
import rich.syntax
import rich.tree
import torch
import yaml
from composer.utils import dist
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer

from src.denoiser.base import Denoiser


def _make_tokenization_config(tokenizer_cfg: DictConfig) -> dict[str, Any]:
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    pad_vocab_size_multiple = getattr(tokenizer, "pad_vocab_size_multiple", 1)
    return {
        "vocab_size": len(tokenizer),
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_vocab_size_multiple": pad_vocab_size_multiple,
    }


def _get_tokenizer_eos_token_id(tokenizer_cfg: DictConfig) -> int:
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    if not hasattr(tokenizer, "eos_token_id"):
        raise ValueError("Tokenizer must have 'eos_token_id'.")
    return tokenizer.eos_token_id


def _get_tokenizer_pad_token_id(tokenizer_cfg: DictConfig) -> int:
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have 'pad_token_id'.")
    return tokenizer.pad_token_id


def _get_tokenizer_mask_token_id(tokenizer_cfg: DictConfig) -> int:
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)
    if not hasattr(tokenizer, "mask_token_id") or tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must have 'mask_token_id'.")
    return tokenizer.mask_token_id


def _get_world_size() -> int:
    # Setup distributed
    if not dist.is_initialized():
        print("Initializing dist")
        dist.initialize_dist(timeout=600)
    return dist.get_world_size()


def _tokenizer_vocab(tokenizer: PreTrainedTokenizer) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return get_vocab()
    return getattr(tokenizer, "vocab", {}) or {}


def _tokenizer_added_vocab(tokenizer: PreTrainedTokenizer) -> dict[str, int]:
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        return get_added_vocab()
    return {}


def _tokenizer_token_text(token) -> str | None:
    if token is None:
        return None
    content = getattr(token, "content", None)
    if content is not None:
        return str(content)
    return str(token)


def _get_tokenizer_special_token(tokenizer: PreTrainedTokenizer, name: str):
    token = getattr(tokenizer, name, None)
    if token is not None:
        return _tokenizer_token_text(token)
    return _tokenizer_token_text(getattr(tokenizer, f"_{name}", None))


def _ensure_tokenizer_eos_token(tokenizer: PreTrainedTokenizer):
    eos_token = _get_tokenizer_special_token(tokenizer, "eos_token")
    if eos_token is None and "<|endoftext|>" in _tokenizer_vocab(tokenizer):
        eos_token = "<|endoftext|>"
    if eos_token is None:
        raise AttributeError(
            "Tokenizer must define eos_token or _eos_token before missing "
            "special tokens can be filled."
        )
    if getattr(tokenizer, "eos_token", None) is None:
        tokenizer.eos_token = eos_token
    return eos_token


def _lookup_token_in_mapping(mapping, token: str) -> int | None:
    if not mapping:
        return None
    if token in mapping:
        return int(mapping[token])
    for key, value in mapping.items():
        if _tokenizer_token_text(key) == token:
            return int(value)
    return None


def _tokenizer_token_id(tokenizer: PreTrainedTokenizer, token: str | None) -> int | None:
    token = _tokenizer_token_text(token)
    if token is None:
        return None

    for mapping in (
        _tokenizer_added_vocab(tokenizer),
        _tokenizer_vocab(tokenizer),
        getattr(tokenizer, "added_tokens_encoder", None),
        getattr(tokenizer, "_added_tokens_encoder", None),
        getattr(tokenizer, "encoder", None),
    ):
        token_id = _lookup_token_in_mapping(mapping, token)
        if token_id is not None:
            return token_id

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        token_id = convert(token)
    except Exception:
        return None
    if not isinstance(token_id, int):
        return None

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = _get_tokenizer_special_token(tokenizer, "unk_token")
    if unk_token_id is not None and token_id == unk_token_id and token != unk_token:
        return None

    convert_back = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert_back):
        try:
            roundtrip_token = convert_back(token_id)
        except Exception:
            roundtrip_token = None
        if isinstance(roundtrip_token, list):
            roundtrip_token = roundtrip_token[0] if len(roundtrip_token) == 1 else None
        roundtrip_token = _tokenizer_token_text(roundtrip_token)
        if roundtrip_token is not None and roundtrip_token != token:
            return None
    return token_id


def _set_tokenizer_special_token_id(
    tokenizer: PreTrainedTokenizer,
    token_name: str,
    token_id: int,
) -> None:
    token_id = int(token_id)
    id_name = f"{token_name}_id"
    for attr_name in (id_name, f"_{id_name}"):
        try:
            setattr(tokenizer, attr_name, token_id)
        except Exception:
            pass


def _ensure_tokenizer_special_token_id(
    tokenizer: PreTrainedTokenizer,
    token_name: str,
) -> None:
    token = _get_tokenizer_special_token(tokenizer, token_name)
    if token is None:
        return
    id_name = f"{token_name}_id"
    if getattr(tokenizer, id_name, None) is not None:
        return
    token_id = _tokenizer_token_id(tokenizer, token)
    if token_id is not None:
        _set_tokenizer_special_token_id(tokenizer, token_name, token_id)


def _ensure_tokenizer_mask_token_registered(
    tokenizer: PreTrainedTokenizer,
    target_vocab_size: int | None = None,
) -> int:
    mask_token = _get_tokenizer_special_token(tokenizer, "mask_token")
    if mask_token is None:
        mask_token = "<|fim_middle|>"
    tokenizer.mask_token = mask_token

    token_id = _tokenizer_token_id(tokenizer, mask_token)
    if token_id is not None:
        _set_tokenizer_special_token_id(tokenizer, "mask_token", token_id)
        return int(token_id)

    try:
        before_len = len(tokenizer)
    except Exception:
        before_len = None

    if target_vocab_size is not None and before_len is not None:
        target_vocab_size = int(target_vocab_size)
        if before_len >= target_vocab_size:
            token_id = target_vocab_size - 1
            _set_tokenizer_special_token_id(tokenizer, "mask_token", token_id)
            return int(token_id)

    added_count = 0
    can_grow = target_vocab_size is None or before_len is None or before_len < target_vocab_size
    if can_grow:
        add_special_tokens = getattr(tokenizer, "add_special_tokens", None)
        if callable(add_special_tokens):
            added = add_special_tokens({"mask_token": mask_token})
            added_count += int(added or 0)

        token_id = _tokenizer_token_id(tokenizer, mask_token)
        if token_id is None:
            add_tokens = getattr(tokenizer, "add_tokens", None)
            if callable(add_tokens):
                try:
                    added = add_tokens([mask_token], special_tokens=True)
                except TypeError:
                    added = add_tokens([mask_token])
                added_count += int(added or 0)
                token_id = _tokenizer_token_id(tokenizer, mask_token)

    if token_id is None:
        try:
            after_len = len(tokenizer)
        except Exception:
            after_len = None
        if before_len is not None and (
            (after_len is not None and after_len > before_len) or added_count > 0
        ):
            token_id = before_len
        elif target_vocab_size is not None:
            token_id = int(target_vocab_size) - 1
    if token_id is None or token_id < 0:
        raise ValueError("Tokenizer must have mask_token_id.")

    _set_tokenizer_special_token_id(tokenizer, "mask_token", token_id)
    return int(token_id)


def get_tokenizer_special_token_id(
    tokenizer: PreTrainedTokenizer,
    token_name: str,
) -> int:
    if token_name == "mask_token":
        return _ensure_tokenizer_mask_token_registered(tokenizer)

    id_name = f"{token_name}_id"
    token_id = getattr(tokenizer, id_name, None)
    if token_id is not None:
        return int(token_id)
    token = _get_tokenizer_special_token(tokenizer, token_name)
    token_id = _tokenizer_token_id(tokenizer, token)
    if token_id is None:
        raise ValueError(f"Tokenizer must have {id_name}.")
    _set_tokenizer_special_token_id(tokenizer, token_name, token_id)
    return int(token_id)


def _ensure_tokenizer_save_pretrained_state(tokenizer: PreTrainedTokenizer) -> None:
    special_token_attrs = getattr(
        tokenizer,
        "SPECIAL_TOKENS_ATTRIBUTES",
        [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens",
        ],
    )
    special_tokens_map = getattr(tokenizer, "_special_tokens_map", None)
    if not isinstance(special_tokens_map, dict):
        special_tokens_map = {}
    for attr in special_token_attrs:
        if attr == "additional_special_tokens":
            special_tokens_map.setdefault(attr, [])
            continue
        special_tokens_map.setdefault(attr, None)
        token = _get_tokenizer_special_token(tokenizer, attr)
        if token is not None:
            special_tokens_map[attr] = token
    tokenizer._special_tokens_map = special_tokens_map

    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, dict):
        init_kwargs = {}
    for attr, value in special_tokens_map.items():
        if value:
            init_kwargs[attr] = value
    tokenizer.init_kwargs = init_kwargs

    if "extra_special_tokens" not in tokenizer.__dict__:
        tokenizer.extra_special_tokens = {}
    if "chat_template" not in tokenizer.__dict__:
        tokenizer.chat_template = None
    if "init_inputs" not in tokenizer.__dict__:
        tokenizer.init_inputs = ()
    if "_processor_class" not in tokenizer.__dict__:
        tokenizer._processor_class = None
    if "_auto_class" not in tokenizer.__dict__:
        tokenizer._auto_class = None
    if not hasattr(tokenizer, "verbose"):
        tokenizer.verbose = False
    if not hasattr(tokenizer, "clean_up_tokenization_spaces"):
        tokenizer.clean_up_tokenization_spaces = False
    if not hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = int(1e30)


def maybe_add_missing_special_tokens(
    tokenizer: PreTrainedTokenizer,
    target_vocab_size: int | None = None,
):
    eos_token = _ensure_tokenizer_eos_token(tokenizer)
    if getattr(tokenizer, "bos_token", None) is None:
        tokenizer.bos_token = eos_token
    if getattr(tokenizer, "pad_token", None) is None:
        added_vocab = _tokenizer_added_vocab(tokenizer)
        if "<|finetune_right_pad_id|>" in added_vocab:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = eos_token
    if getattr(tokenizer, "mask_token", None) is None:
        added_vocab = _tokenizer_added_vocab(tokenizer)
        vocab = _tokenizer_vocab(tokenizer)
        if "<|reserved_special_token_0|>" in added_vocab:
            # llama
            tokenizer.mask_token = "<|reserved_special_token_0|>"
        elif "<|fim_middle|>" in added_vocab:
            # qwen / gpt-style infilling checkpoints
            tokenizer.mask_token = "<|fim_middle|>"
        elif "_MASK" in vocab:
            tokenizer.mask_token = "_MASK"
        else:
            tokenizer.mask_token = "<|fim_middle|>"
    for token_name in ("eos_token", "bos_token", "pad_token"):
        _ensure_tokenizer_special_token_id(tokenizer, token_name)
    _ensure_tokenizer_save_pretrained_state(tokenizer)
    _ensure_tokenizer_mask_token_registered(
        tokenizer,
        target_vocab_size=target_vocab_size,
    )
    _ensure_tokenizer_save_pretrained_state(tokenizer)
    return tokenizer


def register_useful_resolvers() -> None:
    OmegaConf.register_new_resolver("cwd", os.getcwd)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver(
        "if_then_else", lambda condition, x, y: x if condition else y
    )
    OmegaConf.register_new_resolver(
        "set_backend", lambda: "gpu" if torch.cuda.is_available() else "cpu"
    )
    OmegaConf.register_new_resolver("get_world_size", lambda: _get_world_size())
    OmegaConf.register_new_resolver(
        "make_tokenization_config",
        lambda tokenizer_cfg: _make_tokenization_config(tokenizer_cfg),
    )
    OmegaConf.register_new_resolver(
        "get_tokenizer_eos_token_id",
        lambda tokenizer_cfg: _get_tokenizer_eos_token_id(tokenizer_cfg),
    )
    OmegaConf.register_new_resolver(
        "get_tokenizer_pad_token_id",
        lambda tokenizer_cfg: _get_tokenizer_pad_token_id(tokenizer_cfg),
    )
    OmegaConf.register_new_resolver(
        "get_tokenizer_mask_token_id",
        lambda tokenizer_cfg: _get_tokenizer_mask_token_id(tokenizer_cfg),
    )


def count_parameters(model: torch.nn.Module, trainable: bool = True) -> int:
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(num):
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if abs(num) >= 1_000:
        return f"{num / 1_000:.1f}k"
    return str(num)


def replace_strings(obj, old: str, new: str):
    if isinstance(obj, dict):
        return {
            (k.replace(old, new) if isinstance(k, str) else k): replace_strings(
                v, old, new
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [replace_strings(v, old, new) for v in obj]
    elif isinstance(obj, str):
        return obj.replace(old, new)
    else:
        return obj


def print_and_save_config(
    cfg: DictConfig,
    resolve: bool = True,
    save_cfg: bool = True,
    save_dir: str | None = None,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      cfg (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
      save_dir (Optional[str]): Directory to save the configuration tree to.
        If None, defaults to `os.getcwd()`.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = cfg.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = cfg.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        save_dir = save_dir if save_dir is not None else os.getcwd()
        with fsspec.open(os.path.join(save_dir, "config_tree.txt"), "w") as fp:
            rich.print(tree, file=fp)
        with fsspec.open(os.path.join(save_dir, "config.yaml"), "w") as fp:
            OmegaConf.save(cfg, fp, resolve=resolve)


def load_model_from_ckpt_dir_path(
    path_to_ckpt_dir: str,
    ckpt_file: str = "best-rank0.pt",
    load_ema_weights: bool = False,
    verbose: bool = False,
    device: torch.device | str = torch.device("cpu"),
    **model_config_overrides,
) -> Denoiser:
    """Load a model from a checkpoint path (and file).

    Args:
        path_to_ckpt_dir (str): Path to the checkpoint directory.
            Assumed to have `checkpoints` subdirectory with checkpoint file(s).
        ckpt_file (str): Name of the checkpoint file inside `checkpoints` directory.
            Defaults to "best-rank0.pt".
        load_ema_weights (bool): Whether to load the EMA weights. Defaults to False.
        verbose (bool): Whether to print information about the loaded checkpoint,
            e.g., step, metric values. Defaults to False.
        device (torch.device | str): Device for torch.load(map_location=).
            Defaults to torch.device("cpu").
        model_config_overrides (dict[str, Any]): Optional overrides for
            `config.model.config`. Supports both top-level and nested overrides.
            For nested dictionaries, values are merged recursively. For example:
            `{"backbone_config": {"num_layers": 32}}` will update only `num_layers`
            while preserving other keys in `backbone_config`.

    Returns:
        Denoiser: The loaded denoiser model.
    """

    def _replace_in_state_dict_if_present(
        sd: dict[str, Any],
        prefix: str,
        replacement: str = "",
    ) -> None:
        """Replace string in the prefix in state_dict (sd) in place, if any.

        Args:
            sd (OrderedDict): a state-dict to be loaded to the model.
            prefix (str): prefix.
            replacement (Optional; str): replacement string.
        """
        keys = list(sd.keys())
        for key in keys:
            if prefix in key:
                newkey = key.replace(prefix, replacement)
                sd[newkey] = sd.pop(key)

        # also strip the prefix in metadata if any.
        if hasattr(sd, "_metadata"):
            keys = list(sd._metadata.keys())
            for key in keys:
                # for the metadata dict, the key can be:
                # '': for the DDP module, which we want to remove.
                # 'module': for the actual model.
                # 'module.xx.xx': for the rest.
                if len(key) == 0:
                    continue
                # handling both, 'module' case and  'module.' cases
                if key == prefix.replace(".", "") or prefix in key:
                    newkey = key.replace(prefix, replacement)
                    sd._metadata[newkey] = sd._metadata.pop(key)

    def _deep_update(base_dict: dict, update_dict: dict) -> None:
        """Recursively update a dictionary with values from another dictionary.

        Args:
            base_dict: The dictionary to update in-place.
            update_dict: The dictionary containing updates.
        """
        # Convert any DictConfig values to plain dicts for proper merging
        if isinstance(update_dict, DictConfig):
            update_dict = OmegaConf.to_container(update_dict, resolve=True)

        for k, v in update_dict.items():
            # Convert nested DictConfig values to plain dicts
            if isinstance(v, DictConfig):
                v = OmegaConf.to_container(v, resolve=True)

            if (
                k in base_dict
                and isinstance(base_dict[k], dict)
                and isinstance(v, dict)
            ):
                _deep_update(base_dict[k], v)
            else:
                base_dict[k] = v

    def _strip_legacy_kwargs(config_node: Any) -> None:
        """Remove deprecated Hydra kwargs kept in older checkpoint configs."""
        if isinstance(config_node, dict):
            legacy_kwargs_by_target = {
                "src.backbone.dit.DIT": {"norm_type"},
                "src.noise_schedule.noise_schedules.EaseOutPowerNoise": {
                    "plot_schedule",
                    "int_min",
                },
                "src.noise_schedule.noise_schedules.StaggeredNoise": {
                    "plot_schedule",
                    "int_min",
                },
            }
            target = config_node.get("_target_")
            removed_keys = []
            for key in legacy_kwargs_by_target.get(target, set()):
                if key in config_node:
                    config_node.pop(key)
                    removed_keys.append(key)
            if verbose and removed_keys:
                print(
                    f"Removed legacy config keys {sorted(removed_keys)} "
                    f"for target {target}"
                )
            for value in config_node.values():
                _strip_legacy_kwargs(value)
        elif isinstance(config_node, list):
            for value in config_node:
                _strip_legacy_kwargs(value)

    with open(os.path.join(path_to_ckpt_dir, "config.yaml"), "rb") as f:
        config = yaml.safe_load(f)
    if model_config_overrides:
        # Convert OmegaConf DictConfig to plain dict to ensure proper merging
        # This is important because nested DictConfigs don't pass
        # isinstance(..., dict) checks
        overrides_dict = (
            OmegaConf.to_container(model_config_overrides, resolve=True)
            if isinstance(model_config_overrides, DictConfig)
            else model_config_overrides
        )
        _deep_update(config["model"]["config"], overrides_dict)
    config = replace_strings(config, "AnyOrderBD3LM", "SetDLM")
    config = replace_strings(config, "EaseOutPowerNoise", "StaggeredNoise")
    _strip_legacy_kwargs(config)
    config = OmegaConf.create(config)

    model = hydra.utils.instantiate(
        config.model,
        _convert_="all",
    )
    try:
        ckpt = torch.load(
            os.path.join(path_to_ckpt_dir, "checkpoints", ckpt_file),
            weights_only=False,
            map_location=device,
        )
    except FileNotFoundError:
        print("Checkpoint not found; reinitializing model from scratch")
        return model
    if verbose:
        if "timestamp" in ckpt["state"]:
            timestamp = ckpt["state"]["timestamp"]
            print(
                f"Loaded ckpt from ep: {timestamp['Timestamp']['epoch']}; "
                f"batch: {timestamp['Timestamp']['batch']}."
            )
        elif (
            "callbacks" in ckpt["state"]
            and "SaveBestCheckpointing" in ckpt["state"]["callbacks"]
        ):
            timestamp = ckpt["state"]["callbacks"]["SaveBestCheckpointing"][
                "all_saved_checkpoints_to_timestamp"
            ][-1][1]
            metric_dict = ckpt["state"]["callbacks"]["SaveBestCheckpointing"][
                "all_saved_checkpoints_to_timestamp"
            ][-1][-1]
            print(
                f"Loaded ckpt from ep: {timestamp['epoch']}; "
                f"batch: {timestamp['batch']}."
            )
            print("Metric value at best checkpoint:", metric_dict)

    def _normalize_checkpoint_state_dict_keys(state_dict: dict[str, Any]) -> None:
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, "module."
        )
        _replace_in_state_dict_if_present(
            state_dict, "_orig_mod."
        )  # for compiled models
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, "model."
        )

    def _sync_ema_tied_lm_head(
        state_dict: dict[str, Any],
        ema_state_dict: dict[str, Any],
    ) -> None:
        """Use EMA embeddings for tied LM heads omitted from Composer EMA state."""

        tied_weight_pairs = (
            (
                "backbone.model.model.embed_tokens.weight",
                "backbone.model.lm_head.weight",
            ),
            (
                "backbone.model.model.decoder.embed_tokens.weight",
                "backbone.model.lm_head.weight",
            ),
            (
                "backbone.model.transformer.wte.weight",
                "backbone.model.lm_head.weight",
            ),
            ("model.embed_tokens.weight", "lm_head.weight"),
            ("model.decoder.embed_tokens.weight", "lm_head.weight"),
            ("transformer.wte.weight", "lm_head.weight"),
        )
        for embed_key, lm_head_key in tied_weight_pairs:
            if embed_key not in ema_state_dict:
                continue
            if lm_head_key not in state_dict or lm_head_key in ema_state_dict:
                continue
            embed_weight = ema_state_dict[embed_key]
            lm_head_weight = state_dict[lm_head_key]
            if getattr(embed_weight, "shape", None) != getattr(
                lm_head_weight,
                "shape",
                None,
            ):
                continue
            state_dict[lm_head_key] = embed_weight
            if verbose:
                print(
                    "Using EMA embedding weights for tied LM head omitted from "
                    f"EMA state: {embed_key} -> {lm_head_key}"
                )

    state_dict = dict(ckpt["state"]["model"])
    _normalize_checkpoint_state_dict_keys(state_dict)
    if load_ema_weights:
        ema_state_dict = None
        for alg in ckpt["state"]["algorithms"]:
            # algorithms stored as list[tuple[str, dict]]
            if alg[0] == "EMA":
                ema_state_dict = dict(alg[1]["ema_model"]["named_parameters_dict"])
                break
        if ema_state_dict is None:
            raise ValueError("EMA weights not found in checkpoint.")
        _normalize_checkpoint_state_dict_keys(ema_state_dict)
        state_dict.update(ema_state_dict)
        _sync_ema_tied_lm_head(state_dict, ema_state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    allowed_missing_keys = {"static_attention_mask"}
    bad_missing_keys = [k for k in missing_keys if k not in allowed_missing_keys]
    if any(k.endswith("lm_head.weight") for k in bad_missing_keys):
        raise RuntimeError(
            "Checkpoint load is missing the LM head after EMA merge: "
            f"missing={bad_missing_keys}, unexpected={unexpected_keys}"
        )
    if bad_missing_keys or unexpected_keys:
        print(
            "Warning: checkpoint load had non-fatal key mismatch: "
            f"missing={bad_missing_keys}, unexpected={unexpected_keys}"
        )
    return model


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
