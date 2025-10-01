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


def _get_world_size() -> int:
    # Setup distributed
    if not dist.is_initialized():
        print("Initializing dist")
        dist.initialize_dist(timeout=600)
    return dist.get_world_size()


def maybe_add_missing_special_tokens(tokenizer: PreTrainedTokenizer):
    if getattr(tokenizer, "bos_token", None) is None:
        tokenizer.bos_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token", None) is None:
        if hasattr(tokenizer, "get_added_vocab"):
            if "<|finetune_right_pad_id|>" in tokenizer.get_added_vocab().keys():
                tokenizer.pad_token = "<|finetune_right_pad_id|>"
            else:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "mask_token", None) is None:
        if hasattr(tokenizer, "get_added_vocab"):
            if "<|reserved_special_token_0|>" in tokenizer.get_added_vocab().keys():
                # llama
                tokenizer.mask_token = "<|reserved_special_token_0|>"
                tokenizer.mask_token_id = tokenizer.get_added_vocab()[
                    "<|reserved_special_token_0|>"
                ]
            elif "<|fim_middle|>" in tokenizer.get_added_vocab().keys():
                # qwen
                tokenizer.mask_token = "<|fim_middle|>"
                tokenizer.mask_token_id = tokenizer.get_added_vocab()["<|fim_middle|>"]
            elif "_MASK" in tokenizer.vocab:
                tokenizer.mask_token = "_MASK"
                tokenizer.mask_token_id = tokenizer.vocab["_MASK"]
            else:
                # Set mask token id == vocab size
                special_tokens_dict = {"mask_token": "<|fim_middle|>"}
                tokenizer.add_special_tokens(special_tokens_dict)
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
            `config.model.config`.
            Currently, this only supports overriding entries in config.model.config,
            however overriding a nested entry value,
            e.g., config.model.config.backbone_config.num_layers, will not work.

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
                sd[newkey] = state_dict.pop(key)

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

    with open(os.path.join(path_to_ckpt_dir, "config.yaml"), "rb") as f:
        config = yaml.safe_load(f)
    for k, v in model_config_overrides.items():
        config["model"]["config"][k] = v
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
        if (
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

    if load_ema_weights:
        state_dict = None
        for alg in ckpt["state"]["algorithms"]:
            # algorithms stored as list[tuple[str, dict]]
            if alg[0] == "EMA":
                state_dict = alg[1]["ema_model"]["named_parameters_dict"]
                break
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, "module."
        )
        if state_dict is None:
            raise ValueError("EMA weights not found in checkpoint.")
    else:
        state_dict = ckpt["state"]["model"]
    _replace_in_state_dict_if_present(state_dict, "_orig_mod.")  # for compiled models
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "model.")
    model.load_state_dict(state_dict, strict=False)

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
