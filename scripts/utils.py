import os
from typing import Any

import fsspec
import hydra
import rich.syntax
import rich.tree
import torch
import yaml
from composer.utils import dist
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer

from src.denoiser import Denoiser


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


def _get_world_size() -> int:
    # Setup distributed
    if not dist.is_initialized():
        print("Initializing dist")
        dist.initialize_dist()
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
                raise ValueError("[MASK] token not specified for this tokenizer")
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


def format_number(num):
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if abs(num) >= 1_000:
        return f"{num / 1_000:.1f}k"
    return str(num)


def print_and_save_config(
    cfg: DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      cfg (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
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
        with fsspec.open(f"{os.getcwd()}/config_tree.txt", "w") as fp:
            rich.print(tree, file=fp)
        with fsspec.open(f"{os.getcwd()}/config.yaml", "w") as fp:
            OmegaConf.save(cfg, fp, resolve=resolve)


def load_model_from_ckpt_dir_path(
    path_to_ckpt_dir: str,
    ckpt_file: str = "best-rank0.pt",
    load_ema_weights: bool = True,
) -> Denoiser:
    """Load a model from a checkpoint path (and file).

    Args:
        path_to_ckpt_dir (str): Path to the checkpoint directory.
            Assumed to have `checkpoints` subdirectory with checkpoint file(s).
        ckpt_file (str): Name of the checkpoint file inside `checkpoints` directory.
            Defaults to "best-rank0.pt".
        load_ema_weights (bool): Whether to load the EMA weights. Defaults to True.

    Returns:
        Denoiser: The loaded denoiser model.
    """

    with open(os.path.join(path_to_ckpt_dir, "config.yaml"), "rb") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.create(config)

    model = hydra.utils.instantiate(
        config.model,
        _convert_="all",
    )

    ckpt = torch.load(
        os.path.join(path_to_ckpt_dir, "checkpoints", ckpt_file), weights_only=False
    )
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
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "model.")
    model.load_state_dict(state_dict, strict=False)

    return model
