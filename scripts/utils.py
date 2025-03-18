import os
from typing import Any

import fsspec
import hydra
import rich.syntax
import rich.tree
import torch
from composer.utils import dist
from omegaconf import DictConfig, OmegaConf


def _make_tokenization_config(tokenizer_cfg: DictConfig) -> dict[str, Any]:
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    pad_vocab_size_multiple = (
        tokenizer.pad_vocab_size_multiple
        if hasattr(tokenizer, "pad_vocab_size_multiple")
        else 8
    )
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
