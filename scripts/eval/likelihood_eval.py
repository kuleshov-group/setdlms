import logging
import os
import random
from numbers import Integral
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as torch_dist
from composer.models import HuggingFaceModel
from composer.utils import dist, reproducibility
from omegaconf import DictConfig

from scripts.eval.model_loading import (
    load_eval_model,
    maybe_load_legacy_checkpoint_tokenizer,
    normalize_model_config_overrides,
)
from scripts.utils import (
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
)
from src.datasets.collator import DenoisingCollator
from src.denoiser.esolm import EsoLM
from src.denoiser.refusion import ReFusion

log = logging.getLogger(__name__)


class RepeatDataloader:
    """Repeat an existing dataloader for k full passes."""

    def __init__(self, dataloader, k: int):
        self.dataloader = dataloader
        self.k = k

    def __iter__(self):
        for _ in range(self.k):
            yield from iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader) * self.k


def require_refusion_semantics(cfg: DictConfig) -> bool:
    return bool(getattr(cfg.task, "require_refusion_semantics", False))


def _validate_refusion_length(value: Any, source_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            "Likelihood eval requested ReFusion semantics, but "
            f"`{source_name}` must be a positive finite integer sequence length. "
            "Refusing non-ReFusion behavior."
        )
    length = int(value)
    if length <= 0 or length >= 1_000_000:
        raise ValueError(
            "Likelihood eval requested ReFusion semantics, but "
            f"`{source_name}={length}` is not a usable explicit sequence length. "
            "Refusing non-ReFusion behavior."
        )
    return length


def build_likelihood_model_config_overrides(cfg: DictConfig) -> dict[str, Any]:
    model_config_overrides = normalize_model_config_overrides(
        getattr(cfg, "model_config_overrides", None)
    )
    if not require_refusion_semantics(cfg):
        return model_config_overrides
    model_config_overrides["model_type"] = "refusion"
    _validate_refusion_length(
        model_config_overrides.get("length"),
        "model_config_overrides.length",
    )
    return model_config_overrides


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    reproducibility.seed_all(cfg.seed)
    reproducibility.configure_deterministic_mode()
    local_rank = dist.get_local_rank()

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Setup distributed
    if not dist.is_initialized():
        log.info("Initializing dist")
        dist.initialize_dist(timeout=600)
    log.info("All nodes connected")
    rank_seed = int(cfg.seed) + dist.get_global_rank()
    torch.manual_seed(rank_seed)
    np.random.seed(rank_seed)
    random.seed(rank_seed)

    # Load tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    checkpoint_tokenizer = maybe_load_legacy_checkpoint_tokenizer(
        cfg.pretrained_model_name_or_path
    )
    if checkpoint_tokenizer is not None:
        tokenizer = checkpoint_tokenizer
    else:
        tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load model
    model_config_overrides = build_likelihood_model_config_overrides(cfg)
    loaded_model = load_eval_model(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        device=device,
        pretrained_model_revision=getattr(cfg, "pretrained_model_revision", None),
        load_ema_weights=cfg.task.load_ema_weights,
        ckpt_file=cfg.task.ckpt_file,
        model_config_overrides=model_config_overrides,
        verbose=True,
        force_legacy_if_no_generate=True,
        require_explicit_refusion_length=require_refusion_semantics(cfg),
    )
    if require_refusion_semantics(cfg) and not isinstance(loaded_model, ReFusion):
        raise ValueError(
            "Likelihood eval requested ReFusion semantics, but loading returned a "
            f"non-local `ReFusion` wrapper ({type(loaded_model).__name__}). Refusing "
            "non-ReFusion behavior."
        )

    if getattr(cfg, "compile_backbone", False):
        log.info("Compiling model backbone")
        loaded_model.backbone = torch.compile(
            loaded_model.backbone, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
    model = HuggingFaceModel(
        model=loaded_model,
        tokenizer=tokenizer,
        metrics=list(hydra.utils.instantiate(cfg.task.metrics).values()),
    )

    print(f"Running likelihood eval for {cfg.task.eval_dataset}")
    eval_dataset = hydra.utils.instantiate(
        cfg.task.eval_dataset, tokenizer=tokenizer, max_length=model.config.length
    )

    collator = hydra.utils.instantiate(
        cfg.task.collator,
        rank=dist.get_global_rank(),
        world_size=dist.get_world_size(),
        tokenizer=tokenizer,
        max_length=model.config.length,
    )
    if isinstance(loaded_model, EsoLM) and isinstance(collator, DenoisingCollator):
        # Upstream Eso-LMs samples diffusion timesteps inside `algo.nll()` from the
        # active diffusion sub-batch. Do not inject generic collator-side `t`.
        collator.sample_t = False
    eval_sampler = dist.get_sampler(eval_dataset, shuffle=False, drop_last=False)

    eval_dataloader = hydra.utils.instantiate(
        cfg.task.eval_dataloader,
        _convert_="partial",
        dataset=eval_dataset,
        collate_fn=collator,
        sampler=eval_sampler,
    )
    num_importance_samples = getattr(cfg, "num_importance_samples", 1)
    if num_importance_samples > 1:
        eval_dataloader = RepeatDataloader(eval_dataloader, num_importance_samples)

    if hasattr(cfg, "composer") and hasattr(cfg.composer, "callbacks"):
        callbacks = hydra.utils.instantiate(cfg.composer.callbacks)
        callbacks = list(callbacks.values())
    else:
        callbacks = []

    trainer = hydra.utils.instantiate(
        cfg.task.trainer,
        _convert_="all",
        model=model,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
    )
    trainer.eval()
    print(
        "\nEval Metrics:\n\t"
        + "\n\t".join(
            [
                f"{k}: {v.item():0.4f}"
                for k, v in trainer.state.eval_metric_values.items()
            ]
        )
    )

    if torch_dist.is_initialized():
        torch_dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
