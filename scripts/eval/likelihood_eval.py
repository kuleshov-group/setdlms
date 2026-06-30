import logging
import random
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as torch_dist
from composer.models import HuggingFaceModel
from composer.utils import dist, reproducibility
from omegaconf import DictConfig

from scripts.eval.model_loading import (
    configure_rank_local_torchinductor_cache,
    load_eval_model,
    maybe_load_legacy_checkpoint_tokenizer,
    normalize_model_config_overrides,
)
from scripts.utils import (
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
)

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
    if checkpoint_tokenizer is not None and getattr(
        checkpoint_tokenizer, "eos_token", None
    ) is not None:
        tokenizer = checkpoint_tokenizer
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load model
    model_config_overrides = normalize_model_config_overrides(
        getattr(cfg, "model_config_overrides", None)
    )
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
    )

    if getattr(cfg, "compile_backbone", False):
        cache_dir = configure_rank_local_torchinductor_cache()
        if cache_dir:
            log.info("Using rank-local TorchInductor cache: %s", cache_dir)
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
