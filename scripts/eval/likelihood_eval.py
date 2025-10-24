import logging
import os

import hydra
import torch.distributed as torch_dist
from composer.models import HuggingFaceModel
from composer.utils import dist, reproducibility
from omegaconf import DictConfig
from streaming import StreamingDataset
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
)
from src.utils import fsspec_exists

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    reproducibility.seed_all(cfg.seed)
    reproducibility.configure_deterministic_mode()

    # Load tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load model
    if fsspec_exists(os.path.join(cfg.pretrained_model_name_or_path, "config.yaml")):
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=cfg.pretrained_model_name_or_path,
            load_ema_weights=cfg.task.load_ema_weights,
            ckpt_file=cfg.task.ckpt_file,
            verbose=True,
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model_name_or_path,
                trust_remote_code=True,
                revision=getattr(cfg, "pretrained_model_revision", None),
            )
        except:  # Model not compatible with CausalLM
            model = AutoModelForMaskedLM.from_pretrained(
                cfg.pretrained_model_name_or_path,
                trust_remote_code=True,
                revision=getattr(cfg, "pretrained_model_revision", None),
            )
    model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        metrics=list(hydra.utils.instantiate(cfg.task.metrics).values()),
    )

    # Setup distributed
    if not dist.is_initialized():
        log.info("Initializing dist")
        dist.initialize_dist(timeout=600)
    log.info("All nodes connected")

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
    eval_sampler = (
        dist.get_sampler(eval_dataset, shuffle=False, drop_last=False)
        if not isinstance(eval_dataset, StreamingDataset)
        else None
    )

    eval_dataloader = hydra.utils.instantiate(
        cfg.task.eval_dataloader,
        _convert_="partial",
        dataset=eval_dataset,
        collate_fn=collator,
        sampler=eval_sampler,
    )

    trainer = hydra.utils.instantiate(
        cfg.task.trainer,
        _convert_="all",
        model=model,
        eval_dataloader=eval_dataloader,
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
