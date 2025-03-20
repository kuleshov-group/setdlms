import logging

import hydra
from composer.models import HuggingFaceModel
from composer.utils import dist, reproducibility
from omegaconf import DictConfig, OmegaConf
from streaming import StreamingDataset

from scripts.utils import (
    format_number,
    print_and_save_config,
    register_useful_resolvers,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    print_and_save_config(cfg, resolve=True, save_cfg=True)
    reproducibility.seed_all(cfg.seed)

    # Tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = hydra.utils.instantiate(
        cfg.model,
        _convert_="all",  # required to enable json-serialization when saving checkpoint
    )
    model = HuggingFaceModel(
        model,
        tokenizer=tokenizer,
        metrics=list(hydra.utils.instantiate(cfg.metrics).values()),
    )
    log.info(
        f"Num. parameters: {format_number(sum(p.numel() for p in model.parameters()))}"
    )

    # Setup distributed
    if not dist.is_initialized():
        log.info("Initializing dist")
        dist.initialize_dist()
    log.info("All nodes connected")

    # Collator
    collator = hydra.utils.instantiate(cfg.collator, tokenizer=tokenizer)

    # Train dataloader
    train_dataset = hydra.utils.instantiate(
        cfg.train_dataset,
        tokenizer=tokenizer,
    )
    train_sampler = (
        dist.get_sampler(train_dataset, shuffle=True, drop_last=True)
        if not isinstance(train_dataset, StreamingDataset)
        else None
    )
    train_dataloader = hydra.utils.instantiate(
        cfg.train_dataloader,
        _convert_="partial",
        dataset=train_dataset,
        collate_fn=collator,
        sampler=train_sampler,
    )
    # time.sleep(30)  # Needed for multi-node training

    # Val dataloader
    eval_dataset = hydra.utils.instantiate(
        cfg.eval_dataset,
        tokenizer=tokenizer,
    )
    eval_sampler = (
        dist.get_sampler(eval_dataset, shuffle=False, drop_last=False)
        if not isinstance(eval_dataset, StreamingDataset)
        else None
    )
    eval_dataloader = hydra.utils.instantiate(
        cfg.eval_dataloader,
        _convert_="partial",
        dataset=eval_dataset,
        collate_fn=collator,
        sampler=eval_sampler,
    )
    # time.sleep(30)  # Needed for multi-node training

    # Optimizer
    optimizer = hydra.utils.instantiate(
        cfg.composer.optimizer,
        _convert_="all",  # required for compatibility with fsdp
        params=model.parameters(),
    )

    # LR Scheduler
    lr_scheduler = hydra.utils.instantiate(cfg.composer.lr_scheduler)

    # Loggers
    if cfg.composer.loggers is not None:
        logger = hydra.utils.instantiate(
            cfg.composer.loggers,
            _recursive_=False,
            # Prevents config->DictConfig in trainer init; breaks WandB config logging
            _convert_="all",
            init_kwargs={"config": OmegaConf.to_container(cfg, resolve=True)},
        )
    else:
        logger = None

    # Callbacks
    callbacks = hydra.utils.instantiate(cfg.composer.callbacks)

    # Algorithms
    algorithms = hydra.utils.instantiate(cfg.composer.algorithms)

    # Trainer
    trainer = hydra.utils.instantiate(
        cfg.composer.trainer,
        _convert_="all",
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        # TODO: since not using .values() view, need way to access algo.ema.ema_model
        algorithms=list(algorithms.values()),
        loggers=logger,
        callbacks=list(callbacks.values()),
    )

    trainer.fit()

    # TODO: when training is done save / push ema params to hub
    # TODO: check that the ema_model is same as the one from trainer.state
    # algorithms.ema.ema_model.named_parameters()

    # Clean up `tmp` dir potentially created StreamingDataset
    if hasattr(train_dataset, "remove_tmp_files"):
        train_dataset.remove_tmp_files()
    if hasattr(eval_dataset, "remove_tmp_files"):
        eval_dataset.remove_tmp_files()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
