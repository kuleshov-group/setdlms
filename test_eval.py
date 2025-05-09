import os

import hydra
from composer.models import HuggingFaceModel
from composer.utils import dist, reproducibility
from omegaconf import OmegaConf
from streaming import StreamingDataset
from transformers import AutoTokenizer

from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    print_and_save_config,
    register_useful_resolvers,
)

register_useful_resolvers()

DIR_NAME = "/home/ubuntu/qwen3_600M_gsm8k_ckpts"


def main():
    config = OmegaConf.load(os.path.join(DIR_NAME, "config.yaml"))
    reproducibility.seed_all(config.seed)

    config.composer.trainer.autoresume = False
    config.composer.trainer.save_folder = "/home/ubuntu/trash"

    print_and_save_config(config, resolve=True, save_cfg=False)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.pretrained_model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    model = load_model_from_ckpt_dir_path(
        path_to_ckpt_dir=DIR_NAME,
        load_ema_weights=True,
        ckpt_file="best-rank0.pt",
        verbose=True,
    ).to("cuda")

    model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        metrics=list(hydra.utils.instantiate(config.metrics).values()),
    )

    eval_dataset = hydra.utils.instantiate(
        config.eval_dataset,
        tokenizer=tokenizer,
    )

    collator = hydra.utils.instantiate(config.collator, tokenizer=tokenizer)
    eval_sampler = (
        dist.get_sampler(eval_dataset, shuffle=False, drop_last=False)
        if not isinstance(eval_dataset, StreamingDataset)
        else None
    )

    eval_dataloader = hydra.utils.instantiate(
        config.eval_dataloader,
        _convert_="partial",
        dataset=eval_dataset,
        collate_fn=collator,
        sampler=eval_sampler,
    )

    trainer = hydra.utils.instantiate(
        config.composer.trainer,
        _convert_="all",
        model=model,
        eval_dataloader=eval_dataloader,
        loggers=None,
    )
    metrics = trainer.eval()
    print(metrics)


if __name__ == "__main__":
    main()
