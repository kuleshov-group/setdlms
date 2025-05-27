import os
from argparse import ArgumentParser

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


def main(args):
    config = OmegaConf.load(os.path.join(args.model_path, "config.yaml"))
    reproducibility.seed_all(config.seed)

    config.composer.trainer.autoresume = False
    config.composer.trainer.save_folder = "~/trash"

    print_and_save_config(config, resolve=True, save_cfg=False)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.pretrained_model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    model = load_model_from_ckpt_dir_path(
        path_to_ckpt_dir=args.model_path,
        load_ema_weights=False,
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
    register_useful_resolvers()
    parser = ArgumentParser(description="Likelihood evaluation script")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output file",
    )
    opts = parser.parse_args()
    main(opts)
