# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import os
import shutil
from collections.abc import Sequence
from itertools import islice
from typing import Any, Union

import numpy as np
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from src.datasets.collator import ConcatenatedSequenceCollatorWrapper

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def build_tokenizer(
    om_tokenizer_config: DictConfig,
) -> Tokenizer:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    resolved_om_tokenizer_config = OmegaConf.to_container(
        om_tokenizer_config, resolve=True
    )
    tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
        "kwargs", {}
    )
    tokenizer_name = resolved_om_tokenizer_config["name"]  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # HuggingFace does not respect the model_max_length kwarg, and overrides it with
    # min(kwargs['model_max_length'], original_config['model_max_length']), so we
    # explicitly set it here
    tokenizer.model_max_length = tokenizer_kwargs.get("model_max_length", int(1e30))

    return tokenizer


class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.
    Args:
            tokenizer (Tokenizer): HuggingFace tokenizer to tokenize samples.
            max_length (int): The max sequence length of each sample.
            streams (Sequence[Stream], optional): One or more Streams to stream/cache
                samples, from which may be upsampled or downsampled. StreamingDataset
                uses either ``streams`` or ``remote``/``local``.
                Defaults to ``None``.
            remote (str, optional): Remote path or directory to download the dataset
                frOmegaConf. If ``None``, its dataset must exist locally.
                StreamingDataset uses either ``streams`` or ``remote``/``local``.
                Defaults to ``None``.
            local (str, optional): Local working directory to download shards to.
                This is where shards are cached while they are being used. Uses a temp
                directory if not set. StreamingDataset uses either ``streams`` or
                ``remote``/``local``.
                Defaults to ``None``.
            split (str, optional): Which dataset split to use, if any. If provided, we
                stream from/to ``split`` subdirs of ``remote`` and ``local``.
                Defaults to ``None``.
            download_retry (int): Number of download re-attempts before giving up.
                Defaults to ``2``.
            download_timeout (float): Number of seconds to wait for a shard to download
                before raising an exception.
                Defaults to ``60``.
            validate_hash (str, optional): Optional hash or checksum algorithm to use to
                validate shards.
                Defaults to ``None``.
            keep_zip (bool): Whether to keep or delete the compressed form when
                decompressing downloaded shards. If ``False``, keep iff remote is local
                or no remote.
                Defaults to `False``.
            samples_per_epoch (int, optional): Provide this field iff you are weighting
            sub-datasets proportionally.
            Defaults to ``None``.
            predownload (int, optional): Target number of samples ahead to download the
                shards of while iterating.
                Defaults to ``100_000``.
            partition_algo (str): Which partitioning algorithm to use. Defaults to
                ``orig``.
            num_canonical_nodes (int, optional): Canonical number of nodes for shuffling
                with resumption.
                Defaults to ``None``, which is interpreted as the number of nodes of
                the initial run.
            batch_size (int, optional): Batch size of its DataLoader, which affects how
                the dataset is partitioned over the workers.
                Defaults to ``None``.
            shuffle (bool): Whether to iterate over the samples in randomized order.
                Defaults to ``False``.
            shuffle_algo (str): Which shuffling algorithm to use.
                Defaults to ``py1s``.
            shuffle_seed (int): Seed for Deterministic dataset shuffling.
                Defaults to ``9176``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int,
        streams: Sequence[Stream] | None = None,
        remote: str | None = None,
        local: str | None = None,
        split: str | None = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: str | None = None,
        keep_zip: bool = False,
        samples_per_epoch: int | None = None,
        predownload: int = 100_000,
        partition_algo: str = "orig",
        num_canonical_nodes: int | None = None,
        batch_size: int | None = None,
        shuffle: bool = False,
        shuffle_algo: str = "py1s",
        shuffle_seed: int = 9176,
        **kwargs: dict[str, Any],
    ):
        group_method = kwargs.pop("group_method", None)
        if group_method is not None:
            raise NotImplementedError(
                "group_method is deprecated and has been removed.\nTo "
                + "concatenate, use the --concat_tokens "
                + "argument when creating your MDS dataset with concat_c4.py"
            )

        if kwargs is not None and len(kwargs) > 0:
            raise ValueError(
                f"StreamingTextDataset() got an unexpected keyword argument: {kwargs}"
            )

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f"local directory {local} does not contain split {split}"
                    )

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=samples_per_epoch,
            predownload=predownload,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

        # How to tokenize a text sample to a token sample

    def _tokenize(self, text_sample):
        if (
            hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None
        ) or not (
            hasattr(self.tokenizer, "pad_token")
            or hasattr(self.tokenizer, "_pad_token")
        ):
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token; causes bugs
            raise RuntimeError(
                "If tokenizing on-the-fly, tokenizer must have a pad_token_id"
            )

        tokens = self.tokenizer(
            text_sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return tokens

    def _read_binary_tokenized_sample(self, sample):
        return torch.from_numpy(
            np.frombuffer(sample["tokens"], dtype=np.int64)[: self.max_length].copy()
        )

    # How to process a sample
    def __getitem__(self, idx: int) -> dict[str, Any] | torch.Tensor:
        sample = super().__getitem__(idx)
        if "text" in sample:
            sample = self._tokenize(sample)
        elif "tokens" in sample:
            sample = self._read_binary_tokenized_sample(sample)
        else:
            raise RuntimeError(
                "StreamingTextDataset needs samples to have a `text` or `tokens` column"
            )
        return sample

    def remove_tmp_files(self) -> None:
        """Remove temporary files."""
        if getattr(self, "streams", None) is not None:
            shutil.rmtree(self.streams[0].local)


def build_text_dataloader(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    batch_size: int,
):
    assert cfg.name == "text", f"Tried to build dataloader with cfg.name={cfg.name}"
    if cfg.dataset.get("group_method", None) is not None:
        raise NotImplementedError(
            "group_method is deprecated and has been removed.\nTo "
            + "concatenate, use the --concat_tokens "
            + "argument when creating your MDS dataset with convert_dataset.py"
        )

    # build streams
    streams_dict = cfg.dataset.get("streams", None)
    streams = None
    if streams_dict is not None:
        streams = []
        for stream in streams_dict.values():
            streams.append(
                Stream(
                    remote=stream.get("remote", None)
                    or cfg.dataset.get("remote", None),
                    local=stream.get("local", None) or cfg.dataset.get("local", None),
                    split=stream.get("split", None) or cfg.dataset.get("split", None),
                    proportion=stream.get("proportion", None),
                    repeat=stream.get("repeat", None),
                    download_retry=stream.get("download_retry", None)
                    or cfg.dataset.get("download_retry", 2),
                    download_timeout=stream.get("download_timeout", None)
                    or cfg.dataset.get("download_timeout", 60),
                    validate_hash=stream.get("validate_hash", None)
                    or cfg.dataset.get("validate_hash", None),
                    keep_zip=stream.get("keep_zip", None)
                    or cfg.dataset.get("keep_zip", False),
                )
            )

    # build dataset potentially with streams
    dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        max_length=cfg.dataset.max_length,
        streams=streams,
        remote=cfg.dataset.get("remote", None),
        local=cfg.dataset.get("local", None),
        split=cfg.dataset.get("split", None),
        download_retry=cfg.dataset.get("download_retry", 2),
        download_timeout=cfg.dataset.get("download_timeout", 60),
        validate_hash=cfg.dataset.get("validate_hash", None),
        keep_zip=cfg.dataset.get("keep_zip", False),
        keep_raw=cfg.dataset.get("keep_raw", True),
        samples_per_epoch=cfg.dataset.get("samples_per_epoch", None),
        predownload=cfg.dataset.get("predownload", 100_000),
        partition_algo=cfg.dataset.get("partition_algo", "orig"),
        num_canonical_nodes=cfg.dataset.get("num_canonical_nodes", 128),
        batch_size=batch_size,
        shuffle=cfg.dataset.get("shuffle", False),
        shuffle_algo=cfg.dataset.get("shuffle_algo", "py1s"),
        shuffle_seed=cfg.dataset.get("shuffle_seed", 9176),
    )

    mlm_probability = cfg.dataset.get("mlm_probability", None)
    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer,
        mlm=mlm_probability is not None,
        mlm_probability=mlm_probability,
    )

    eos_token_id = cfg.dataset.get("eos_token_id")
    bos_token_id = cfg.dataset.get("bos_token_id")
    if (eos_token_id is not None) or (bos_token_id is not None):
        # Note: Will raise an error if both are non-None
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=collate_fn,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
    )


# Helpful to test if your dataloader is working locally
# Run `python dataset.py --local_path [local] [--remote_path remote, optional]`
# and verify that batches are printed out.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2", help="the name of the tokenizer to use"
    )
    parser.add_argument(
        "--local_path",
        type=str,
        required=True,
        help="the path to the local copy of the dataset",
    )
    parser.add_argument(
        "--remote_path",
        type=str,
        default=None,
        help="the path to the remote copy to stream from (optional)",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="which split of the dataset to use"
    )
    parser.add_argument(
        "--max_length", type=int, default=32, help="max sequence length to test"
    )

    args = parser.parse_args()

    if args.remote_path is not None:
        print(
            f"Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}"  # noqa: E501
        )
    else:
        print(f"Reading {args.split} split from {args.local_path}")

    config = {
        "name": "text",
        "dataset": {
            "local": args.local_path,
            "remote": args.remote_path,
            "split": args.split,
            "shuffle": False,
            "max_length": args.max_length,
        },
        "drop_last": False,
        "num_workers": 4,
    }
    config = OmegaConf.create(config)
    device_batch_size = 2

    tokenizer_cfg = {
        "name": args.tokenizer,
        "kwargs": {"model_max_length": args.max_length},
    }
    tokenizer_cfg = OmegaConf.create(tokenizer_cfg)
    testing_tokenizer = build_tokenizer(tokenizer_cfg)

    loader = build_text_dataloader(config, testing_tokenizer, device_batch_size)
    testing_tokenizer = loader.dataset.tokenizer  # type: ignore
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print("\n")
        print("#" * 20, f"Batch {batch_ix}", "#" * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
            for sample_ix, token_sample in enumerate(batch["input_ids"]):
                print("-" * 20, f" Sample {sample_ix} ", "-" * 20)
                print(testing_tokenizer.decode(token_sample))
