from typing import Any, Dict

import torch.distributed as dist
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from datasets import load_dataset


def load_streaming_dataset(
    name: str,
    split: str,
    tokenizer: PreTrainedTokenizer,
    shuffle_buffer_size: int,
    seed: int,
    max_length: int,
    take_size: int = 0,
    padding: bool = False,
    **_: Dict[str, Any],
):
    ds = load_dataset(name, split=split, streaming=True)

    if dist.is_available() and dist.is_initialized():
        num_shards = dist.get_world_size()
        index = dist.get_rank()
        ds = ds.shard(num_shards=num_shards, index=index)

    if shuffle_buffer_size:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

    if take_size > 0:
        ds = ds.take(take_size)

    columns_to_keep = ["input_ids", "attention_mask"]
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

    ds = ds.map(
        lambda example: tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=padding,
            return_tensors=None,
        ),
        remove_columns=columns_to_remove,
    )

    return ds


class StreamingHFDataset(IterableDataset):
    def __init__(
        self,
        name: str,
        split: str,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        shuffle_buffer_size: int = 1000,
        seed: int = 42,
        take_size: int = -1,
        padding: bool = False,
        **_: Dict[str, Any],
    ):
        self.dataset = load_streaming_dataset(
            name=name,
            split=split,
            tokenizer=tokenizer,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            take_size=take_size,
            max_length=max_length,
            padding=padding,
            **_,
        )

    def __iter__(self):
        return iter(self.dataset)
