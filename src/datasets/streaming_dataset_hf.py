from typing import Any, Dict
import logging
import torch.distributed as dist
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_streaming_dataset(
    name: str,
    split: str,
    tokenizer: PreTrainedTokenizer,
    shuffle_buffer_size: int,
    seed: int,
    max_length: int,
    take_size: int = 0,
    padding: bool = False,
    config: str | None = None,
    **_: Dict[str, Any],
):
    logger.info(f"Loading dataset {name} ({split=}, {config=})")
    ds = load_dataset(path=name, name=config, split=split, streaming=True)

    if dist.is_available() and dist.is_initialized():
        num_shards = dist.get_world_size()
        index = dist.get_rank()
        if hasattr(ds, "num_shards") and index >= ds.num_shards:
            raise ValueError(
                f"Rank {index} assignment to shard {index+1} in dataset {name} ({split=}, {config=}) "
                f"is invalid because there are only {ds.num_shards} shards available"
            )
        logger.info(f"Assigning shard {index+1} in dataset {name} ({split=}, {config=}) to rank {index}")
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
        config: str | None = None,
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
            config=config,
            **_,
        )

    def __iter__(self):
        return iter(self.dataset)