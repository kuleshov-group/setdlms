from typing import Any, Dict

from transformers import PreTrainedTokenizer

from datasets import load_dataset


def load_streaming_dataset(
    name: str,
    split: str,
    shuffle_buffer_size: int,
    seed: int,
    take_size: int,
):
    ds = load_dataset(name, split=split, streaming=True)

    if shuffle_buffer_size:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

    if take_size:
        ds = ds.take(take_size)

    return ds


class StreamingHFDataset:
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
        self.raw_dataset = load_streaming_dataset(
            name=name,
            split=split,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            take_size=take_size,
        )
        self.tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __iter__(self):
        for example in self.raw_dataset:
            yield self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.max_length,
                padding=self.padding,
                return_tensors=None,
            )
