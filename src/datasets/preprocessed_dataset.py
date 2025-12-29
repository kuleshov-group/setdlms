from typing import Any, Dict, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from datasets import Dataset, DatasetDict, load_from_disk
from src.utils import fsspec_exists
import torch

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def load_preprocessed_dataset(
    dataset_path: str,
    keep_in_memory: bool | None = None,
    storage_options: dict | None = None,
    limit_size: int | None = None,
    inject_context_mask: bool = False,
    token_to_split: str | None = None,
    split_offset: int | None = None,
    tokenizer: Tokenizer | None = None,
    # Unused tokenizer arg (compat. with other dataset loading functions/classes)
    **_: Dict[str, Any],
) -> Dataset | DatasetDict:
    """Load a preprocessed dataset from disk.

    Accepts (unused) tokenizer argument for compatibility with other dataset loading
    functions / classes.

    Args:
        dataset_path (str): Path to the preprocessed dataset.
        keep_in_memory (`bool`, defaults to `None`):
            Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting
            `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero.
        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the file-system backend, if any.

    Returns:
        Union[Dataset, DatasetDict]: The loaded dataset.
    """
    if not fsspec_exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
    dataset = load_from_disk(
        dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options
    ).with_format("torch")

    if limit_size is not None:
        dataset = dataset.select(range(limit_size))
    if inject_context_mask:
        tokens_to_split = tokenizer.encode(token_to_split)[0]
        map_fn = lambda x: {
            "context_mask": (torch.arange(x["input_ids"].shape[-1]) <= (x["input_ids"] == tokens_to_split).nonzero()[-1]+split_offset).to(torch.int)}
        dataset = dataset.map(map_fn)
    return dataset