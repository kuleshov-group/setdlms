from typing import Any, Dict, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from datasets import Dataset, DatasetDict, load_from_disk
from src.utils import fsspec_exists

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def load_preprocessed_dataset(
    dataset_path: str,
    keep_in_memory: bool | None = None,
    storage_options: dict | None = None,
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
    return load_from_disk(
        dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options
    ).with_format("torch")
