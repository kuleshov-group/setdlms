from collections.abc import Callable
from typing import Any

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


class DenoisingCollator:
    """Custom collator that samples a random t value for each example in the batch."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: bool = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: str = "pt",
        # Use to bias sampling in certain region
        restricted_t_range: tuple[float, float] | None = None,
        sampling_eps: float = 0.05,
        antithetic_sampling: bool = False,
    ):
        self.base_collate_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        self.restricted_t_range = restricted_t_range
        self.sampling_eps = sampling_eps
        self.antithetic_sampling = antithetic_sampling

    def _sample_t(self, batch_size, device):
        _eps_t = torch.rand(batch_size, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(batch_size, device=device) / batch_size
            _eps_t = (_eps_t / batch_size + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.restricted_t_range is not None:
            low, high = self.restricted_t_range
            t = (low - high) * t + high
        return t

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base_collate_fn(features)
        t = self._sample_t(
            batch_size=batch["input_ids"].shape[0], device=batch["input_ids"].device
        )
        batch["t"] = t
        return batch


class ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(
        self,
        base_collator: Callable,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None,
    ):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError(
                "Must supply a value for either eos_token_id or bos_token_id,"
                " but got None for both."
            )
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError(
                "Cannot use *both* EOS and BOS tokens for detecting sequence"
                " boundaries. Please supply `eos_token_id` if sequences end with an EOS"
                " token, or use `bos_token_id` if sequences start with a BOS token."
            )
        if eos_token_id is None:
            self.split_token_id = bos_token_id
            self.bos_mode = True
        else:
            self.split_token_id = eos_token_id
            self.bos_mode = False

    def get_sequence_id_from_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        assert self.split_token_id is not None
        is_separator = torch.eq(batch["input_ids"], self.split_token_id)
        cumulative_sep = torch.cumsum(is_separator, dim=1).to(batch["input_ids"].dtype)
        # If separator token is bos, we're already done
        if self.bos_mode:
            return cumulative_sep

        # If separator token is eos, right shift 1 space
        left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
        return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base_collator(features)
        batch["sequence_id"] = self.get_sequence_id_from_batch(batch)
        return batch
