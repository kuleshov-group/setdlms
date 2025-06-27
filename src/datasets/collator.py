from collections.abc import Callable
from typing import Any

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


# TODO: For AR init diffusion models, implement attn_mask annealing? (see DiffuLlama)
class DenoisingCollator:
    """Custom collator that samples a random t value for each example in the batch."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        global_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        padding: bool = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: str = "pt",
        predict_padding: bool = False,
        restricted_t_range: tuple[float, float] | None = None,
        sampling_eps: float = 0.05,
        antithetic_sampling: bool = False,
        block_size: int | None = None,
        base_collator: Callable | None = None,
    ):
        """
        Parameters:
            tokenizer (PreTrainedTokenizerBase): Tokenizer used in base collator
            global_batch_size (int): Used for sampling t.
            rank (int): Used for sampling t.
            world_size (int): Used for sampling t.
            padding (bool; default: True): Whether to pad the sequences
            max_length: (Optional: int): Maximum length of the sequences.
            pad_to_multiple_of: (Optional: int): if specified,
                pad sequences to a multiple of this value.
            return_tensors: (str; default: "pt"): Format of the returned tensors.
            predict_padding (bool; default: False): Whether to predict padding tokens.
            restricted_t_range (Optional: tuple[min: float, max: float]): If specified,
                sampling of timestep (t) sampling is restricted to [min, max] range,
                as opposed to [0, 1].
            sampling_eps (float; default: 0.05): Effective minimum sampled t.
            antithetic_sampling (bool; default: False): Whether to use antithetic
                sampling.
            block_size (int): Specified when using block-denoising;
                if specified, sampled t will have shape (batch_size, max_length),
                where within each block the same sampled_t will be repeated.
                If not specified, sampled t will have shape (batch_size,)
            base_collator (Optional: Callable): The base collator that is being wrapped.
                If None, defaults to transformers.DataCollatorWithPadding.
        """
        if base_collator is not None:
            self.base_collate_fn = base_collator
        else:
            self.base_collate_fn = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=padding,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        self.padding_side = tokenizer.padding_side
        self.predict_padding = predict_padding
        self.restricted_t_range = restricted_t_range
        self.sampling_eps = sampling_eps
        self.antithetic_sampling = antithetic_sampling
        self.global_batch_size = global_batch_size
        self.max_length = max_length
        self.block_size = block_size
        # TODO: Confirm that this works on multi-node
        self._rank = rank
        self._world_size = world_size

    def _sample_t(self, global_batch_size, batch_size, t_index, device):
        num_blocks = self.max_length // self.block_size if self.block_size else 1
        if self.block_size is not None and self.block_size > 0:
            _eps_t = torch.rand(batch_size, num_blocks, device=device)
        else:
            _eps_t = torch.rand(batch_size, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(
                start=t_index[0] * num_blocks,
                end=t_index[1] * num_blocks,
                device=device,
            ) / (global_batch_size * num_blocks)
            offset = offset.view(batch_size, num_blocks)
            _eps_t = (_eps_t / (global_batch_size * num_blocks) + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.restricted_t_range is not None:
            low, high = self.restricted_t_range
            t = (low - high) * t + high
        if self.block_size is not None and self.block_size > 0:
            t = t[..., torch.randperm(t.shape[-1])]
            return t.repeat_interleave(self.block_size, dim=1)
        return t

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        context_mask = [f.pop("context_mask", None) for f in features]
        batch = self.base_collate_fn(features)
        batch_size = batch["input_ids"].shape[0]
        global_batch_size = self._world_size * batch_size
        t_index = (  # index t's for the given device (used for antithetic_sampling
            self._rank * batch_size,
            min((self._rank + 1) * batch_size, global_batch_size),
        )
        t = self._sample_t(
            global_batch_size=global_batch_size,
            batch_size=batch_size,
            t_index=t_index,
            device=batch["input_ids"].device,
        )
        if all([c is not None for c in context_mask]):
            context_mask = torch.nn.utils.rnn.pad_sequence(
                context_mask,  # type: ignore
                batch_first=True,
            )[..., : self.max_length]
            context_mask = torch.nn.functional.pad(
                context_mask,
                (0, self.max_length - context_mask.shape[-1])
                if self.padding_side == "right"
                else (self.max_length - context_mask.shape[-1], 0),
            )
            batch.update({"context_mask": context_mask})
        batch.update({"t": t})

        # Override the attention mask to attend to all tokens (including [PAD])
        if self.predict_padding:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
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
