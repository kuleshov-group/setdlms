import re

import torch
from transformers.generation import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from scripts.utils import maybe_add_missing_special_tokens


class HydraCompatibleLogitsProcessorList(LogitsProcessorList):
    """Hydra-compatible version of LogitsProcessorList.

    Initialized using dict[str, LogitsProcessor], which in turn initializes
    the parent object as: LogitsProcessorList(list(dict.values())).
    """

    def __init__(self, logits_processor_dict: dict[str, LogitsProcessor]):
        super().__init__(list(logits_processor_dict.values()))


class HydraCompatibleStoppingCriteriaList(StoppingCriteriaList):
    """Hydra-compatible version of StoppingCriteriaList.

    Initialized using dict[str, StoppingCriteria], which in turn initializes
    the parent object as: StoppingCriteriaList(list(dict.values())).
    """

    def __init__(self, stopping_criteria_dict: dict[str, StoppingCriteria]):
        super().__init__(list(stopping_criteria_dict.values()))


class RegexStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, pattern, num_matches: int = 1):
        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        self.pattern = pattern
        self.num_matches = num_matches

    def __call__(
        self, input_ids: torch.LongTensor, scores: None | torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor:
        if input_ids.numel() == 0:
            return torch.tensor([False], device=input_ids.device, dtype=torch.bool)
        matches = []
        if input_ids.ndim > 1:
            text = self.tokenizer.batch_decode(input_ids)
        else:
            text = [self.tokenizer.decode(input_ids)]
        for i in range(len(text)):
            matches_i = re.findall(self.pattern, text[i])
            matches_i = [match for match in matches_i if self.tokenizer.mask_token not in match]
            matches.append(len(matches_i) >= self.num_matches)
        return torch.tensor(matches, device=input_ids.device, dtype=torch.bool)

class RepeatingTokenStoppingCriteria(StoppingCriteria):
    """
    Stop if the *suffix* contains a contiguous run of the same token id
    of length >= min_run_length, excluding mask tokens.

    Efficient: checks only the last min_run_length tokens each call.
    """

    def __init__(self, tokenizer, min_run_length: int = 20):
        if min_run_length < 2:
            raise ValueError("min_run_length must be >= 2")
        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        self.min_run_length = min_run_length
        self.mask_token_id = getattr(self.tokenizer, "mask_token_id", None)

    def __call__(
        self, input_ids: torch.LongTensor, scores: None | torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor:
        if input_ids.numel() == 0:
            return torch.tensor([False], device=input_ids.device, dtype=torch.bool)

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)  # [B=1, T]

        B, T = input_ids.shape
        if B > 1:
            raise NotImplementedError("Only batch size 1 is supported")

        last_non_mask_token = input_ids.ne(self.mask_token_id).nonzero(as_tuple=False)[-1, -1].item()
        input_ids = input_ids[:, :last_non_mask_token + 1]

        B, T = input_ids.shape
        L = self.min_run_length

        if T < L:
            return torch.zeros(B, device=input_ids.device, dtype=torch.bool)

        window = input_ids[:, -L:]          # [B, L]
        last = input_ids[:, -1].unsqueeze(1)  # [B, 1]

        # All last L tokens must equal the last token.
        return (window == last).all(dim=1)