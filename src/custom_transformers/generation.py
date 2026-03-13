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

class EntropyEosStoppingCriteria(StoppingCriteria):
    """
    Hugging Face stopping criteria replicating:
      - stop if entropy on last 256 tokens < threshold
      - if var_length: stop if a second EOS is present
    Note: cannot truncate inside generate(); store truncation info for post-processing.
    """

    def __init__(
        self,
        tokenizer,
        entropy_threshold: float = 4.0,
        block_size: int = 128,
        var_length: bool = False,
        num_matches: int = 1,
    ):
        super().__init__()
        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = self.tokenizer.cls_token
        self.eos_token_id = tokenizer.eos_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.entropy_threshold = entropy_threshold
        self.block_size = block_size
        self.var_length = var_length
        self.num_matches = num_matches
        
        # For optional post-processing:
        # one entry per batch item, filled with either None or an int truncate index.
        self.truncate_idx = None
        self.stop_reason = None

    def _ensure_state(self, batch_size: int):
        if self.truncate_idx is None or len(self.truncate_idx) != batch_size:
            self.truncate_idx = [None] * batch_size
            self.stop_reason = [None] * batch_size

    def _compute_entropy(self, x):
        # exclude mask tokens
        x = x[x != self.mask_token_id]
        _, counts = torch.unique(x, return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum()
        return entropy
  
    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        input_ids: (batch, seq_len)
        scores:    (batch, vocab) for the last step (may be None depending on config)
        """
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        self._ensure_state(bsz)
        if self.var_length:
            eos_mask = (input_ids == self.eos_token_id)  # (B, L)
            eos_counts = eos_mask.sum(dim=1)

            if torch.any(eos_counts >= self.num_matches):
                for i in range(bsz):
                    if eos_counts[i].item() >= self.num_matches:
                        eos_positions = torch.nonzero(
                            eos_mask[i], as_tuple=False
                        ).squeeze(-1)

                        last_eos_pos = int(eos_positions[self.num_matches - 1].item())

                        has_mask_before = (
                            input_ids[i, :last_eos_pos] == self.mask_token_id
                        ).any()

                        if not has_mask_before:
                            stop_any = True
                            self.truncate_idx[i] = min(last_eos_pos + 1, seq_len)
                            self.stop_reason[i] = "last_eos"
                            return True
    
        if seq_len < self.block_size:
            return False


        last_block = input_ids[:, -self.block_size:]
        entropy = self._compute_entropy(last_block)

        if torch.is_tensor(entropy):
            entropy_val = float(entropy.detach().mean().cpu())
        else:
            entropy_val = float(entropy)

        stop_any = False

        # Criterion: low entropy => stop
        if entropy_val < self.entropy_threshold:
            stop_any = True
            if self.var_length:
                # In your code: truncate_idx = x.shape[1] - 256
                for i in range(bsz):
                    self.truncate_idx[i] = max(seq_len - self.block_size, 0)
                    self.stop_reason[i] = "low_entropy"
        return stop_any