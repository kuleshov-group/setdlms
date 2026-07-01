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

    def reset(self):
        for criterion in self:
            reset = getattr(criterion, "reset", None)
            if callable(reset):
                reset()


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
            matches_i = [
                match for match in matches_i if self.tokenizer.mask_token not in match
            ]
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

        last_non_mask_token = (
            input_ids.ne(self.mask_token_id).nonzero(as_tuple=False)[-1, -1].item()
        )
        input_ids = input_ids[:, : last_non_mask_token + 1]

        B, T = input_ids.shape
        L = self.min_run_length

        if T < L:
            return torch.zeros(B, device=input_ids.device, dtype=torch.bool)

        window = input_ids[:, -L:]  # [B, L]
        last = input_ids[:, -1].unsqueeze(1)  # [B, 1]

        # All last L tokens must equal the last token.
        return (window == last).all(dim=1)


class EntropyEosStoppingCriteria(StoppingCriteria):
    """
    Hugging Face stopping criteria replicating:
      - stop if entropy on last 256 tokens < threshold
      - if var_length: stop if enough EOS tokens are present
    Note: cannot truncate inside generate(); store truncation info for post-processing.
    """

    def __init__(
        self,
        tokenizer,
        entropy_threshold: float = 4.0,
        block_size: int = 128,
        var_length: bool = False,
        num_matches: int = 1,
        ignore_prompt_eos: bool = False,
        prompt_length: int = 1,
        confidence_threshold: float | None = None,
        confidence_window: int = 128,
        confidence_min_tokens: int = 32,
        confidence_patience: int = 1,
        low_entropy_truncation: str = "legacy",
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
        self.ignore_prompt_eos = ignore_prompt_eos
        self.prompt_length = max(int(prompt_length), 0)
        self.confidence_threshold = confidence_threshold
        # Kept for config compatibility; confidence stopping now uses consecutive
        # sampled-token confidences instead of a rolling-window mean.
        self.confidence_window = max(int(confidence_window), 1)
        self.confidence_min_tokens = max(int(confidence_min_tokens), 1)
        self.confidence_patience = max(int(confidence_patience), 1)
        valid_low_entropy_truncation = {"legacy", "stop_only", "disabled"}
        if low_entropy_truncation not in valid_low_entropy_truncation:
            raise ValueError(
                "low_entropy_truncation must be one of "
                f"{sorted(valid_low_entropy_truncation)}, got "
                f"{low_entropy_truncation!r}"
            )
        self.low_entropy_truncation = low_entropy_truncation

        # For optional post-processing:
        # one entry per batch item, filled with either None or an int truncate index.
        self.truncate_idx = None
        self.stop_reason = None
        self.confidence_strikes = None
        self.confidence_seen_positions = None

    def reset(self):
        self.truncate_idx = None
        self.stop_reason = None
        self.confidence_strikes = None
        self.confidence_seen_positions = None

    def _ensure_state(self, batch_size: int):
        if self.truncate_idx is None or len(self.truncate_idx) != batch_size:
            self.truncate_idx = [None] * batch_size
            self.stop_reason = [None] * batch_size
            self.confidence_strikes = [0] * batch_size
            self.confidence_seen_positions = [set() for _ in range(batch_size)]
        else:
            if (
                self.confidence_strikes is None
                or len(self.confidence_strikes) != batch_size
            ):
                self.confidence_strikes = [0] * batch_size
            if (
                self.confidence_seen_positions is None
                or len(self.confidence_seen_positions) != batch_size
            ):
                self.confidence_seen_positions = [set() for _ in range(batch_size)]

    def _compute_entropy(self, x):
        # exclude mask tokens
        x = x[x != self.mask_token_id]
        _, counts = torch.unique(x, return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum()
        return entropy

    def _stop_offset(self, seq_len: int) -> int:
        return min(self.prompt_length, seq_len) if self.ignore_prompt_eos else 0

    def _generated_tokens_seen(self, seq_len: int) -> int:
        return max(seq_len - self._stop_offset(seq_len), 0)

    def _has_min_generated_tokens(self, seq_len: int) -> bool:
        return self._generated_tokens_seen(seq_len) >= self.confidence_min_tokens

    def _min_truncate_idx(self, seq_len: int) -> int:
        return min(self._stop_offset(seq_len) + self.confidence_min_tokens, seq_len)

    def _confidence_stop(self, input_ids, token_confidence, seq_len):
        if self.confidence_threshold is None or token_confidence is None:
            return False
        if token_confidence.ndim == 1:
            token_confidence = token_confidence.unsqueeze(0)
        token_confidence = token_confidence[:, :seq_len]

        stop_any = False
        confidence_offset = self._stop_offset(seq_len)
        for i in range(input_ids.shape[0]):
            confidence_i = token_confidence[i]
            valid = torch.isfinite(confidence_i)
            valid &= input_ids[i, :seq_len] != self.mask_token_id
            if confidence_offset > 0:
                valid[:confidence_offset] = False
            positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                self.confidence_strikes[i] = 0
                continue

            seen_positions = self.confidence_seen_positions[i]
            new_positions = [
                int(position.item())
                for position in positions
                if int(position.item()) not in seen_positions
            ]
            if not new_positions:
                continue

            if len(seen_positions) < self.confidence_min_tokens:
                num_to_skip = min(
                    self.confidence_min_tokens - len(seen_positions),
                    len(new_positions),
                )
                seen_positions.update(new_positions[:num_to_skip])
                new_positions = new_positions[num_to_skip:]
                self.confidence_strikes[i] = 0
                if not new_positions:
                    continue

            for pos in new_positions:
                seen_positions.add(pos)
                conf = float(confidence_i[pos].detach().cpu())
                if conf < self.confidence_threshold:
                    self.confidence_strikes[i] += 1
                else:
                    self.confidence_strikes[i] = 0

                if self.confidence_strikes[i] >= self.confidence_patience:
                    stop_any = True
                    self.truncate_idx[i] = min(pos + 1, seq_len)
                    self.stop_reason[i] = "low_confidence"
                    break
        return stop_any

    @torch.no_grad()
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        input_ids: (batch, seq_len)
        scores:    (batch, vocab) for the last step (may be None depending on config)
        """
        bsz, seq_len = input_ids.shape
        self._ensure_state(bsz)
        if self.var_length:
            eos_offset = self._stop_offset(seq_len)
            eos_input_ids = input_ids[:, eos_offset:]
            eos_mask = eos_input_ids == self.eos_token_id  # (B, L - eos_offset)
            eos_counts = eos_mask.sum(dim=1)
            min_truncate_idx = self._min_truncate_idx(seq_len)

            if torch.any(eos_counts >= self.num_matches):
                for i in range(bsz):
                    if eos_counts[i].item() >= self.num_matches:
                        eos_positions = (
                            torch.nonzero(eos_mask[i], as_tuple=False).squeeze(-1)
                            + eos_offset
                        )
                        eos_positions = eos_positions[eos_positions + 1 >= min_truncate_idx]
                        if eos_positions.numel() < self.num_matches:
                            continue

                        last_eos_pos = int(eos_positions[self.num_matches - 1].item())

                        has_mask_before = (
                            input_ids[i, eos_offset:last_eos_pos] == self.mask_token_id
                        ).any()

                        if not has_mask_before:
                            self.truncate_idx[i] = min(last_eos_pos + 1, seq_len)
                            self.stop_reason[i] = "last_eos"
                            return True

        token_confidence = kwargs.get("token_confidence")
        if self._confidence_stop(input_ids, token_confidence, seq_len):
            return True

        if seq_len < self.block_size:
            return False
        if not self._has_min_generated_tokens(seq_len):
            return False

        last_block = input_ids[:, -self.block_size :]
        entropy = self._compute_entropy(last_block)

        if torch.is_tensor(entropy):
            entropy_val = float(entropy.detach().mean().cpu())
        else:
            entropy_val = float(entropy)

        stop_any = False

        # Criterion: low entropy => stop
        if entropy_val < self.entropy_threshold:
            if self.low_entropy_truncation == "disabled":
                return False

            stop_any = True
            if self.var_length:
                for i in range(bsz):
                    self.stop_reason[i] = "low_entropy"
                    if self.low_entropy_truncation == "legacy":
                        min_truncate_idx = self._min_truncate_idx(seq_len)
                        self.truncate_idx[i] = min(
                            max(seq_len - self.block_size, min_truncate_idx), seq_len
                        )
                    elif self.low_entropy_truncation == "stop_only":
                        self.truncate_idx[i] = seq_len
        return stop_any
