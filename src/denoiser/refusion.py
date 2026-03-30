from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerateOutput

from src.denoiser.ar import AR, ARConfig
from src.denoiser.base import DenoiserInput, LossAndNllOutput
from src.denoiser.diffusion_config import DiffusionGenerationOutput

IGNORE_INDEX = -100
REFUSION_OFFICIAL_BOS_TOKEN = "<|beginoftext|>"
REFUSION_OFFICIAL_MASK_TOKEN = "<|mask|>"
REFUSION_SPECIAL_TOKEN_ATTRS = (
    "mask_token_id",
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
)


class ReFusionDynamicCache(DynamicCache):
    """Minimal cache extension needed by ReFusion decoding."""

    # Upstream ReFusion source:
    # - file: qwen3/diffusion_cache_utils.py
    # - symbol: DiffusionDynamicCache.full_update
    # - adaptation: none
    def full_update(
        self,
        new_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        del cache_kwargs
        for layer_idx, (key, value) in enumerate(new_kv):
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value], dim=-2
            )

    # Upstream ReFusion source:
    # - file: qwen3/diffusion_cache_utils.py
    # - symbol: DiffusionDynamicCache.select_partial
    # - adaptation: renamed locally to clarify that the selection acts on the sequence axis.
    def select_sequence_indices(self, indices: torch.Tensor) -> None:
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, indices, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, indices, :]

    def select_partial(self, indices: torch.Tensor) -> None:
        self.select_sequence_indices(indices)

    # Upstream ReFusion source:
    # - file: qwen3/diffusion_cache_utils.py
    # - symbol: DiffusionDynamicCache.batch_select_minibatch
    # - adaptation: accepts either an integer prefix length or explicit batch indices to fit this repo's cache helpers.
    def batch_select_minibatch(self, indices: Union[int, torch.Tensor]) -> None:
        if isinstance(indices, int):
            for layer_idx in range(len(self.key_cache)):
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:indices, ...]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:indices, ...]
            return
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats <= 1:
            return
        if hasattr(super(), "batch_repeat_interleave"):
            super().batch_repeat_interleave(repeats)
            return
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if hasattr(super(), "batch_select_indices"):
            super().batch_select_indices(indices)
            return
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class ReFusionGenerationConfig(GenerationConfig):
    """Generation config for ReFusion's slot-wise decoding loop."""

    def __init__(
        self,
        slot_size: int = 8,
        serial_num_blocks: int = 2,
        slot_threshold: float = 0.9,
        token_threshold: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if slot_size <= 0:
            raise ValueError(f"`slot_size` must be positive, got {slot_size}.")
        if serial_num_blocks <= 0:
            raise ValueError(
                f"`serial_num_blocks` must be positive, got {serial_num_blocks}."
            )
        self.slot_size = slot_size
        self.serial_num_blocks = serial_num_blocks
        self.slot_threshold = slot_threshold
        self.token_threshold = token_threshold
        self.use_cache = True


class ReFusionConfig(ARConfig):
    """Configuration class for ReFusion models."""

    model_type = "refusion"
    auto_map = {
        "AutoConfig": "diffusion.ReFusionConfig",
        "AutoModel": "diffusion.ReFusion",
        "AutoModelForCausalLM": "diffusion.ReFusion",
    }

    def __init__(
        self,
        slot_size_set: Sequence[int] = (4, 8, 16, 32),
        training_eps: float = 1e-3,
        ignore_index: int = IGNORE_INDEX,
        **kwargs,
    ):
        super().__init__(**kwargs)
        slot_size_set = tuple(int(size) for size in slot_size_set)
        if len(slot_size_set) == 0:
            raise ValueError("`slot_size_set` must not be empty.")
        if any(size <= 0 for size in slot_size_set):
            raise ValueError(f"`slot_size_set` must contain positive ints: {slot_size_set}")
        if not (0.0 < training_eps <= 1.0):
            raise ValueError(
                f"`training_eps` must be in (0, 1], got {training_eps}."
            )
        self.slot_size_set = list(slot_size_set)
        self.training_eps = float(training_eps)
        self.ignore_index = int(ignore_index)


class ReFusion(AR):
    """ReFusion denoiser adapted to this repository's denoiser/backbone split."""

    config_class = ReFusionConfig

    @staticmethod
    def _token_in_tokenizer_vocab(
        tokenizer: PreTrainedTokenizer,
        token: str,
    ) -> bool:
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            vocab = get_vocab()
            if token in vocab:
                return True
        get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
        if callable(get_added_vocab):
            return token in get_added_vocab()
        return False

    @classmethod
    def prepare_tokenizer_for_refusion(
        cls,
        tokenizer: Optional[PreTrainedTokenizer],
    ) -> Optional[PreTrainedTokenizer]:
        if tokenizer is None:
            return None

        special_tokens_to_add: dict[str, str] = {}
        bos_token = getattr(tokenizer, "bos_token", None)
        eos_token = getattr(tokenizer, "eos_token", None)
        if bos_token is None or bos_token == eos_token:
            if cls._token_in_tokenizer_vocab(tokenizer, REFUSION_OFFICIAL_BOS_TOKEN):
                tokenizer.bos_token = REFUSION_OFFICIAL_BOS_TOKEN
            else:
                special_tokens_to_add["bos_token"] = REFUSION_OFFICIAL_BOS_TOKEN

        if getattr(tokenizer, "mask_token", None) != REFUSION_OFFICIAL_MASK_TOKEN:
            if cls._token_in_tokenizer_vocab(tokenizer, REFUSION_OFFICIAL_MASK_TOKEN):
                tokenizer.mask_token = REFUSION_OFFICIAL_MASK_TOKEN
            else:
                special_tokens_to_add["mask_token"] = REFUSION_OFFICIAL_MASK_TOKEN

        if len(special_tokens_to_add) > 0:
            tokenizer.add_special_tokens(special_tokens_to_add)

        if getattr(tokenizer, "bos_token", None) is None:
            tokenizer.bos_token = tokenizer.eos_token
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    @staticmethod
    def _resolve_resizeable_backbone_module(backbone: Any) -> Any | None:
        for candidate in (backbone, getattr(backbone, "model", None)):
            if candidate is None:
                continue
            resize_fn = getattr(candidate, "resize_token_embeddings", None)
            config = getattr(candidate, "config", None)
            if callable(resize_fn) and config is not None:
                return candidate
        return None

    def sync_tokenizer_and_backbone(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer
        tokenizer = self.prepare_tokenizer_for_refusion(self.tokenizer)
        if tokenizer is None:
            return

        target_vocab_size = len(tokenizer)
        backbone_module = self._resolve_resizeable_backbone_module(self.backbone)
        if backbone_module is not None:
            current_vocab_size = getattr(backbone_module.config, "vocab_size", None)
            if (
                isinstance(current_vocab_size, int)
                and current_vocab_size != target_vocab_size
            ):
                backbone_module.resize_token_embeddings(target_vocab_size)
            backbone_module.config.vocab_size = target_vocab_size
            if hasattr(backbone_module, "vocab_size"):
                backbone_module.vocab_size = target_vocab_size
            for token_attr in REFUSION_SPECIAL_TOKEN_ATTRS:
                token_id = getattr(tokenizer, token_attr, None)
                if token_id is not None:
                    setattr(backbone_module.config, token_attr, int(token_id))
                    if hasattr(backbone_module, token_attr):
                        setattr(backbone_module, token_attr, int(token_id))

        self.config.vocab_size = target_vocab_size
        self.vocab_size = target_vocab_size
        for token_attr in REFUSION_SPECIAL_TOKEN_ATTRS:
            token_id = getattr(tokenizer, token_attr, None)
            if token_id is not None:
                setattr(self.config, token_attr, int(token_id))
                setattr(self, token_attr, int(token_id))

    # Upstream ReFusion source:
    # - file: generate.py
    # - symbol: add_gumbel_noise
    # - adaptation: none
    @staticmethod
    def add_gumbel_noise(
        logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def __init__(
        self,
        config: ReFusionConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super().__init__(config, tokenizer=tokenizer, **kwargs)
        self.sync_tokenizer_and_backbone(tokenizer=self.tokenizer)

    @staticmethod
    def _stack_batch_tensors(items: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        return torch.stack(items).to(device)

    @staticmethod
    def _get_prompt_lengths_from_context_mask(
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[int]:
        if context_mask.shape != attention_mask.shape:
            raise ValueError(
                "`context_mask` and `attention_mask` must have the same shape for ReFusion."
            )
        if ((context_mask != 0) & (context_mask != 1)).any():
            raise ValueError("ReFusion expects `context_mask` to be binary.")
        prompt_lengths = context_mask.sum(dim=1, dtype=torch.long)
        total_lengths = attention_mask.sum(dim=1, dtype=torch.long)
        for batch_idx in range(context_mask.shape[0]):
            prompt_len = int(prompt_lengths[batch_idx].item())
            total_len = int(total_lengths[batch_idx].item())
            if prompt_len > total_len:
                raise ValueError(
                    "ReFusion requires `context_mask` to stay within the attended prefix."
                )
            expected_prefix = torch.zeros(total_len, dtype=context_mask.dtype, device=context_mask.device)
            expected_prefix[:prompt_len] = 1
            if not torch.equal(context_mask[batch_idx, :total_len], expected_prefix):
                raise ValueError(
                    "ReFusion training requires `context_mask` to mark a contiguous prefix."
                )
            if context_mask[batch_idx, total_len:].any():
                raise ValueError(
                    "ReFusion training requires `context_mask` to be zero on padded positions."
                )
        return prompt_lengths.tolist()

    # Upstream ReFusion source:
    # - file: qwen3/modeling_qwen3_refusion.py
    # - symbol: forward_process
    # - adaptation: validates that this repo's `context_mask` is an explicit contiguous
    #   prefix before deriving prompt lengths, requires callers to provide prompt
    #   structure explicitly instead of silently defaulting to an all-answer batch,
    #   drops empty-answer rows like upstream,
    #   and stores the extra supervision metadata in `DenoiserInput` fields rather
    #   than mutating the backbone `forward`.
    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
    ) -> DenoiserInput:
        del t, past_key_values
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            raise ValueError(
                "ReFusion training requires explicit prompt structure via "
                "`context_mask`; omitting it would silently treat the full attended "
                "sequence as answer tokens."
            )
        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
        if torch.is_floating_point(context_mask):
            context_mask = context_mask.to(torch.int)

        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        prompt_lengths = self._get_prompt_lengths_from_context_mask(
            context_mask=context_mask,
            attention_mask=attention_mask,
        )
        total_lengths = attention_mask.sum(dim=1).tolist()
        raw_input_ids = input_ids.tolist()

        kept_attention_masks = []
        kept_context_masks = []
        processed_inputs = []
        processed_labels = []
        processed_masked_indices = []
        processed_p_masks = []
        processed_answer_lengths = []
        processed_position_ids = []

        for batch_idx in range(batch_size):
            prompt_len = min(int(prompt_lengths[batch_idx]), int(total_lengths[batch_idx]))
            total_len = int(total_lengths[batch_idx])
            pad_len = seq_length - total_len
            prompt_tokens = raw_input_ids[batch_idx][:prompt_len]
            answer_tokens = raw_input_ids[batch_idx][prompt_len:total_len]

            if len(answer_tokens) == 0:
                continue

            slot_size = random.choice(self.config.slot_size_set)
            answer_slots = [
                answer_tokens[offset : offset + slot_size]
                for offset in range(0, len(answer_tokens), slot_size)
            ]
            answer_positions = list(range(prompt_len, prompt_len + len(answer_tokens)))
            answer_position_slots = [
                answer_positions[offset : offset + slot_size]
                for offset in range(0, len(answer_positions), slot_size)
            ]
            p_mask = (1 - self.config.training_eps) * random.random() + self.config.training_eps
            slot_mask = [random.random() < p_mask for _ in answer_slots]
            unmasked_indices = [
                idx for idx, masked in enumerate(slot_mask) if not masked
            ]
            masked_indices = [idx for idx, masked in enumerate(slot_mask) if masked]
            random.shuffle(unmasked_indices)

            final_answer_tokens: list[int] = []
            final_answer_labels: list[int] = []
            final_answer_masked_indices: list[bool] = []
            final_answer_position_ids: list[int] = []

            for slot_idx in unmasked_indices:
                slot_content = answer_slots[slot_idx]
                final_answer_tokens.extend(slot_content)
                final_answer_labels.extend(
                    slot_content[1:] + [self.config.ignore_index]
                )
                final_answer_masked_indices.extend([False] * len(slot_content))
                final_answer_position_ids.extend(answer_position_slots[slot_idx])

            for slot_idx in masked_indices:
                slot_content = answer_slots[slot_idx]
                final_answer_tokens.extend([self.mask_token_id] * len(slot_content))
                final_answer_labels.extend(slot_content)
                final_answer_masked_indices.extend([True] * len(slot_content))
                final_answer_position_ids.extend(answer_position_slots[slot_idx])

            final_input = (
                prompt_tokens + final_answer_tokens + raw_input_ids[batch_idx][total_len:]
            )
            final_label = (
                [self.config.ignore_index] * len(prompt_tokens)
                + final_answer_labels
                + [self.config.ignore_index] * pad_len
            )
            final_masked = (
                [False] * len(prompt_tokens)
                + final_answer_masked_indices
                + [False] * pad_len
            )
            final_position_ids = (
                list(range(len(prompt_tokens)))
                + final_answer_position_ids
                + list(range(total_len, seq_length))
            )

            processed_inputs.append(torch.tensor(final_input))
            processed_labels.append(torch.tensor(final_label))
            processed_masked_indices.append(torch.tensor(final_masked))
            processed_p_masks.append(
                torch.full((seq_length,), float(p_mask), dtype=torch.float32)
            )
            processed_answer_lengths.append(
                torch.full((seq_length,), float(len(answer_tokens)), dtype=torch.float32)
            )
            processed_position_ids.append(torch.tensor(final_position_ids))
            kept_attention_masks.append(attention_mask[batch_idx])
            kept_context_masks.append(context_mask[batch_idx])

        if len(processed_inputs) == 0:
            raise ValueError(
                "ReFusion training batch has no answer tokens after applying the "
                "contiguous-prefix `context_mask`."
            )

        labels = self._stack_batch_tensors(processed_labels, device)
        masked_indices = self._stack_batch_tensors(
            processed_masked_indices, device
        ).bool()
        token_loss_mask = (labels != self.config.ignore_index).to(attention_mask.dtype)

        return DenoiserInput(
            xt=self._stack_batch_tensors(processed_inputs, device),
            x0=labels,
            attention_mask=self._stack_batch_tensors(kept_attention_masks, device),
            context_mask=self._stack_batch_tensors(kept_context_masks, device),
            valid_tokens=masked_indices.to(torch.float32),
            tokens_mask=token_loss_mask,
            t=self._stack_batch_tensors(processed_p_masks, device),
            alpha_t=self._stack_batch_tensors(processed_answer_lengths, device),
            backbone_kwargs={
                "position_ids": self._stack_batch_tensors(processed_position_ids, device),
            },
        )

    def _prepare_inputs_inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context: Optional[torch.LongTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        cache: Optional[Dict[str, Any]] = None,
        return_updated_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        **backbone_kwargs: Any,
    ) -> Tuple[DenoiserInput, Dict[str, Any]]:
        del context, context_mask, return_updated_cache
        assert input_ids is not None, "ReFusion inference requires `input_ids`."
        cache = {} if cache is None else cache
        past_key_values = cache.pop("past_key_values", ReFusionDynamicCache())
        cache_length = self._get_past_key_values_seq_length(past_key_values)
        full_seq_length = cache_length + input_ids.shape[-1]
        if full_seq_length > self.config.length:
            overflow = full_seq_length - self.config.length
            past_key_values = self._crop_kv_cache_left(past_key_values, overflow)
            cache_length = self._get_past_key_values_seq_length(past_key_values)
        if position_ids is None:
            position_ids = torch.arange(
                cache_length,
                cache_length + input_ids.shape[-1],
                device=input_ids.device,
            )[None, :]
        return (
            DenoiserInput(
                xt=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                backbone_kwargs={"position_ids": position_ids} | backbone_kwargs,
            ),
            cache,
        )

    # Upstream ReFusion source:
    # - file: qwen3/modeling_qwen3_refusion.py
    # - symbol: Qwen3ForCausalLM.forward
    # - adaptation: reused the hybrid AR+masked loss formula on top of this repo's `DenoiserOutput` fields instead of computing the loss inside the Qwen wrapper.
    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        del kwargs
        labels = denoiser_inputs.x0
        safe_labels = labels.masked_fill(labels == self.config.ignore_index, 0)
        log_p_theta = torch.gather(
            model_output,
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)

        valid_label_mask = denoiser_inputs.tokens_mask.bool()
        masked_indices = denoiser_inputs.valid_tokens.bool()
        p_mask = denoiser_inputs.t.clamp_min(1e-8)
        answer_lengths = denoiser_inputs.alpha_t.clamp_min(1.0)

        ar_mask = valid_label_mask & ~masked_indices
        mdm_mask = valid_label_mask & masked_indices

        ar_loss = log_p_theta.new_zeros(())
        if ar_mask.any():
            ar_loss = -log_p_theta[ar_mask].mean()

        mdm_loss = log_p_theta.new_zeros(())
        if mdm_mask.any():
            mdm_token_loss = -log_p_theta[mdm_mask] / p_mask[mdm_mask]
            mdm_loss = (mdm_token_loss / answer_lengths[mdm_mask]).sum() / labels.shape[0]

        nlls = torch.zeros_like(log_p_theta)
        nlls[ar_mask] = -log_p_theta[ar_mask]
        nlls[mdm_mask] = (
            -log_p_theta[mdm_mask]
            / p_mask[mdm_mask]
            / answer_lengths[mdm_mask]
        )

        return LossAndNllOutput(
            loss=ar_loss + mdm_loss,
            nlls=nlls,
            other_loss_terms={
                "ar_loss": ar_loss,
                "mdm_loss": mdm_loss,
                "masked_tokens": masked_indices.to(torch.int),
            },
        )

    def _backbone_generate_step(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values: ReFusionDynamicCache,
        use_cache: bool,
    ):
        return self.backbone(
            input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    @staticmethod
    def _cache_from_outputs(
        outputs: Any,
        fallback_cache: ReFusionDynamicCache,
    ) -> ReFusionDynamicCache:
        if getattr(outputs, "past_key_values", None) is None:
            return fallback_cache
        return outputs.past_key_values

    @staticmethod
    def _resolve_eos_token_id(
        generation_config: ReFusionGenerationConfig,
        tokenizer: Optional[PreTrainedTokenizer],
        model_eos_token_id: Optional[int],
    ) -> Optional[int]:
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if len(eos_token_id) > 0 else None
        if eos_token_id is not None:
            return int(eos_token_id)
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            return int(tokenizer.eos_token_id)
        if model_eos_token_id is not None:
            return int(model_eos_token_id)
        return None

    @staticmethod
    def _apply_stopping_criteria(
        sequences: torch.LongTensor,
        stopping_criteria: StoppingCriteriaList | None,
        prompt_length: int = 0,
        pad_token_id: int = 0,
    ) -> torch.LongTensor:
        if stopping_criteria is None or len(stopping_criteria) == 0:
            return sequences
        if sequences.ndim != 2 or sequences.shape[1] == 0:
            return sequences

        stop_positions: list[int] = []
        search_start = max(prompt_length + 1, 1)
        total_length = sequences.shape[1]
        for batch_idx in range(sequences.shape[0]):
            stop_pos = total_length
            for end in range(search_start, total_length + 1):
                should_stop = stopping_criteria(
                    input_ids=sequences[batch_idx : batch_idx + 1, :end],
                    scores=None,
                )
                if isinstance(should_stop, torch.Tensor):
                    stop_flag = bool(should_stop.reshape(-1)[0].item())
                else:
                    stop_flag = bool(should_stop)
                if stop_flag:
                    stop_pos = end
                    break
            stop_positions.append(stop_pos)

        if all(stop_pos == total_length for stop_pos in stop_positions):
            return sequences
        max_stop = max(stop_positions)
        truncated = sequences.new_full(
            (sequences.shape[0], max_stop),
            fill_value=pad_token_id,
        )
        for batch_idx, stop_pos in enumerate(stop_positions):
            truncated[batch_idx, :stop_pos] = sequences[batch_idx, :stop_pos]
        return truncated

    # Upstream ReFusion source:
    # - file: generate.py
    # - symbol: generate_refusion
    # - adaptation: preserves the upstream decode schedule exactly and rejects
    #   generation lengths that would violate the upstream reshape invariant,
    #   instead of silently padding to a different serial-block partition.
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[ReFusionGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        return_dict_in_generate: Optional[bool] = None,
        disable_pbar: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        del logits_processor, batch_size, disable_pbar, kwargs
        if inputs is None:
            raise ValueError("ReFusion generation requires `inputs`.")
        if inputs.shape[0] != 1:
            raise ValueError("ReFusion generation currently supports batch size 1 only.")
        if (inputs == self.mask_token_id).any():
            raise ValueError("ReFusion generation does not support infilling inputs.")
        if generation_config is None:
            generation_config = getattr(self, "generation_config", None)
        if generation_config is None:
            generation_config = ReFusionGenerationConfig()
        if not isinstance(generation_config, ReFusionGenerationConfig):
            generation_config = ReFusionGenerationConfig(**generation_config.to_dict())
        if device is None:
            device = inputs.device

        prompt = inputs.to(device)
        prompt_len = prompt.shape[1]
        if max_new_tokens is None:
            if generation_config.max_new_tokens is not None:
                max_new_tokens = generation_config.max_new_tokens
            elif max_length is not None:
                max_new_tokens = max_length - prompt_len
            else:
                raise ValueError("ReFusion generation requires `max_new_tokens` or `max_length`.")
        requested_new_tokens = int(max_new_tokens)
        if requested_new_tokens <= 0:
            if return_dict_in_generate:
                return DiffusionGenerationOutput(
                    sequences=prompt,
                    parallelism_factor=0.0,
                    non_ar_tokens_per_step=0.0,
                )
            return prompt

        slot_size = generation_config.slot_size
        serial_num_blocks = generation_config.serial_num_blocks
        required_multiple = slot_size * serial_num_blocks
        gen_pad_len = (slot_size - (requested_new_tokens % slot_size)) % slot_size
        padded_new_tokens = requested_new_tokens + gen_pad_len
        if padded_new_tokens % required_multiple != 0:
            raise ValueError(
                "Official ReFusion generation requires `max_new_tokens` "
                "(after slot-size padding) to be divisible by "
                "`slot_size * serial_num_blocks`."
            )
        slot_mask_id = self.mask_token_id
        temperature = float(generation_config.temperature)
        slot_threshold = generation_config.slot_threshold
        token_threshold = generation_config.token_threshold
        eos_token_id = self._resolve_eos_token_id(
            generation_config,
            tokenizer,
            getattr(self, "eos_token_id", None),
        )

        gen_x = torch.full(
            (1, padded_new_tokens),
            slot_mask_id,
            dtype=torch.long,
            device=device,
        )
        prompt_pos_ids = torch.arange(prompt_len, device=device, dtype=torch.long)[
            None, :
        ]
        gen_pos_ids = torch.arange(
            prompt_len, prompt_len + padded_new_tokens, device=device, dtype=torch.long
        )[None, :]

        cur_x = prompt.clone()
        cur_pos = prompt_pos_ids.clone()
        past_key_values = ReFusionDynamicCache()
        sum_tpf = 0.0
        forward_count = 0
        non_ar_tokens_committed = 0.0
        non_ar_step_count = 0
        eos_flag = False
        block_length = padded_new_tokens // serial_num_blocks

        for serial_block_idx in range(serial_num_blocks):
            block_start = serial_block_idx * block_length
            block_end = (serial_block_idx + 1) * block_length
            cur_gen_x = gen_x[:, block_start:block_end]
            cur_gen_pos_ids = gen_pos_ids[:, block_start:block_end]
            cur_gen_blocks_x = cur_gen_x.reshape(1, -1, slot_size)
            cur_gen_blocks_pos_ids = cur_gen_pos_ids.reshape(1, -1, slot_size)

            while cur_gen_blocks_x.numel() > 0:
                cur_gen_blocks_x = cur_gen_blocks_x.reshape(1, -1, slot_size)
                cur_gen_blocks_pos_ids = cur_gen_blocks_pos_ids.reshape(1, -1, slot_size)
                flat_gen_blocks_x = cur_gen_blocks_x.view(1, -1)
                flat_gen_blocks_pos_ids = cur_gen_blocks_pos_ids.view(1, -1)
                prefix_block_tag = False

                if past_key_values.get_seq_length() == 0:
                    model_inputs = torch.cat((cur_x, flat_gen_blocks_x), dim=1)
                    model_position_ids = torch.cat((cur_pos, flat_gen_blocks_pos_ids), dim=1)
                else:
                    model_inputs = flat_gen_blocks_x
                    model_position_ids = flat_gen_blocks_pos_ids
                outputs = self._backbone_generate_step(
                    input_ids=model_inputs,
                    position_ids=model_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                gen_logits = outputs.logits[:, -flat_gen_blocks_x.shape[1] :, :]
                past_key_values = self._cache_from_outputs(outputs, past_key_values)
                past_key_values.crop(cur_x.shape[1])

                logits_with_noise = self.add_gumbel_noise(
                    gen_logits, temperature=temperature
                )
                x0_gen = torch.argmax(logits_with_noise, dim=-1)
                x0_gen_blocks = x0_gen.view(1, -1, slot_size)
                p_softmax = F.softmax(gen_logits, dim=-1)
                x0_p_softmax = torch.gather(
                    p_softmax,
                    dim=-1,
                    index=x0_gen.unsqueeze(-1),
                ).squeeze(-1)
                x0_p_softmax_blocks = x0_p_softmax.view(1, -1, slot_size)
                block_confidence_softmax = x0_p_softmax_blocks[:, :, 0]
                is_confident_block = block_confidence_softmax > slot_threshold
                counts_block = int(is_confident_block.sum(dim=1).item())
                topk_indices_relative = is_confident_block[0].nonzero(as_tuple=True)[0]
                if counts_block <= 0:
                    counts_block = 1
                    _, topk_indices_relative = torch.topk(
                        block_confidence_softmax.squeeze(0), k=1
                    )
                topk_indices_relative, _ = torch.sort(topk_indices_relative)

                chosen_gen_blocks = x0_gen_blocks[0, topk_indices_relative, :]
                chosen_position_ids = cur_gen_blocks_pos_ids[0, topk_indices_relative, :]
                chosen_p_softmax_blocks = x0_p_softmax_blocks[
                    0, topk_indices_relative, :
                ].clone()

                outputs = self._backbone_generate_step(
                    input_ids=chosen_gen_blocks.reshape(1, -1),
                    position_ids=chosen_position_ids.reshape(1, -1),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                ar_logits = outputs.logits
                ar_logits = torch.cat([ar_logits[:, :1], ar_logits[:, :-1]], dim=1)
                ar_probs = F.softmax(ar_logits, dim=-1)
                ar_x0_probs = torch.gather(
                    ar_probs,
                    dim=-1,
                    index=chosen_gen_blocks.reshape(1, -1).unsqueeze(-1),
                ).squeeze(-1)
                ar_x0_prob_blocks = ar_x0_probs.reshape(-1, slot_size)
                chosen_p_softmax_blocks[:, 1:] = ar_x0_prob_blocks[:, 1:]

                prob_mask = chosen_p_softmax_blocks > token_threshold
                prob_mask[:, 0] = True
                tag_blocks = torch.cumprod(prob_mask.int(), dim=-1)
                tag_tokens = torch.cumprod(prob_mask.int().reshape(1, -1), dim=-1)
                prefix_len = int(tag_tokens.sum().item())
                flat_chosen_gen_blocks = chosen_gen_blocks.reshape(1, -1)
                confident_prefix_tokens = flat_chosen_gen_blocks[:, :prefix_len]

                remain_indices: list[int] = []
                indices_to_remove: set[int] = set()
                eos_found_flag = False
                eos_token_pos = 0
                if prefix_len > 0:
                    if eos_token_id is not None:
                        is_eos_in_prefix = confident_prefix_tokens.squeeze(0) == eos_token_id
                        eos_found_flag = bool(torch.any(is_eos_in_prefix))
                    else:
                        is_eos_in_prefix = torch.zeros(
                            prefix_len, dtype=torch.bool, device=device
                        )
                    if eos_found_flag:
                        first_eos_pos_tensor = int(torch.argmax(is_eos_in_prefix.int()).item())
                        eos_block_pos = first_eos_pos_tensor // slot_size + 1
                        eos_token_pos = first_eos_pos_tensor % slot_size
                        eos_block = int(topk_indices_relative[eos_block_pos - 1].item())
                        remain_indices.extend(
                            topk_indices_relative[:eos_block_pos].tolist()
                        )
                        topk_indices_relative = torch.empty(
                            0, dtype=torch.long, device=device
                        )
                        eos_flag = True
                        indices_to_remove.update(
                            range(eos_block, cur_gen_blocks_x.shape[1])
                        )
                    elif (prefix_len // slot_size) > 0:
                        num_prefix_blocks = prefix_len // slot_size
                        remain_indices.extend(
                            topk_indices_relative[:num_prefix_blocks].tolist()
                        )
                        topk_indices_relative = topk_indices_relative[num_prefix_blocks:]
                        tag_blocks = tag_blocks[num_prefix_blocks:]

                    if len(remain_indices) > 0:
                        indices_to_remove.update(remain_indices)
                        token_indices = []
                        for remain_idx, block_idx in enumerate(remain_indices):
                            start_index = block_idx * slot_size
                            current_block_len = slot_size
                            if eos_found_flag and remain_idx == len(remain_indices) - 1:
                                current_block_len = eos_token_pos + 1
                            token_indices.append(
                                torch.arange(
                                    start_index,
                                    start_index + current_block_len,
                                    device=device,
                                    dtype=torch.long,
                                )
                            )
                        full_token_indices = torch.cat(token_indices)
                        cur_x = torch.cat(
                            (cur_x, x0_gen[:, full_token_indices]),
                            dim=1,
                        )
                        cur_pos = torch.cat(
                            (cur_pos, flat_gen_blocks_pos_ids[:, full_token_indices]),
                            dim=1,
                        )
                        past_key_values = self._cache_from_outputs(
                            outputs, past_key_values
                        )
                        past_key_values.crop(cur_x.shape[1])
                        prefix_block_tag = True
                        non_ar_tokens_committed += float(len(remain_indices))
                        non_ar_step_count += 1
                        sum_tpf += slot_size * len(remain_indices) / 2
                        forward_count += 1

                if prefix_block_tag:
                    keep_mask = torch.ones(
                        cur_gen_blocks_x.shape[1], dtype=torch.bool, device=device
                    )
                    if len(indices_to_remove) > 0:
                        keep_mask[list(indices_to_remove)] = False
                    cur_gen_blocks_x = cur_gen_blocks_x[:, keep_mask, :]
                    cur_gen_blocks_pos_ids = cur_gen_blocks_pos_ids[:, keep_mask, :]
                    continue

                past_key_values = self._cache_from_outputs(outputs, past_key_values)
                past_key_values.crop(cur_x.shape[1])
                indices_to_remove = set(topk_indices_relative.tolist())
                current_speculative_blocks = chosen_gen_blocks.clone()
                accepted_prefix_len = 0
                eos_found_in_loop = False
                first_eos_block_idx = 0

                if past_key_values.get_seq_length() > 0 and counts_block > 1:
                    past_key_values.batch_repeat_interleave(counts_block)

                loop_iter = -1
                for loop_iter in range(slot_size):
                    if not torch.any(tag_blocks == 0):
                        break
                    input_tokens = current_speculative_blocks[:, accepted_prefix_len:]
                    input_pos = chosen_position_ids[:, accepted_prefix_len:]
                    current_tags = tag_blocks[:, accepted_prefix_len:]
                    masked_input_tokens = torch.where(
                        current_tags.bool(),
                        input_tokens,
                        torch.full_like(input_tokens, slot_mask_id),
                    )

                    draft_len = past_key_values.get_seq_length()
                    draft_outputs = self._backbone_generate_step(
                        input_ids=masked_input_tokens,
                        position_ids=input_pos,
                        past_key_values=past_key_values,
                        use_cache=False,
                    )
                    past_key_values.crop(draft_len)
                    draft_logits = draft_outputs.logits
                    proposed_tokens = torch.argmax(draft_logits, dim=-1)
                    input_tokens = torch.where(
                        current_tags.bool(),
                        input_tokens,
                        proposed_tokens,
                    )
                    current_speculative_blocks[:, accepted_prefix_len:] = input_tokens

                    verify_outputs = self._backbone_generate_step(
                        input_ids=input_tokens,
                        position_ids=input_pos,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    verify_logits = verify_outputs.logits
                    verify_logits = torch.cat(
                        [verify_logits[:, :1], verify_logits[:, :-1]], dim=1
                    )
                    verify_probs = F.softmax(verify_logits, dim=-1)
                    gathered_probs = torch.gather(
                        verify_probs,
                        -1,
                        input_tokens.unsqueeze(-1),
                    ).squeeze(-1)
                    prob_mask = gathered_probs > token_threshold
                    update_tag_blocks = F.pad(
                        tag_blocks[:, accepted_prefix_len:],
                        (1, 0),
                        value=1,
                    )[:, :-1]
                    prob_mask[update_tag_blocks == 1] = True
                    tag_blocks[:, accepted_prefix_len:] = torch.cumprod(
                        prob_mask.int(), dim=-1
                    )

                    newly_verified_mask = tag_blocks[:, accepted_prefix_len:] == 1
                    if eos_token_id is not None:
                        is_eos_in_new = (
                            current_speculative_blocks[:, accepted_prefix_len:]
                            == eos_token_id
                        ) & newly_verified_mask
                    else:
                        is_eos_in_new = torch.zeros_like(newly_verified_mask)

                    if torch.any(is_eos_in_new):
                        eos_found_in_loop = True
                        first_eos_block_idx = int(
                            torch.where(torch.any(is_eos_in_new, dim=1))[0][0].item()
                        )
                        current_speculative_blocks = current_speculative_blocks[
                            : first_eos_block_idx + 1
                        ]
                        tag_blocks = tag_blocks[: first_eos_block_idx + 1]
                        tag_blocks[first_eos_block_idx] = 1
                        chosen_position_ids = chosen_position_ids[
                            : first_eos_block_idx + 1
                        ]
                        topk_indices_relative = topk_indices_relative[
                            : first_eos_block_idx + 1
                        ]
                        verify_cache = self._cache_from_outputs(
                            verify_outputs, past_key_values
                        )
                        verify_cache.batch_select_minibatch(first_eos_block_idx + 1)
                        verify_outputs.past_key_values = verify_cache

                    current_tags = tag_blocks[:, accepted_prefix_len:]
                    len_per_block = torch.sum(current_tags, dim=1)
                    newly_accepted_len = int(torch.min(len_per_block).item())
                    if newly_accepted_len > 0:
                        if torch.any(tag_blocks == 0):
                            accepted_prefix_len += newly_accepted_len - 1
                        else:
                            accepted_prefix_len += newly_accepted_len
                        past_key_values = self._cache_from_outputs(
                            verify_outputs, past_key_values
                        )
                        past_key_values.crop(cur_x.shape[1] + accepted_prefix_len)

                denom = max(loop_iter * 2 + 2, 1)
                sum_tpf += (slot_size * counts_block) / denom
                forward_count += 1

                ar_kv_cache = tuple(
                    (
                        layer_past[0][:, :, -slot_size:, :],
                        layer_past[1][:, :, -slot_size:, :],
                    )
                    for layer_past in past_key_values
                )
                past_key_values.crop(cur_x.shape[1])
                past_key_values.batch_select_indices(
                    torch.tensor([0], dtype=torch.long, device=device)
                )

                eos_mask = current_speculative_blocks == eos_token_id if eos_token_id is not None else torch.zeros_like(current_speculative_blocks, dtype=torch.bool)
                keep_mask = (
                    torch.cumsum(eos_mask.flatten().int(), dim=-1)
                    - eos_mask.flatten().int()
                ) == 0
                kept_tokens = current_speculative_blocks.flatten()[keep_mask].reshape(1, -1)
                kept_pos_ids = chosen_position_ids.flatten()[keep_mask].reshape(1, -1)

                if kept_tokens.numel() > 0 and len(ar_kv_cache) > 0:
                    new_past = []
                    for key, value in ar_kv_cache:
                        num_heads = key.shape[1]
                        head_dim = key.shape[3]
                        flat_key = key.permute(1, 0, 2, 3).reshape(
                            1, num_heads, -1, head_dim
                        )
                        flat_value = value.permute(1, 0, 2, 3).reshape(
                            1, num_heads, -1, head_dim
                        )
                        new_past.append(
                            (
                                flat_key[:, :, keep_mask, :],
                                flat_value[:, :, keep_mask, :],
                            )
                        )
                    past_key_values.full_update(tuple(new_past))

                cur_x = torch.cat((cur_x, kept_tokens), dim=1)
                cur_pos = torch.cat((cur_pos, kept_pos_ids), dim=1)
                non_ar_tokens_committed += float(current_speculative_blocks.shape[0])
                non_ar_step_count += 1

                if eos_found_in_loop:
                    indices_to_remove.update(
                        range(first_eos_block_idx, cur_gen_blocks_x.shape[1])
                    )
                    eos_flag = True

                keep_block_mask = torch.ones(
                    cur_gen_blocks_x.shape[1], dtype=torch.bool, device=device
                )
                if len(indices_to_remove) > 0:
                    keep_block_mask[list(indices_to_remove)] = False
                cur_gen_blocks_x = cur_gen_blocks_x[:, keep_block_mask, :]
                cur_gen_blocks_pos_ids = cur_gen_blocks_pos_ids[:, keep_block_mask, :]

            if eos_flag:
                break

        _, reorder_indices = torch.sort(cur_pos, dim=-1)
        sequences = torch.gather(cur_x, dim=-1, index=reorder_indices)
        sequences = self._apply_stopping_criteria(
            sequences=sequences,
            stopping_criteria=stopping_criteria,
            prompt_length=prompt_len,
            pad_token_id=self.pad_token_id,
        )
        parallelism_factor = sum_tpf / forward_count if forward_count > 0 else 0.0
        non_ar_tokens_per_step = (
            non_ar_tokens_committed / non_ar_step_count
            if non_ar_step_count > 0
            else 0.0
        )
        if return_dict_in_generate:
            return DiffusionGenerationOutput(
                sequences=sequences,
                parallelism_factor=parallelism_factor,
                non_ar_tokens_per_step=non_ar_tokens_per_step,
            )
        return sequences
