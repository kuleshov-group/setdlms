from functools import partial
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList, PreTrainedTokenizer, StoppingCriteriaList
from transformers.cache_utils import Cache, DynamicCache
from tqdm.auto import tqdm

try:
    from torch.nn.attention.flex_attention import and_masks, create_block_mask
except ImportError:
    and_masks, create_block_mask = None, None

from src.denoiser.base import DenoiserInput, LossAndNllOutput
from src.denoiser.bd3lm import BD3LM, BD3LMConfig
from src.denoiser.diffusion_config import (
    DiffusionGenerationConfig,
    DiffusionGenerationOutput,
    SetDiffusionGenerationConfig,
    create_attn_mask,
)


class SetDLMStaticCache:
    """Append-only KV cache with stable backing storage for compiled SetDLM eval.

    DynamicCache returns tensors allocated inside the compiled/captured backbone. In
    reduce-overhead mode those graph-owned tensors must be cloned before the next
    replay. This cache instead copies each layer update into persistent tensors and
    returns active views with the same logical shape/order as DynamicCache.
    """

    def __init__(self, max_cache_len: int):
        self.max_cache_len = int(max_cache_len)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._seq_length = 0
        self._write_start = 0
        self._write_end = 0
        self._logical_write_start = 0
        self._prepared_physical_write_start: int | None = None
        self._prepared_return_length: int | None = None
        self._logical_bucket_write = False
        self._return_length: int | None = None
        self._pending_physical_write_start: int | None = None
        self._pending_logical_write_start: int | None = None

    @staticmethod
    def _as_int(value: Any) -> int:
        return int(value.item()) if hasattr(value, "item") else int(value)

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        read_length = (
            self._return_length if self._return_length is not None else self._seq_length
        )
        return (
            self.key_cache[layer_idx][..., :read_length, :],
            self.value_cache[layer_idx][..., :read_length, :],
        )

    def _allocate_layer(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_shape = (
            key_states.shape[0],
            key_states.shape[1],
            self.max_cache_len,
            key_states.shape[-1],
        )
        key_cache = torch.zeros(
            cache_shape,
            dtype=key_states.dtype,
            device=key_states.device,
        )
        value_cache = torch.zeros(
            cache_shape,
            dtype=value_states.dtype,
            device=value_states.device,
        )
        torch._dynamo.mark_static_address(key_cache)
        torch._dynamo.mark_static_address(value_cache)
        return key_cache, value_cache

    def initialize(
        self,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        cache_shape = (batch_size, num_heads, self.max_cache_len, head_dim)
        for _ in range(num_layers):
            key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
            value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
            torch._dynamo.mark_static_address(key_cache)
            torch._dynamo.mark_static_address(value_cache)
            self.key_cache.append(key_cache)
            self.value_cache.append(value_cache)

    def _ensure_layer(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        while len(self.key_cache) <= layer_idx:
            key_cache, value_cache = self._allocate_layer(key_states, value_states)
            self.key_cache.append(key_cache)
            self.value_cache.append(value_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        del layer_idx
        return self._seq_length

    def prepare_write(self, physical_write_start: int | None = None) -> None:
        self._prepared_physical_write_start = (
            None
            if physical_write_start is None
            else self._as_int(physical_write_start)
        )
        self._prepared_return_length = None
        self._logical_bucket_write = False

    def prepare_logical_write(self, return_length: int | None = None) -> None:
        self._prepared_physical_write_start = None
        self._prepared_return_length = (
            None if return_length is None else self._as_int(return_length)
        )
        self._logical_bucket_write = True

    def crop(self, max_length: int) -> None:
        max_length = self._as_int(max_length)
        if max_length < 0:
            max_length = max(self._seq_length + max_length, 0)
        if (
            self._pending_physical_write_start is not None
            and self._pending_logical_write_start is not None
            and self._pending_physical_write_start != self._pending_logical_write_start
            and max_length > self._pending_logical_write_start
        ):
            keep_len = max_length - self._pending_logical_write_start
            src_start = self._pending_physical_write_start
            src_end = src_start + keep_len
            dst_start = self._pending_logical_write_start
            dst_end = dst_start + keep_len
            for layer_idx in range(len(self.key_cache)):
                self.key_cache[layer_idx][..., dst_start:dst_end, :].copy_(
                    self.key_cache[layer_idx][..., src_start:src_end, :]
                )
                self.value_cache[layer_idx][..., dst_start:dst_end, :].copy_(
                    self.value_cache[layer_idx][..., src_start:src_end, :]
                )
        self._seq_length = min(max_length, self._seq_length)
        self._return_length = None
        self._prepared_physical_write_start = None
        self._prepared_return_length = None
        self._logical_bucket_write = False
        self._pending_physical_write_start = None
        self._pending_logical_write_start = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        self._ensure_layer(layer_idx, key_states, value_states)
        incoming_len = key_states.shape[-2]
        if layer_idx == 0:
            self._logical_write_start = self._seq_length
            logical_bucket_write = self._logical_bucket_write
            if logical_bucket_write:
                self._write_start = self._logical_write_start
            else:
                self._write_start = (
                    self._prepared_physical_write_start
                    if self._prepared_physical_write_start is not None
                    else self._logical_write_start
                )
            self._write_end = self._write_start + incoming_len
            return_length = max(
                self._logical_write_start + incoming_len,
                self._write_end,
            )
            if self._prepared_return_length is not None:
                return_length = max(return_length, self._prepared_return_length)
            if return_length > self.max_cache_len:
                raise ValueError(
                    "SetDLMStaticCache capacity exceeded: "
                    f"{return_length} > {self.max_cache_len}"
                )
            self._seq_length = self._logical_write_start + incoming_len
            self._return_length = return_length
            if logical_bucket_write:
                self._pending_physical_write_start = None
                self._pending_logical_write_start = None
            else:
                self._pending_physical_write_start = self._write_start
                self._pending_logical_write_start = self._logical_write_start
        self.key_cache[layer_idx][..., self._write_start : self._write_end, :].copy_(
            key_states
        )
        self.value_cache[layer_idx][
            ..., self._write_start : self._write_end, :
        ].copy_(value_states)
        return self[layer_idx]



class SetDLM(BD3LM):
    """Denoiser class for SetDLM models."""

    _KV_CACHE_POSITION_IDS_KEY = "_setdlm_kv_cache_position_ids"

    def __init__(
        self,
        config: BD3LMConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, **kwargs)
        self.block_size = config.block_size
        if config.attn_backend == "flex_attention":
            self.static_attention_mask = None
            self.encoder_static_attention_mask = None
        self._create_static_mask()
        self._warned_compile_stable_decode_unbucketed = False
        self._setdlm_fast_mask_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    @staticmethod
    def _backbone_cache_kwargs(cache: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v
            for k, v in cache.items()
            if k != SetDLM._KV_CACHE_POSITION_IDS_KEY
        }

    @staticmethod
    def _select_kv_cache_positions(
        past_key_values: Any,
        keep_indices: torch.LongTensor,
    ) -> Any:
        if past_key_values is None:
            return past_key_values
        assert hasattr(past_key_values, "key_cache") and hasattr(
            past_key_values, "value_cache"
        ), "DynamicCache-like structure not found"
        key_cache = getattr(past_key_values, "key_cache")
        value_cache = getattr(past_key_values, "value_cache")
        for i in range(len(past_key_values)):
            k = key_cache[i]
            v = value_cache[i]
            if k is None or v is None:
                continue
            layer_keep_indices = keep_indices.to(device=k.device)
            key_cache[i] = k.index_select(dim=-2, index=layer_keep_indices)
            value_cache[i] = v.index_select(dim=-2, index=layer_keep_indices)
        return past_key_values

    # noinspection PyUnusedLocal
    @staticmethod
    def _block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        seq_length: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del b, h

        # Indicate whether token belongs to xt or x0:
        xt_flag_q = (q_idx >= seq_length).bool()
        xt_flag_kv = (kv_idx >= seq_length).bool()

        q_idx = q_idx % seq_length
        kv_idx = kv_idx % seq_length

        # **1. Offset Causal Mask **
        offset_causal = (q_idx > kv_idx) & ~xt_flag_kv & xt_flag_q

        # **2. Diagonal Mask **
        diagonal = (q_idx == kv_idx) & (xt_flag_q == xt_flag_kv)

        # **3. Causal Mask **
        causal = (q_idx >= kv_idx) & ~xt_flag_kv & ~xt_flag_q

        # **4. Combine Masks **
        return diagonal | offset_causal | causal

    # noinspection PyUnusedLocal
    @staticmethod
    def _block_mask_eso(
        b,
        h,
        q_idx,
        kv_idx,
        seq_length: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del b, h

        # Indicate whether token belongs to xt or x0:
        xt_flag_q = (q_idx >= seq_length).bool()
        xt_flag_kv = (kv_idx >= seq_length).bool()

        q_idx = q_idx % seq_length
        kv_idx = kv_idx % seq_length

        # **1. Offset Causal Mask **
        offset_causal = (q_idx > kv_idx) & ~xt_flag_kv & xt_flag_q

        # **2. Causal Mask **
        diagonal = (q_idx >= kv_idx) & (xt_flag_q == xt_flag_kv)

        # **3. Causal Mask **
        causal = (q_idx >= kv_idx) & ~xt_flag_kv & ~xt_flag_q

        # **4. Combine Masks **
        return diagonal | offset_causal | causal

    def _create_static_mask(self) -> None:
        if self.config.attn_backend == "flex_attention":
            seq_length = self.config.length

            def mask_mod(b, h, q_idx, kv_idx):
                return self._block_mask(b, h, q_idx, kv_idx, seq_length=seq_length)

            self.static_attention_mask = create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=self.config.length * 2,
                KV_LEN=self.config.length * 2,
            )
        elif self.config.attn_backend == "sdpa":
            if getattr(self.config, "inefficient_training", False):
                mask = self._block_mask_eso(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.length * 2)[:, None],
                    kv_idx=torch.arange(self.config.length * 2)[None, :],
                    seq_length=self.config.length,
                )
            else:
                mask = self._block_mask(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.length * 2)[:, None],
                    kv_idx=torch.arange(self.config.length * 2)[None, :],
                    seq_length=self.config.length,
                )
            self.register_buffer("static_attention_mask", mask)
        else:
            raise ValueError("Unknown attention backend")

    def _new_generation_cache(
        self, batch_size: int, device: torch.device | str
    ) -> Dict[str, Any]:
        if getattr(self, "_setdlm_static_compile_cache", False) or getattr(
            self, "_setdlm_fast_inference", False
        ):
            cache = SetDLMStaticCache(self.config.length)
            backbone = getattr(self.backbone, "_orig_mod", self.backbone)
            if all(
                hasattr(backbone, attr)
                for attr in ("blocks", "n_heads", "vocab_embed")
            ):
                embedding = getattr(backbone.vocab_embed, "embedding", None)
                if embedding is not None:
                    hidden_size = int(embedding.shape[-1])
                    num_heads = int(backbone.n_heads)
                    cache.initialize(
                        batch_size=batch_size,
                        num_layers=len(backbone.blocks),
                        num_heads=num_heads,
                        head_dim=hidden_size // num_heads,
                        device=device,
                        dtype=embedding.dtype,
                    )
            return {"past_key_values": cache}
        del batch_size, device
        return {"past_key_values": DynamicCache()}

    @staticmethod
    def _clone_dynamic_cache_tensors(cache: Cache | None) -> Cache | None:
        if cache is None or not hasattr(cache, "key_cache"):
            return cache
        for cache_list_name in ("key_cache", "value_cache"):
            cache_list = getattr(cache, cache_list_name, None)
            if cache_list is None:
                continue
            for idx, tensor in enumerate(cache_list):
                if isinstance(tensor, torch.Tensor):
                    cache_list[idx] = tensor.clone()
        return cache

    def _should_clone_compile_cache(self) -> bool:
        return (
            hasattr(self.backbone, "_orig_mod")
            and getattr(self, "_setdlm_clone_compile_cache", False)
            and not getattr(self, "_setdlm_static_compile_cache", False)
        )

    def _clone_compile_cache_if_needed(self, cache: Dict[str, Any] | None) -> None:
        if self._should_clone_compile_cache() and cache is not None:
            self._clone_dynamic_cache_tensors(cache.get("past_key_values"))

    def _backbone_forward(
        self,
        denoiser_inputs: DenoiserInput,
        **backbone_kwargs: Any,
    ):
        compiled_backbone = hasattr(self.backbone, "_orig_mod")
        clone_compile_cache = getattr(self, "_setdlm_clone_compile_cache", False)
        static_compile_cache = getattr(self, "_setdlm_static_compile_cache", False)
        if compiled_backbone and (clone_compile_cache or static_compile_cache):
            mark_step_begin = getattr(
                getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
            )
            if mark_step_begin is not None:
                mark_step_begin()
        backbone_output = super()._backbone_forward(denoiser_inputs, **backbone_kwargs)
        return_updated_cache = denoiser_inputs.backbone_kwargs.get(
            "return_updated_cache", False
        ) or backbone_kwargs.get("return_updated_cache", False)
        if self._should_clone_compile_cache() and not return_updated_cache:
            self._clone_dynamic_cache_tensors(
                getattr(backbone_output, "past_key_values", None)
            )
        return backbone_output

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        input_length = denoiser_inputs.x0.shape[1]
        model_output = model_output[:, input_length : input_length * 2, ...]
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=denoiser_inputs.x0[:, :, None],  # .repeat(1, num_repetitions, 1)
        ).squeeze(-1)
        if getattr(self.config, "keep_clean_bos", False) and not self.training:
            log_p_theta = log_p_theta[:, 1:]
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask[:, 1:]
            denoiser_inputs.x0 = denoiser_inputs.x0[:, 1:]
            denoiser_inputs.attention_mask = denoiser_inputs.attention_mask[:, 1:]
            denoiser_inputs.alpha_t_prime = denoiser_inputs.alpha_t_prime[:, 1:]
            denoiser_inputs.alpha_t = denoiser_inputs.alpha_t[:, 1:]
        loss = -log_p_theta
        if not self.training:
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask * (
                denoiser_inputs.x0 != self.pad_token_id
            )
        inefficient_eval = getattr(self.config, "inefficient_training", False)
        if not self.training and inefficient_eval and self.config.block_size > 1:
            coeff = denoiser_inputs.alpha_t_prime / (1 - denoiser_inputs.alpha_t)
            coeff = torch.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
            seq_len = denoiser_inputs.x0.shape[1]
            masked_indices = denoiser_inputs.xt[:, -seq_len:] == self.mask_token_id
            nlls = log_p_theta * coeff * masked_indices
        else:
            nlls = loss * denoiser_inputs.tokens_mask
            if getattr(self.config, "inefficient_training", False):
                nlls *= denoiser_inputs.alpha_t_prime != 0.0

        # Compute per-batch counts and losses to avoid division by zero
        count = denoiser_inputs.tokens_mask.sum(dim=-1)  # Per-batch counts
        batch_nll = nlls.sum(dim=-1)  # Per-batch losses

        # Avoid division by zero: if count is 0, set token_nll to 0
        token_nll = torch.where(
            count > 0, batch_nll / count, torch.zeros_like(batch_nll)
        ).mean()

        permutation_order = denoiser_inputs.backbone_kwargs.get("permutation_order")
        other_loss_terms = {}
        if permutation_order is not None:
            other_loss_terms["permutation_order"] = permutation_order
            other_loss_terms["attention_mask"] = denoiser_inputs.attention_mask
            other_loss_terms["log_p_theta"] = -log_p_theta * denoiser_inputs.tokens_mask
        return LossAndNllOutput(
            loss=token_nll,
            nlls=nlls,
            other_loss_terms=other_loss_terms,
        )  # type: ignore

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)

        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
            context_mask = context_mask.to(torch.int)

        if t is None:
            t = torch.rand(
                input_ids.shape[0],
                input_ids.shape[1] // self.config.block_size,
                device=input_ids.device,
            ).repeat_interleave(self.config.block_size, dim=-1)
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]

        # permute input ids (where context_mask > 0 and attention_mask > 0)
        noise_mask = context_mask | ~(attention_mask.bool())
        permute_flag = noise_mask != 1
        if getattr(self.config, "keep_clean_bos", False):
            permute_flag[:, 0] = False
        perm_indices = None
        xt = input_ids.clone()
        evaluate_nll_flag = getattr(self.config, "inefficient_training", False)
        if not evaluate_nll_flag:
            xt = torch.where(
                (attention_mask == 1) & (context_mask == 0), self.mask_token_id, xt
            )
        else:
            xt = self._sample_q_xt(
                x0=input_ids,
                alpha_t=alpha_t,
                mask=noise_mask,
            )
            if getattr(self.config, "inefficient_training", False):
                xt = torch.where(alpha_t_prime != 0.0, self.mask_token_id, xt)

        batch_size, seq_len = input_ids.shape
        num_repetitions = 2
        if permute_flag.any():
            with torch.no_grad():
                perm_indices = self.noise_schedule.sample_permutation_order(
                    t,
                    permute_flag,
                    (
                        self.config.block_size
                        if self.training
                        else self.config.eval_block_size
                    ),
                    masked_tokens=(
                        (xt == self.mask_token_id) if evaluate_nll_flag else None
                    ),
                )
        else:
            perm_indices = torch.arange(seq_len, device=input_ids.device)[None, :]

        if (
            self.training
            and getattr(self.config, "setdlm_eos_last_in_target_order", False)
            and getattr(self.config, "eos_token_id", None) is not None
        ):
            perm_indices = self._move_target_eos_to_end_of_permutation(
                perm_indices=perm_indices,
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
                eos_token_id=int(self.config.eos_token_id),
            )

        alpha_t = torch.gather(alpha_t, dim=-1, index=perm_indices)
        alpha_t_prime = torch.gather(alpha_t_prime, dim=-1, index=perm_indices)
        if self.config.attn_backend == "sdpa":
            decoder_attention_mask = (
                self.static_attention_mask[None, ...]
                & attention_mask.repeat(1, num_repetitions)[:, None, :]
                & attention_mask.repeat(1, num_repetitions)[..., None]
            )
        elif self.config.attn_backend == "flex_attention":
            if context_mask.any():
                raise NotImplementedError(
                    "flex_attention with context_mask not implemented yet."
                )
            elif attention_mask is not None and (attention_mask != 1).any():
                padding_mask = create_attn_mask(attention_mask.repeat(1, 2).bool())
                dec_masks = [
                    partial(
                        self._block_mask,
                        block_size=(
                            self.config.block_size
                            if self.training
                            else self.config.eval_block_size
                        ),
                        seq_length=self.config.length,
                    ),
                    padding_mask,
                ]
                decoder_attention_mask = create_block_mask(
                    and_masks(*dec_masks),
                    B=input_ids.shape[0],
                    H=None,
                    Q_LEN=input_ids.shape[1] * 2,
                    KV_LEN=input_ids.shape[1] * 2,
                )
            else:
                decoder_attention_mask = self.static_attention_mask
        else:
            raise ValueError("Unknown attention backend")

        xt = torch.gather(xt, dim=-1, index=perm_indices)
        input_ids = torch.gather(input_ids, dim=-1, index=perm_indices)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[
            None, :
        ].repeat(batch_size, 1)
        position_ids = torch.gather(position_ids, dim=-1, index=perm_indices)

        if evaluate_nll_flag:
            xt = xt.repeat(1, 2)
        else:
            xt = torch.cat((input_ids, xt), dim=-1)
        position_ids = position_ids.repeat(1, num_repetitions)

        # NLL evaluation uses a dense SDPA mask to expose prefix context.
        if evaluate_nll_flag:
            assert self.config.attn_backend == "sdpa", (
                "eval_nll flag only supported for sdpa backend"
            )
            for i in range(batch_size):
                if self.mask_token_id not in xt[i]:
                    continue
                first_mask_token_idx = (
                    (xt[i] == self.mask_token_id).float().argmax().item()
                )
                masked_indices = torch.arange(
                    first_mask_token_idx + seq_len, seq_len * 2
                )

                # masked tokens may only attend to clean tokens and
                # previous masked tokens
                clean_indices = torch.arange(seq_len, first_mask_token_idx + seq_len)
                decoder_attention_mask[i][
                    masked_indices[:, None], clean_indices[None, :]
                ] = False
                decoder_attention_mask[i][
                    masked_indices[:, None], masked_indices[None, :] - seq_len
                ] = False
        if self.config.attn_backend == "sdpa":
            decoder_attention_mask = self._preprocess_attention_mask(
                decoder_attention_mask[:, None], dtype=torch.float
            )
        tokens_mask = attention_mask * (1 - context_mask)

        return DenoiserInput(
            xt=xt,  # type: ignore
            x0=input_ids,
            attention_mask=decoder_attention_mask,  # type: ignore
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
            backbone_kwargs={
                "position_ids": position_ids,
                "permutation_order": perm_indices,
            },
        )

    def update_cache(
        self,
        inputs: torch.LongTensor,
        cache: Optional[Dict[str, Any]] = None,
        first_hitting_times: Optional[torch.LongTensor] = None,
        **backbone_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Cache the key-value pairs for the context.
        Args:
            inputs (torch.LongTensor): The context tensor.
            cache (Dict[str, Any | None): Cache objects, e.g., past_key_values.
        Returns:
            Dict: Updated cache objects, e.g., past_key_values.
        """
        context_input, cache = self._prepare_inputs_inference(
            input_ids=inputs,
            cache=cache,
            return_updated_cache=True,
            first_hitting_times=first_hitting_times,
            **backbone_kwargs,
        )
        backbone_output = self._backbone_forward(
            context_input,
            **self._backbone_cache_kwargs(cache),
        )
        backbone_output = {k: v for k, v in backbone_output.items()}
        backbone_output.pop("logits", None)  # Do not store logits in cache
        cache = cache | backbone_output
        if getattr(self.config, "setdlm_fht_cache_order", False):
            prepared_position_ids = context_input.backbone_kwargs.get("position_ids")
            if prepared_position_ids is not None:
                cache[self._KV_CACHE_POSITION_IDS_KEY] = prepared_position_ids.clone()
        self._clone_compile_cache_if_needed(cache)
        return cache

    @staticmethod
    def _patch_decode_attention_mask_from_input_mask(
        base: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = input_mask.shape[-1]
        edit = torch.zeros_like(base, dtype=torch.bool)
        diag_r = torch.arange(seq_len, device=base.device)
        diag_c = base.size(1) - seq_len + diag_r
        edit[diag_r, diag_c] = True

        patched = torch.zeros_like(base, dtype=torch.bool)
        mask_pair = input_mask[0, :, None] & input_mask[0, None, :]
        patched[:, -seq_len:] = mask_pair
        return torch.where(patched, edit, base)

    def _prepare_inputs_inference(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        context: torch.LongTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        cache: Optional[Dict[str, Any]] = None,
        return_updated_cache: bool = False,
        position_ids: torch.LongTensor | None = None,
        first_hitting_times: torch.LongTensor | None = None,
        fast_attention_mask_kwargs: Optional[Dict[str, Any]] = None,
        **backbone_kwargs: Any,
    ) -> DenoiserInput:
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        device = input_ids.device
        seq_len = input_ids.shape[1]
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        cache_len = self._get_past_key_values_seq_length(past_key_values)
        full_seq_length = cache_len + seq_len
        cache_position_ids = cache.get(self._KV_CACHE_POSITION_IDS_KEY)
        if cache_position_ids is not None:
            assert cache_position_ids.shape[-1] == cache_len, (
                "KV cache position ledger length must match KV cache length"
            )
        fht_permuted_cache_order = bool(
            getattr(self.config, "setdlm_fht_cache_order", False)
            and first_hitting_times is not None
        )
        # crop KV cache if we would exceed model context
        if full_seq_length > self.config.length:
            overflow = full_seq_length - self.config.length
            if fht_permuted_cache_order:
                if cache_position_ids is None:
                    raise ValueError(
                        "setdlm_fht_cache_order requires a KV cache position "
                        "ledger before overflow cropping."
                    )
                keep_len = cache_len - overflow
                assert keep_len > 0, "cache_len must remain greater than 0"
                semantic_keep = cache_position_ids[0].argsort(
                    stable=True
                )[overflow:]
                keep_indices = semantic_keep.sort().values
                past_key_values = self._select_kv_cache_positions(
                    past_key_values,
                    keep_indices,
                )
                cache_position_ids = cache_position_ids.index_select(
                    dim=-1,
                    index=keep_indices.to(device=cache_position_ids.device),
                )
                cache[self._KV_CACHE_POSITION_IDS_KEY] = cache_position_ids
            else:
                past_key_values = self._crop_kv_cache_left(past_key_values, overflow)
                if cache_position_ids is not None:
                    cache[self._KV_CACHE_POSITION_IDS_KEY] = cache_position_ids[
                        :, overflow:
                    ]
            cache_len -= overflow
            full_seq_length = cache_len + input_ids.shape[-1]
            assert cache_len > 0, "cache_len must be greater than 0"
            assert cache_len + seq_len <= self.config.length, (
                "full seq length must be less than or equal to length"
            )

        perm_indices = None
        if first_hitting_times is not None:
            # mask tokens at the end
            fhs = torch.where(
                input_ids == self.mask_token_id, -1e6, first_hitting_times
            )
            perm_indices = fhs.argsort(dim=-1, stable=True, descending=True)
        else:
            cache_flag = (input_ids != self.mask_token_id).float()
            perm_indices = cache_flag.argsort(dim=-1, stable=True, descending=True)
        if return_updated_cache:
            # create a view of attention mask to prevent rematerialization
            base = self.static_attention_mask[
                cache_len : cache_len + seq_len, : cache_len + seq_len
            ]

            position_ids = torch.gather(position_ids, dim=-1, index=perm_indices)
            input_ids = torch.gather(input_ids, dim=-1, index=perm_indices)

            if fast_attention_mask_kwargs is None:
                input_mask = input_ids == self.mask_token_id
                if getattr(self, "_setdlm_dynamic_tensor_attention_mask", False):
                    # Tensor-only equivalent of the old first-mask-token patch.
                    # The old path synchronizes on input_mask.any()/argmax().item();
                    # this keeps the hot decode loop on-device.
                    attention_mask = self._patch_decode_attention_mask_from_input_mask(
                        base=base,
                        input_mask=input_mask,
                    )
                else:
                    edit = torch.zeros_like(base, dtype=torch.bool)
                    # keep self-attention
                    diag_r = torch.arange(seq_len, device=base.device)
                    diag_c = base.size(1) - seq_len + diag_r
                    edit[diag_r, diag_c] = True
                    if input_mask.any():
                        first_mask_token_idx = self._value_to_int(
                            input_mask.float().argmax(dim=-1)[0]
                        )
                    else:
                        first_mask_token_idx = seq_len
                    patched = torch.zeros_like(base)
                    if first_mask_token_idx < seq_len:
                        num_masked_tokens = seq_len - first_mask_token_idx
                        # values to write where edit=True
                        patched[-num_masked_tokens:, -num_masked_tokens:] = True
                    attention_mask = torch.where(patched, edit, base)

                attention_mask = self._preprocess_attention_mask(
                    attention_mask[None, None, ...], dtype=torch.float
                )
            else:
                input_mask = input_ids == self.mask_token_id
                if input_mask.any():
                    first_mask_token_idx = self._value_to_int(
                        input_mask.float().argmax(dim=-1)[0]
                    )
                else:
                    first_mask_token_idx = seq_len
                attention_mask = self._cached_fast_attention_mask(
                    cache_len=cache_len,
                    seq_len=seq_len,
                    first_mask_token_idx=first_mask_token_idx,
                    device=device,
                    **fast_attention_mask_kwargs,
                )
        else:
            full_possible_len = self.static_attention_mask.shape[-1] // 2
            valid_attn_indices = torch.cat(
                (
                    torch.arange(cache_len, device=device),
                    torch.arange(cache_len, cache_len + seq_len, device=device)
                    + full_possible_len,
                )
            )
            base = self.static_attention_mask[
                cache_len : cache_len + seq_len,
                valid_attn_indices,
            ]

            row = torch.arange(seq_len, device=device)
            col = cache_len + row
            diag_mask = torch.zeros_like(base, dtype=torch.bool)
            diag_mask[row, col] = True
            attention_mask = torch.where(diag_mask, torch.ones_like(base), base)
            attention_mask = self._preprocess_attention_mask(
                attention_mask[None, None, ...],
                dtype=torch.float,
            )
        return (
            DenoiserInput(
                xt=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
                past_key_values=past_key_values,
                backbone_kwargs={
                    "position_ids": position_ids,
                    "permutation_order": perm_indices,
                    "return_updated_cache": return_updated_cache,
                }
                | backbone_kwargs,
            ),
            cache,
        )

    @staticmethod
    def _value_to_int(value: Any) -> int:
        return int(value.item()) if hasattr(value, "item") else int(value)

    @staticmethod
    def _ranks_from_permutation(perm_indices: torch.LongTensor) -> torch.LongTensor:
        ranks = torch.empty_like(perm_indices)
        order = torch.arange(
            perm_indices.shape[-1], device=perm_indices.device, dtype=perm_indices.dtype
        )[None, :].expand_as(perm_indices)
        ranks.scatter_(dim=-1, index=perm_indices, src=order)
        return ranks

    @staticmethod
    def _move_target_eos_to_end_of_permutation(
        perm_indices: torch.LongTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        eos_token_id: int,
    ) -> torch.LongTensor:
        attention = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        context = context_mask.to(device=input_ids.device, dtype=torch.bool)
        target = attention & ~context
        target_eos = target & (input_ids == int(eos_token_id))
        if not target_eos.any():
            return perm_indices

        reordered = torch.empty_like(perm_indices)
        for batch_idx in range(perm_indices.shape[0]):
            order = perm_indices[batch_idx]
            ordered_attention = attention[batch_idx, order]
            ordered_context = context[batch_idx, order]
            ordered_target = target[batch_idx, order]
            ordered_eos = target_eos[batch_idx, order]
            pieces = (
                order[ordered_attention & ordered_context],
                order[ordered_target & ~ordered_eos],
                order[ordered_target & ordered_eos],
                order[~ordered_attention],
            )
            reordered[batch_idx] = torch.cat(pieces, dim=0)
        return reordered

    @staticmethod
    def _train_order_active_mask(
        *,
        accumulated_samples: torch.LongTensor,
        cache_flag: torch.BoolTensor,
        train_order_ranks: torch.LongTensor,
        target_start_idx: int,
        target_end_idx: int,
        inputs_offset: int,
        window_size: int,
        mask_token_id: int,
        pad_token_id: Optional[int],
    ) -> tuple[torch.BoolTensor, int]:
        batch_size = accumulated_samples.shape[0]
        target_positions = torch.arange(
            target_start_idx, target_end_idx, device=accumulated_samples.device
        )[None, :].expand(batch_size, -1)
        target_ranks = torch.gather(train_order_ranks, dim=-1, index=target_positions)
        sorted_target_idx = target_ranks.argsort(dim=-1, stable=True)
        sorted_positions = torch.gather(target_positions, dim=-1, index=sorted_target_idx)
        sorted_tokens = torch.gather(accumulated_samples, dim=-1, index=sorted_positions)
        sorted_cached = torch.gather(cache_flag, dim=-1, index=sorted_positions)

        visible = sorted_tokens != mask_token_id
        if pad_token_id is not None:
            visible = visible & (sorted_tokens != pad_token_id)

        uncached_visible = visible & ~sorted_cached
        uncached_missing = (~visible) & ~sorted_cached
        blocked_by_earlier_missing = uncached_missing.cumsum(dim=-1) > 0
        promotable = uncached_visible & ~blocked_by_earlier_missing

        masked_uncached = (sorted_tokens == mask_token_id) & ~sorted_cached
        active_masked = masked_uncached & (
            masked_uncached.cumsum(dim=-1) <= int(window_size)
        )
        active_sorted = promotable | active_masked

        active_absolute = torch.zeros_like(cache_flag, dtype=torch.bool)
        active_absolute.scatter_(dim=-1, index=sorted_positions, src=active_sorted)
        active_relative = active_absolute[:, inputs_offset:]

        # Generation currently asserts batch size 1. Keep the helper batch-safe by
        # using the minimum visible prefix if that assertion is ever relaxed.
        clean_len = int(promotable.sum(dim=-1).min().item())
        return active_relative, clean_len

    @staticmethod
    def _sort_relative_indices_by_train_order(
        *,
        relative_indices: torch.LongTensor,
        train_order_ranks: torch.LongTensor,
        inputs_offset: int,
    ) -> torch.LongTensor:
        absolute_indices = relative_indices + int(inputs_offset)
        order_values = torch.gather(train_order_ranks, dim=-1, index=absolute_indices)
        order = order_values.argsort(dim=-1, stable=True)
        return torch.gather(relative_indices, dim=-1, index=order)

    @staticmethod
    def _resolve_generation_eos_token_id(
        generation_config: SetDiffusionGenerationConfig,
        tokenizer: PreTrainedTokenizer | None,
    ) -> int | None:
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if torch.is_tensor(eos_token_id):
            if eos_token_id.numel() == 0:
                return None
            eos_token_id = eos_token_id.flatten()[0].item()
        elif isinstance(eos_token_id, (list, tuple)):
            if not eos_token_id:
                return None
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            return None
        try:
            return int(eos_token_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _planned_predict_and_noise_decode_counts(
        timesteps: torch.Tensor,
        block_size: int,
    ) -> list[int]:
        timestep_row = timesteps[0] if timesteps.ndim == 2 else timesteps
        next_timesteps = torch.empty_like(timestep_row)
        if timestep_row.shape[-1] > 1:
            next_timesteps[:-1] = timestep_row[1:]
        next_timesteps[-1] = 0
        counts = (
            (timestep_row.detach().cpu() * block_size).round().to(torch.int64)
            - (next_timesteps.detach().cpu() * block_size).round().to(torch.int64)
        )
        return counts.clamp_min(0).tolist()

    @staticmethod
    def _can_use_dynamic_full_window_fastpath(
        *,
        generation_config: SetDiffusionGenerationConfig,
        fast_inference: bool,
        bucketed_decode: bool,
        is_infill_task: bool,
        window_size: int,
        num_mask_tokens: int,
        stopping_criteria: StoppingCriteriaList | None,
    ) -> bool:
        return (
            bool(
                getattr(
                    generation_config,
                    "setdlm_dynamic_full_window_fastpath",
                    False,
                )
            )
            and not fast_inference
            and not bucketed_decode
            and is_infill_task
            and bool(getattr(generation_config, "cache_full_infill_context", False))
            and window_size >= num_mask_tokens
            and generation_config.sampling_strategy == "predict_and_noise"
            and not generation_config.do_sample
            and generation_config.nucleus_p >= 1.0
            and not generation_config.first_hitting
            and not generation_config.confidence_based_noising
            and not generation_config.confidence_margin_based_noising
            and not generation_config.compute_inf_budget
            and stopping_criteria is None
        )

    @staticmethod
    def _fast_inference_enabled(
        generation_config: SetDiffusionGenerationConfig,
    ) -> bool:
        return bool(getattr(generation_config, "setdlm_fast_inference", False))

    def _compile_decode_bucket_len(
        self,
        active_len: int,
        cache_len: int,
        generation_config: SetDiffusionGenerationConfig,
    ) -> int:
        if active_len <= 0 or not (
            getattr(generation_config, "compile_stable_decode", False)
            or self._fast_inference_enabled(generation_config)
        ):
            return active_len

        bucket_sizes = sorted(
            {
                int(bucket_size)
                for bucket_size in getattr(
                    generation_config, "compile_decode_bucket_sizes", ()
                )
                if int(bucket_size) > 0
            }
        )
        for bucket_len in bucket_sizes:
            if active_len <= bucket_len:
                break
        else:
            if not self._warned_compile_stable_decode_unbucketed:
                print(
                    "SetDLM compile_stable_decode: active decode length "
                    f"{active_len} exceeds configured buckets {bucket_sizes}; "
                    "falling back to the unbucketed length."
                )
                self._warned_compile_stable_decode_unbucketed = True
            return active_len

        if cache_len + bucket_len > self.config.length:
            if not self._warned_compile_stable_decode_unbucketed:
                print(
                    "SetDLM compile_stable_decode: bucketed decode length would "
                    "overflow the model context; falling back to the unbucketed "
                    "length to preserve cache behavior."
                )
                self._warned_compile_stable_decode_unbucketed = True
            return active_len
        return bucket_len

    def _pad_compile_decode_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        first_hitting_times: torch.LongTensor | None,
        cache_len: int,
        generation_config: SetDiffusionGenerationConfig,
    ) -> tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor | None,
        int,
        int,
    ]:
        active_len = input_ids.shape[-1]
        bucket_len = self._compile_decode_bucket_len(
            active_len=active_len,
            cache_len=cache_len,
            generation_config=generation_config,
        )
        if bucket_len == active_len:
            return (
                input_ids,
                position_ids,
                first_hitting_times,
                active_len,
                bucket_len,
            )

        pad_len = bucket_len - active_len
        pad_shape = (input_ids.shape[0], pad_len)
        input_pad = torch.full(
            pad_shape,
            self.mask_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat((input_ids, input_pad), dim=-1)

        position_pad = position_ids[..., -1:].expand(-1, pad_len)
        position_ids = torch.cat((position_ids, position_pad), dim=-1)

        if first_hitting_times is not None:
            fht_pad = torch.zeros(
                pad_shape,
                dtype=first_hitting_times.dtype,
                device=first_hitting_times.device,
            )
            first_hitting_times = torch.cat((first_hitting_times, fht_pad), dim=-1)

        return input_ids, position_ids, first_hitting_times, active_len, bucket_len

    def _fast_cache_bucket_len(
        self,
        cache_len: int,
        decode_bucket_len: int,
        generation_config: SetDiffusionGenerationConfig,
    ) -> int:
        if not self._fast_inference_enabled(generation_config):
            return cache_len
        bucket_sizes = sorted(
            {
                int(bucket_size)
                for bucket_size in getattr(
                    generation_config, "setdlm_fast_cache_bucket_sizes", ()
                )
                if int(bucket_size) > 0
            }
        )
        for bucket_len in bucket_sizes:
            if (
                cache_len <= bucket_len
                and bucket_len + decode_bucket_len <= self.config.length
            ):
                return bucket_len
        return cache_len

    @staticmethod
    def _insert_fast_cache_padding_mask(
        denoiser_inputs: DenoiserInput,
        cache_len: int,
        cache_bucket_len: int,
    ) -> None:
        if cache_bucket_len <= cache_len or denoiser_inputs.attention_mask is None:
            return
        attention_mask = denoiser_inputs.attention_mask
        pad_len = cache_bucket_len - cache_len
        min_dtype = torch.finfo(attention_mask.dtype).min
        new_shape = attention_mask.shape[:-1] + (attention_mask.shape[-1] + pad_len,)
        padded_attention_mask = torch.full(
            new_shape,
            min_dtype,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        padded_attention_mask[..., :cache_len] = attention_mask[..., :cache_len]
        padded_attention_mask[..., cache_bucket_len:] = attention_mask[..., cache_len:]
        denoiser_inputs.attention_mask = padded_attention_mask

    @staticmethod
    def _append_fast_cache_padding_mask(
        denoiser_inputs: DenoiserInput,
        cache_len: int,
        cache_bucket_len: int,
    ) -> None:
        if cache_bucket_len <= cache_len or denoiser_inputs.attention_mask is None:
            return
        attention_mask = denoiser_inputs.attention_mask
        pad_len = cache_bucket_len - cache_len
        min_dtype = torch.finfo(attention_mask.dtype).min
        new_shape = attention_mask.shape[:-1] + (attention_mask.shape[-1] + pad_len,)
        padded_attention_mask = torch.full(
            new_shape,
            min_dtype,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        padded_attention_mask[..., : attention_mask.shape[-1]] = attention_mask
        denoiser_inputs.attention_mask = padded_attention_mask

    def _cached_fast_attention_mask(
        self,
        *,
        cache_len: int,
        cache_bucket_len: int,
        seq_len: int,
        active_len: int,
        first_mask_token_idx: int,
        logical_fast_cache: bool,
        device: torch.device,
    ) -> torch.Tensor:
        mask_cache = getattr(self, "_setdlm_fast_mask_cache", None)
        if mask_cache is None:
            mask_cache = {}
            self._setdlm_fast_mask_cache = mask_cache
        key = (
            device.type,
            device.index,
            int(cache_len),
            int(cache_bucket_len),
            int(seq_len),
            int(active_len),
            int(first_mask_token_idx),
            bool(logical_fast_cache),
        )
        cached = mask_cache.get(key)
        if cached is not None:
            return cached

        base = self.static_attention_mask[
            cache_len : cache_len + seq_len,
            : cache_len + seq_len,
        ]
        row = torch.arange(seq_len, device=device)
        col = base.size(1) - seq_len + row
        edit = torch.zeros_like(base, dtype=torch.bool)
        edit[row, col] = True
        patched = torch.zeros_like(base, dtype=torch.bool)
        if first_mask_token_idx < seq_len:
            num_masked_tokens = seq_len - first_mask_token_idx
            patched[-num_masked_tokens:, -num_masked_tokens:] = True
        attention_mask = torch.where(patched, edit, base)
        attention_mask = self._preprocess_attention_mask(
            attention_mask[None, None, ...], dtype=torch.float
        )

        min_dtype = torch.finfo(attention_mask.dtype).min
        if cache_bucket_len > cache_len:
            padded_shape = attention_mask.shape[:-1] + (
                cache_bucket_len + seq_len,
            )
            padded_attention_mask = torch.full(
                padded_shape,
                min_dtype,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            if logical_fast_cache:
                padded_attention_mask[..., : attention_mask.shape[-1]] = (
                    attention_mask
                )
                decode_cache_len = cache_len
            else:
                padded_attention_mask[..., :cache_len] = attention_mask[
                    ..., :cache_len
                ]
                padded_attention_mask[
                    ..., cache_bucket_len : cache_bucket_len + seq_len
                ] = attention_mask[..., cache_len:]
                decode_cache_len = cache_bucket_len
            attention_mask = padded_attention_mask
        else:
            decode_cache_len = cache_len

        if seq_len > active_len:
            attention_mask[
                ...,
                :active_len,
                decode_cache_len + active_len : decode_cache_len + seq_len,
            ] = min_dtype
        mask_cache[key] = attention_mask
        return attention_mask

    @staticmethod
    def _mask_compile_decode_padding(
        denoiser_inputs: DenoiserInput,
        active_len: int,
        bucket_len: int,
        cache_len: int,
    ) -> None:
        if bucket_len <= active_len or denoiser_inputs.attention_mask is None:
            return
        pad_key_start = cache_len + active_len
        pad_key_end = cache_len + bucket_len
        min_dtype = torch.finfo(denoiser_inputs.attention_mask.dtype).min
        denoiser_inputs.attention_mask[
            ..., :active_len, pad_key_start:pad_key_end
        ] = min_dtype

    def _sample_prior(
        self,
        inputs: torch.LongTensor,
        generation_config: DiffusionGenerationConfig,
        batch_size: int,
        max_new_tokens: int,
        block_size: int,
        is_infill_task: bool,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> torch.LongTensor:
        if is_infill_task:
            num_mask_tokens = (inputs == self.mask_token_id).sum()
        else:
            num_mask_tokens = max_new_tokens

        # Sample max generation length tensor from prior
        if is_infill_task:
            masked_tensor = inputs
        else:
            masked_tensor = self.mask_token_id * torch.ones(
                (batch_size, num_mask_tokens), dtype=torch.int64, device=device
            )
            if inputs is not None:
                masked_tensor = torch.cat([inputs, masked_tensor], dim=-1)
        return masked_tensor

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.LongTensor | None = None,
        generation_config: SetDiffusionGenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        return_dict_in_generate: Optional[bool] = False,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer: PreTrainedTokenizer | None = None,
        **kwargs: Any,
    ) -> torch.LongTensor:
        # Setup sampling variables
        if generation_config is None:
            assert getattr(self, "generation_config", None) is not None, (
                "Generation config must be provided if not present in the model."
            )
            generation_config = self.generation_config
        assert generation_config.use_cache, (
            "Generation with SetDLM requires use_cache=True."
        )
        input_length = inputs.shape[-1] if inputs is not None else 0
        batch_size = inputs.shape[0] if inputs is not None else batch_size
        assert batch_size == 1, "only batch size 1 supported for inference"
        max_length, max_new_tokens = self._compute_sampling_lengths(
            generation_config=generation_config,
            input_length=input_length,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )
        is_infill_task = inputs is not None and self.mask_token_id in inputs
        use_fht_cache_order = bool(
            getattr(self.config, "setdlm_fht_cache_order", False)
            and not is_infill_task
            and not getattr(generation_config, "align_inputs_to_blocks", False)
        )
        block_size = generation_config.block_size
        assert block_size == self.config.length, "ao-bd3lm not supported yet"
        fast_inference = self._fast_inference_enabled(generation_config)
        if fast_inference:
            if not is_infill_task:
                raise ValueError("SetDLM fast inference currently supports infilling only.")
            if generation_config.sampling_strategy != "predict_and_noise":
                raise ValueError(
                    "SetDLM fast inference currently targets predict_and_noise sampling."
                )
            if not getattr(generation_config, "cache_full_infill_context", False):
                raise ValueError(
                    "SetDLM fast inference requires cache_full_infill_context=True."
                )
            if getattr(generation_config, "align_inputs_to_blocks", False):
                raise ValueError(
                    "SetDLM fast inference requires align_inputs_to_blocks=False."
                )
            generation_config.compile_stable_decode = True
            self._setdlm_fast_inference = True
            self._setdlm_static_compile_cache = True
        bucketed_decode = fast_inference or bool(
            getattr(generation_config, "compile_stable_decode", False)
        )
        self._setdlm_dynamic_tensor_attention_mask = (
            not fast_inference
            and bool(
                getattr(
                    generation_config,
                    "setdlm_dynamic_tensor_attention_mask",
                    False,
                )
            )
        )

        pad_length = None
        if is_infill_task:
            all_position_ids = torch.arange(input_length, device=device)[
                None, :
            ].repeat(batch_size, 1)
        else:
            all_position_ids = torch.arange(
                input_length + max_new_tokens, device=device
            )[None, :].repeat(batch_size, 1)

        if is_infill_task:
            num_mask_tokens = (inputs == self.mask_token_id).sum()
            first_mask_token_idx = (
                (inputs == self.mask_token_id).float().argmax(dim=-1)[0]
            )
            last_mask_token_idx = (inputs != self.mask_token_id).float()[
                :, first_mask_token_idx:
            ].argmax(dim=-1)[0] + first_mask_token_idx
            inputs_offset = 0
        else:
            num_mask_tokens = max_new_tokens
            first_mask_token_idx = input_length
            last_mask_token_idx = input_length + max_new_tokens
            inputs_offset = input_length
        accumulated_samples = self._sample_prior(
            inputs=inputs,
            batch_size=batch_size,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            is_infill_task=is_infill_task,
            device=device,
        )
        accumulated_confidence = torch.full(
            accumulated_samples.shape,
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        accumulated_confidence[accumulated_samples != self.mask_token_id] = 1.0
        confidence_state: Dict[str, torch.Tensor] = {}
        cache_flag = torch.zeros_like(accumulated_samples, dtype=torch.bool)
        if input_length > 0 and not is_infill_task:
            cache_flag[:, :first_mask_token_idx] = True
        infill_cache_promotion_order = getattr(
            generation_config, "setdlm_infill_cache_promotion_order", "legacy"
        )
        if infill_cache_promotion_order not in {"legacy", "l2r", "first_hitting"}:
            raise ValueError(
                "setdlm_infill_cache_promotion_order must be one of "
                "['first_hitting', 'l2r', 'legacy'], got "
                f"{infill_cache_promotion_order!r}"
            )
        explicit_infill_cache_promotion_order = (
            infill_cache_promotion_order != "legacy"
        )
        decode_eos_token_id = self._resolve_generation_eos_token_id(
            generation_config, tokenizer
        )
        l2r_eos_frontier_constraint_requested = bool(
            getattr(generation_config, "setdlm_l2r_eos_frontier_constraint", False)
        )
        l2r_eos_frontier_constraint = (
            (not is_infill_task)
            and l2r_eos_frontier_constraint_requested
            and decode_eos_token_id is not None
        )
        use_first_hitting_order_in_decode = not is_infill_task and getattr(
            generation_config, "use_first_hitting_order_in_decode", False
        )
        legacy_active_window_order = not is_infill_task and bool(
            getattr(generation_config, "setdlm_legacy_active_window_order", False)
        )
        if legacy_active_window_order and use_first_hitting_order_in_decode:
            raise ValueError(
                "setdlm_legacy_active_window_order cannot be combined with "
                "use_first_hitting_order_in_decode."
            )
        decode_first_hitting_length = None
        # Keep the 1D timestep schedule separate from the per-example decode-order
        # tensor; _sample_generation_timesteps expects a vector, not [batch, length].
        timestep_first_hitting_times = None
        decode_order_first_hitting_times = None
        decode_train_order_permutation = None
        decode_train_order_ranks = None
        if generation_config.first_hitting and (
            generation_config.sampling_strategy == "posterior"
        ):
            timestep_first_hitting_times = (
                self.noise_schedule.compute_first_hitting_times(
                    batch_size=1,
                    length=num_mask_tokens,
                    device=device,
                )[0]
            )
            decode_first_hitting_length = num_mask_tokens
        if use_first_hitting_order_in_decode:
            if accumulated_samples.shape[-1] > self.config.length:
                raise ValueError(
                    "SetDLM train-order decode requires the sampled sequence to fit "
                    f"within config.length={self.config.length}, got "
                    f"{accumulated_samples.shape[-1]}."
                )
            decode_first_hitting_length = self.config.length
            train_order_t = torch.ones(
                batch_size, self.config.length, device=device, dtype=torch.float32
            )
            train_order_target_mask = torch.zeros(
                batch_size, self.config.length, device=device, dtype=torch.bool
            )
            target_order_end = min(int(input_length + max_new_tokens), self.config.length)
            train_order_target_mask[:, int(input_length) : target_order_end] = True
            decode_train_order_permutation = self.noise_schedule.sample_permutation_order(
                train_order_t,
                train_order_target_mask,
                self.config.block_size,
            )
            decode_train_order_ranks = self._ranks_from_permutation(
                decode_train_order_permutation
            )
            decode_order_scores = (
                self.config.length - decode_train_order_ranks
            ).to(torch.float32)
            decode_order_first_hitting_times = decode_order_scores[
                :, inputs_offset : inputs_offset + max_new_tokens
            ]
        if legacy_active_window_order or use_fht_cache_order:
            decode_first_hitting_length = self._value_to_int(num_mask_tokens)
            decode_order_first_hitting_times = (
                self.noise_schedule.compute_first_hitting_times(
                    batch_size=batch_size,
                    length=decode_first_hitting_length,
                    device=device,
                )
            )
        xt = accumulated_samples[:, inputs_offset:]
        timesteps = self._sample_generation_timesteps(
            generation_config,
            max_length=num_mask_tokens,
            device=device,
            first_hitting_times=timestep_first_hitting_times,
        )
        timesteps = timesteps[None, :].repeat(batch_size, 1)
        xt_position_ids = all_position_ids[:, inputs_offset:]
        all_masked_positions = xt == self.mask_token_id
        window_size = min(
            self.noise_schedule.compute_window_size(), generation_config.max_window_size
        )
        if use_first_hitting_order_in_decode and decode_train_order_ranks is not None:
            masked_positions, clean_len = self._train_order_active_mask(
                accumulated_samples=accumulated_samples,
                cache_flag=cache_flag,
                train_order_ranks=decode_train_order_ranks,
                target_start_idx=int(input_length),
                target_end_idx=int(input_length + max_new_tokens),
                inputs_offset=int(inputs_offset),
                window_size=int(window_size),
                mask_token_id=self.mask_token_id,
                pad_token_id=self.pad_token_id,
            )
            window_start = inputs_offset + masked_positions.float().argmax(dim=-1)[:, None]
        else:
            masked_positions = all_masked_positions
            window_start = inputs_offset + masked_positions.float().argmax(dim=-1)[:, None]
            masked_positions = masked_positions & (
                xt_position_ids < (window_start + window_size)
            )
        num_mask_tokens_value = self._value_to_int(num_mask_tokens)
        dynamic_full_window_fastpath = self._can_use_dynamic_full_window_fastpath(
            generation_config=generation_config,
            fast_inference=fast_inference,
            bucketed_decode=bucketed_decode,
            is_infill_task=is_infill_task,
            window_size=window_size,
            num_mask_tokens=num_mask_tokens_value,
            stopping_criteria=stopping_criteria,
        )
        planned_decode_counts = (
            self._planned_predict_and_noise_decode_counts(
                timesteps=timesteps,
                block_size=num_mask_tokens_value,
            )
            if dynamic_full_window_fastpath
            else None
        )
        remaining_masks_to_decode = num_mask_tokens_value
        infill_cache_first_hitting_times = None
        infill_cache_first_hitting_length = None
        if is_infill_task:
            infill_cache_first_hitting_length = accumulated_samples.shape[-1]
            # Cache-promotion order should be a property of this generation, not
            # resampled every decode step. Reusing it also avoids repeated RNG
            # calls and allocations in the hottest infilling loop. Existing eval
            # rows keep the historical behavior unless the explicit order flag is
            # set.
            if infill_cache_promotion_order in {"first_hitting", "legacy"}:
                infill_cache_first_hitting_times = (
                    self.noise_schedule.compute_first_hitting_times(
                        batch_size=batch_size,
                        length=infill_cache_first_hitting_length,
                        device=device,
                    )
                )

        if input_length > 0:
            if is_infill_task and getattr(
                generation_config, "cache_full_infill_context", False
            ):
                # Match the pre-migration AnyOrderBD3LM infilling behavior:
                # condition on both left and right context from the start.
                cache_flag = accumulated_samples != self.mask_token_id
                inputs_indices = cache_flag.nonzero(as_tuple=False)[:, 1].sort().values
                if inputs_indices.numel() > self.config.length:
                    inputs_indices = inputs_indices[-self.config.length :]
                    cache_flag = torch.zeros_like(accumulated_samples, dtype=torch.bool)
                    cache_flag[:, inputs_indices] = True
            else:
                # Prefix-only cache used by the newer SetDLM path.
                cache_flag = torch.zeros_like(accumulated_samples, dtype=torch.bool)
                cache_flag[:, :first_mask_token_idx] = True
                first_mask_token_idx_value = (
                    first_mask_token_idx.item()
                    if hasattr(first_mask_token_idx, "item")
                    else first_mask_token_idx
                )
                inputs_indices = torch.arange(
                    int(first_mask_token_idx_value), device=device
                )
            cache = self._new_generation_cache(batch_size=batch_size, device=device)
            initial_cache_first_hitting_times = None
            if (
                is_infill_task
                and infill_cache_promotion_order == "first_hitting"
                and infill_cache_first_hitting_times is not None
            ):
                initial_cache_first_hitting_times = torch.gather(
                    infill_cache_first_hitting_times,
                    dim=-1,
                    index=inputs_indices[None, :].expand(batch_size, -1),
                )
            cache = self.update_cache(
                inputs=inputs[:, inputs_indices],
                position_ids=all_position_ids[:, inputs_indices],
                cache=cache,
                first_hitting_times=initial_cache_first_hitting_times,
            )
        else:
            cache = self._new_generation_cache(batch_size=batch_size, device=device)
            inputs_indices = torch.empty(0, device=device)
            if getattr(self.config, "setdlm_fht_cache_order", False):
                cache[self._KV_CACHE_POSITION_IDS_KEY] = all_position_ids[:, :0]

        total_NFEs = 0
        is_done = torch.Tensor([False])
        num_tokens_generated_per_step = []
        inf_budget_per_step = []
        block_NFEs = 0
        clean_len = 0
        for i in range(timesteps.shape[-1]):
            block_NFEs += 1
            total_NFEs += 1
            t = timesteps[:, i]
            if generation_config.first_hitting:
                num_generated = sum(num_tokens_generated_per_step)
                next_t = (
                    timesteps[:, num_generated + 1]
                    if num_generated + 1 < timesteps.shape[-1]
                    else timesteps[:, -1] * 0
                )
            else:
                next_t = (
                    timesteps[:, i + 1]
                    if i < timesteps.shape[-1] - 1
                    else timesteps[:, -1] * 0
                )
            planned_num_generated = (
                planned_decode_counts[i] if planned_decode_counts is not None else None
            )

            masked_positions_indices = masked_positions.nonzero(as_tuple=False)[
                :, -1
            ].view(batch_size, -1)
            if (
                use_first_hitting_order_in_decode
                and decode_train_order_ranks is not None
                and masked_positions_indices.numel() > 0
            ):
                masked_positions_indices = self._sort_relative_indices_by_train_order(
                    relative_indices=masked_positions_indices,
                    train_order_ranks=decode_train_order_ranks,
                    inputs_offset=int(inputs_offset),
                )
            masked_xt = torch.gather(xt, dim=-1, index=masked_positions_indices)
            masked_position_ids = torch.gather(
                xt_position_ids, dim=-1, index=masked_positions_indices
            )
            unpadded_masked_position_ids = masked_position_ids
            masked_first_hitting_times = None
            if (
                (
                    use_first_hitting_order_in_decode
                    or legacy_active_window_order
                    or use_fht_cache_order
                )
                and decode_order_first_hitting_times is not None
            ):
                masked_first_hitting_times = torch.gather(
                    decode_order_first_hitting_times,
                    dim=-1,
                    index=masked_positions_indices,
                )
            elif (
                is_infill_task
                and infill_cache_promotion_order == "first_hitting"
                and infill_cache_first_hitting_times is not None
            ):
                masked_first_hitting_times = torch.gather(
                    infill_cache_first_hitting_times,
                    dim=-1,
                    index=masked_position_ids,
                )
            # Only decode masked tokens
            cache_len = cache["past_key_values"].get_seq_length()
            return_updated_cache = i > 0
            if bucketed_decode:
                (
                    masked_xt,
                    masked_position_ids,
                    masked_first_hitting_times,
                    active_len,
                    bucket_len,
                ) = self._pad_compile_decode_inputs(
                    input_ids=masked_xt,
                    position_ids=masked_position_ids,
                    first_hitting_times=masked_first_hitting_times,
                    cache_len=cache_len,
                    generation_config=generation_config,
                )
                cache_bucket_len = self._fast_cache_bucket_len(
                    cache_len=cache_len,
                    decode_bucket_len=bucket_len,
                    generation_config=generation_config,
                )
            else:
                active_len = masked_xt.shape[-1]
                bucket_len = active_len
                cache_bucket_len = cache_len
            past_key_values = cache.get("past_key_values")
            logical_fast_cache = (
                fast_inference
                and bool(getattr(generation_config, "setdlm_fast_logical_cache", False))
                and hasattr(past_key_values, "prepare_logical_write")
            )
            if logical_fast_cache:
                past_key_values.prepare_logical_write(cache_bucket_len + bucket_len)
            elif fast_inference and hasattr(past_key_values, "prepare_write"):
                past_key_values.prepare_write(cache_bucket_len)
            fast_attention_mask_kwargs = None
            if return_updated_cache and fast_inference and bool(
                getattr(generation_config, "setdlm_fast_tensor_cache", False)
            ):
                fast_attention_mask_kwargs = {
                    "cache_bucket_len": cache_bucket_len,
                    "active_len": active_len,
                    "logical_fast_cache": logical_fast_cache,
                }
            denoiser_inputs, cache = self._prepare_inputs_inference(
                input_ids=masked_xt,
                cache=cache,
                position_ids=masked_position_ids,
                first_hitting_times=masked_first_hitting_times,
                return_updated_cache=return_updated_cache,
                fast_attention_mask_kwargs=fast_attention_mask_kwargs,
            )
            # _prepare_inputs_inference may crop an overflowing KV cache. Refresh the
            # logical cache length before building masks or promoting new clean tokens.
            cache_len = self._get_past_key_values_seq_length(
                denoiser_inputs.past_key_values
            )
            if bucketed_decode:
                if fast_attention_mask_kwargs is not None:
                    padding_mask_cache_len = (
                        cache_len if logical_fast_cache else cache_bucket_len
                    )
                else:
                    if logical_fast_cache:
                        self._append_fast_cache_padding_mask(
                            denoiser_inputs=denoiser_inputs,
                            cache_len=cache_len,
                            cache_bucket_len=cache_bucket_len,
                        )
                        padding_mask_cache_len = cache_len
                    else:
                        self._insert_fast_cache_padding_mask(
                            denoiser_inputs=denoiser_inputs,
                            cache_len=cache_len,
                            cache_bucket_len=cache_bucket_len,
                        )
                        padding_mask_cache_len = cache_bucket_len
                    self._mask_compile_decode_padding(
                        denoiser_inputs=denoiser_inputs,
                        active_len=active_len,
                        bucket_len=bucket_len,
                        cache_len=padding_mask_cache_len,
                    )
            active_decode_len = None
            position_ids_for_sample = denoiser_inputs.backbone_kwargs.get(
                "position_ids"
            )
            if position_ids_for_sample is None:
                sample_indices = masked_positions_indices + inputs_offset
            else:
                clean_len_value = (
                    self._value_to_int(clean_len) if return_updated_cache else 0
                )
                if bucketed_decode:
                    active_decode_len = max(active_len - clean_len_value, 0)
                    sample_indices = position_ids_for_sample[
                        :, clean_len_value : clean_len_value + active_decode_len
                    ]
                else:
                    sample_indices = position_ids_for_sample[:, clean_len_value:]
            if not is_infill_task:
                running_generation = accumulated_samples[
                    :, first_mask_token_idx:last_mask_token_idx
                ]
            else:
                running_generation = accumulated_samples[cache_flag].unsqueeze(0)
            repetition_penalty_context = None
            infill_context_no_repeat_ngram_context = None
            infill_context_ngram_size = int(
                getattr(
                    generation_config,
                    "infill_context_no_repeat_ngram_size",
                    0,
                )
                or 0
            )
            needs_visible_infill_context = is_infill_task and (
                getattr(
                    generation_config,
                    "infill_repetition_penalty_include_right_context",
                    False,
                )
                or infill_context_ngram_size > 0
            )
            if needs_visible_infill_context:
                visible_infill_context = self._visible_infill_context(
                    accumulated_samples=accumulated_samples,
                    mask_token_id=self.mask_token_id,
                    pad_token_id=self.pad_token_id,
                )
                if getattr(
                    generation_config,
                    "infill_repetition_penalty_include_right_context",
                    False,
                ):
                    repetition_penalty_context = visible_infill_context
                if infill_context_ngram_size > 0:
                    infill_context_no_repeat_ngram_context = accumulated_samples
            length_penalty_prefix_lengths = None
            if logits_processor is not None and len(logits_processor) > 0:
                length_penalty_prefix_lengths = self._length_penalty_prefix_lengths(
                    accumulated_samples=accumulated_samples,
                    sample_indices=sample_indices,
                    target_start_idx=first_mask_token_idx,
                    mask_token_id=self.mask_token_id,
                    pad_token_id=self.pad_token_id,
                )
            eos_frontier_prefix_lengths = None
            eos_allowed_mask = None
            eos_constraints_active = (
                (not is_infill_task)
                and decode_eos_token_id is not None
                and l2r_eos_frontier_constraint
            )
            if eos_constraints_active:
                if l2r_eos_frontier_constraint:
                    eos_frontier_prefix_lengths, eos_allowed_mask = (
                        self._l2r_eos_frontier_mask(
                            accumulated_samples=accumulated_samples,
                            sample_indices=sample_indices,
                            target_start_idx=first_mask_token_idx,
                            mask_token_id=self.mask_token_id,
                            pad_token_id=self.pad_token_id,
                        )
                    )

            generation_output = self._generate_unconditional(
                generation_config=generation_config,
                t=t[0],
                next_t=next_t[0],
                denoiser_inputs=denoiser_inputs,
                cache=cache,
                xt=xt,
                running_generation=running_generation,
                repetition_penalty_context=repetition_penalty_context,
                infill_context_no_repeat_ngram_context=infill_context_no_repeat_ngram_context,
                inputs_offset=inputs_offset,
                length_penalty_prefix_lengths=length_penalty_prefix_lengths,
                eos_allowed_mask=eos_allowed_mask,
                eos_token_id=(
                    decode_eos_token_id
                    if eos_allowed_mask is not None
                    else None
                ),
                logits_processor=logits_processor,
                return_updated_cache=return_updated_cache,
                cache_len=clean_len,
                sample_indices=sample_indices,
                active_decode_len=active_decode_len,
                project_active_logits=(
                    (
                        fast_inference
                        and bool(
                            getattr(
                                generation_config,
                                "setdlm_fast_active_logits",
                                False,
                            )
                        )
                    )
                    or (
                        not fast_inference
                        and bool(
                            getattr(
                                generation_config,
                                "setdlm_dynamic_active_logits",
                                False,
                            )
                        )
                    )
                ),
                window_size=window_size,
                block_size=(
                    num_mask_tokens_value
                    if dynamic_full_window_fastpath
                    else num_mask_tokens
                ),
                confidence_state=confidence_state,
                **kwargs,
            )
            xs, cache = generation_output
            sample_confidence = confidence_state.get("sample_confidence")
            # crop kv cache and sampling output
            if return_updated_cache:
                # only keep cache for the clean tokens
                position_ids = denoiser_inputs.backbone_kwargs["position_ids"]
                num_cached_position_ids = int(self._value_to_int(clean_len))
                # update accumulated_samples, xt, masked_positions
                cached_position_ids = position_ids[:, :num_cached_position_ids]
                unmasked_position_ids = position_ids[:, num_cached_position_ids:]
                cache["past_key_values"].crop(cache_len + num_cached_position_ids)
                if self._KV_CACHE_POSITION_IDS_KEY in cache:
                    cache[self._KV_CACHE_POSITION_IDS_KEY] = torch.cat(
                        (
                            cache[self._KV_CACHE_POSITION_IDS_KEY],
                            cached_position_ids,
                        ),
                        dim=-1,
                    )
                self._clone_compile_cache_if_needed(cache)
                if planned_num_generated is not None:
                    clean_len = min(
                        int(planned_num_generated),
                        int(unmasked_position_ids.shape[-1]),
                    )
                else:
                    clean_len = (
                        (
                            torch.gather(
                                xt,
                                dim=-1,
                                index=(unmasked_position_ids - inputs_offset),
                            )
                            != xs
                        )
                        .sum(dim=-1)
                        .min()
                    )
                accumulated_samples.scatter_(1, unmasked_position_ids, xs)
                if sample_confidence is not None:
                    confidence_values = sample_confidence.to(accumulated_confidence)
                    confidence_values = torch.where(
                        xs != self.mask_token_id,
                        confidence_values,
                        torch.full_like(confidence_values, float("nan")),
                    )
                    accumulated_confidence.scatter_(
                        1, unmasked_position_ids, confidence_values
                    )
                if is_infill_task or use_first_hitting_order_in_decode:
                    cache_flag.scatter_(1, cached_position_ids, True)
                xt.scatter_(1, unmasked_position_ids - inputs_offset, xs)
            else:
                xt.scatter_(1, unpadded_masked_position_ids - inputs_offset, xs)
                if planned_num_generated is not None:
                    clean_len = min(int(planned_num_generated), int(xs.shape[-1]))
                else:
                    clean_len = (xs != self.mask_token_id).sum(dim=-1).min()
                accumulated_samples.scatter_(1, unpadded_masked_position_ids, xs)
                if sample_confidence is not None:
                    confidence_values = sample_confidence.to(accumulated_confidence)
                    confidence_values = torch.where(
                        xs != self.mask_token_id,
                        confidence_values,
                        torch.full_like(confidence_values, float("nan")),
                    )
                    accumulated_confidence.scatter_(
                        1, unpadded_masked_position_ids, confidence_values
                    )
            # for infilling, cache tokens that would be decoded before the
            # response tokens are generated
            if (
                is_infill_task
                and not dynamic_full_window_fastpath
                and (accumulated_samples == self.mask_token_id).any()
            ):
                just_unmasked_idx = (masked_positions) & (xt != self.mask_token_id)
                first_masked_position_id = 0
                last_potential_position_id = accumulated_samples.shape[-1]
                just_unmasked_slice = just_unmasked_idx[
                    :, first_masked_position_id:last_potential_position_id
                ]
                accumulated_slice = accumulated_samples[
                    :, first_masked_position_id:last_potential_position_id
                ]
                cache_flag_slice = cache_flag[
                    :, first_masked_position_id:last_potential_position_id
                ]
                masked_positions_slice = masked_positions[
                    :, first_masked_position_id:last_potential_position_id
                ]
                new_cache_positions = None
                if infill_cache_promotion_order == "l2r" and just_unmasked_slice.any():
                    position_ids = torch.arange(
                        last_potential_position_id - first_masked_position_id,
                        device=device,
                    )[None, :]
                    l2r_frontier = torch.where(
                        just_unmasked_slice,
                        position_ids,
                        torch.full_like(position_ids, -1),
                    ).max(dim=-1).values[:, None]
                    new_cache_positions = (
                        (position_ids <= l2r_frontier)
                        & (accumulated_slice != self.mask_token_id)
                        & ~cache_flag_slice
                        & ~masked_positions_slice
                    )
                    target_unmasking_time = None
                    cache_first_hitting_times = None
                elif (
                    infill_cache_first_hitting_times is not None
                    and just_unmasked_slice.any()
                ):
                    cache_first_hitting_times = infill_cache_first_hitting_times[
                        :, first_masked_position_id:last_potential_position_id
                    ]
                    target_unmasking_time = cache_first_hitting_times[
                        just_unmasked_slice
                    ].min()
                    new_cache_positions = (
                        (cache_first_hitting_times >= target_unmasking_time)
                        & (accumulated_slice != self.mask_token_id)
                        & ~cache_flag_slice
                        & ~masked_positions_slice
                )
                if new_cache_positions is not None:
                    clean_len += new_cache_positions.sum()
                    masked_positions[
                        :, first_masked_position_id:last_potential_position_id
                    ] |= new_cache_positions
                    cache_flag[
                        :, first_masked_position_id:last_potential_position_id
                    ] |= new_cache_positions
            if return_updated_cache:
                masked_positions.scatter_(1, cached_position_ids - inputs_offset, False)
            if use_first_hitting_order_in_decode and decode_train_order_ranks is not None:
                masked_positions, clean_len = self._train_order_active_mask(
                    accumulated_samples=accumulated_samples,
                    cache_flag=cache_flag,
                    train_order_ranks=decode_train_order_ranks,
                    target_start_idx=int(input_length),
                    target_end_idx=int(input_length + max_new_tokens),
                    inputs_offset=int(inputs_offset),
                    window_size=int(window_size),
                    mask_token_id=self.mask_token_id,
                    pad_token_id=self.pad_token_id,
                )
            elif dynamic_full_window_fastpath:
                masked_positions |= xt == self.mask_token_id
            else:
                window_start = (
                    (accumulated_samples == self.mask_token_id)
                    .float()
                    .argmax(dim=-1)[:, None]
                )
                masked_positions |= (xt == self.mask_token_id) & (
                    xt_position_ids < (window_start + window_size)
                )
            if planned_num_generated is not None:
                generated_this_step = min(
                    int(planned_num_generated),
                    remaining_masks_to_decode,
                )
                remaining_masks_to_decode -= generated_this_step
                num_tokens_generated_per_step.append(generated_this_step)
            else:
                num_tokens_generated_per_step.append(
                    (xs != self.mask_token_id).sum().item()
                )
            if generation_config.compute_inf_budget:
                t_for_budget = t.unsqueeze(1).repeat(1, num_mask_tokens)
                next_t_for_budget = next_t.unsqueeze(1).repeat(1, num_mask_tokens)
                alpha_t_schedule, _ = self.noise_schedule(t_for_budget)
                alpha_s_schedule, _ = self.noise_schedule(next_t_for_budget)
                alpha_t_prime = (alpha_s_schedule - alpha_t_schedule).abs()
                inf_budget = (
                    ((xt == self.mask_token_id) & (alpha_t_prime != 0.0)).sum().item()
                )
                inf_budget_per_step.append(inf_budget)
            done_decoding = (
                remaining_masks_to_decode <= 0
                if planned_num_generated is not None
                else (xt == self.mask_token_id).sum().item() == 0
            )
            if done_decoding:
                if generation_config.compute_inf_budget:
                    # for inf budget calculation avg over all timesteps
                    remaining_steps = timesteps.shape[-1] - len(inf_budget_per_step)
                    inf_budget_per_step.extend([0] * remaining_steps)
                break
            check_stopping_criteria = (i % window_size == 0) and (i > 0)
            if (
                check_stopping_criteria
                and (not generation_config.compute_inf_budget)
                and stopping_criteria is not None
            ):
                is_done = stopping_criteria(
                    input_ids=accumulated_samples[  # type: ignore
                        :, : window_start[0] + window_size
                    ],
                    scores=None,  # type: ignore
                    token_confidence=accumulated_confidence[
                        :, : window_start[0] + window_size
                    ],
                )
                if torch.any(is_done):
                    if not is_infill_task:
                        accumulated_samples = accumulated_samples[
                            :,
                            : window_start[0] + window_size,
                        ]
                    break
        if pad_length is not None:
            accumulated_samples = accumulated_samples[:, :-pad_length]
        parallelism_factor = sum(num_tokens_generated_per_step) / len(
            num_tokens_generated_per_step
        )
        inf_budget = None
        if generation_config.compute_inf_budget:
            inf_budget = sum(inf_budget_per_step) / len(inf_budget_per_step)
        accumulated_samples = accumulated_samples[
            accumulated_samples != self.mask_token_id
        ].unsqueeze(0)
        if return_dict_in_generate:
            return DiffusionGenerationOutput(
                sequences=accumulated_samples,
                parallelism_factor=parallelism_factor,
                inf_budget=inf_budget,
                inf_budgets=inf_budget_per_step,
            )
        return accumulated_samples  # type: ignore

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs,
    ) -> torch.FloatTensor:
        # Zero-mask probability
        backbone_output[..., self.mask_token_id] = self.neg_infinity
        log_probs = backbone_output - torch.logsumexp(
            backbone_output, dim=-1, keepdim=True
        )
        return log_probs  # type: ignore
