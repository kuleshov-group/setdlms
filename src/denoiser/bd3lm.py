from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.cache_utils import Cache, DynamicCache

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        and_masks,
        create_block_mask,
    )
except ImportError:
    BlockMask, and_masks, create_block_mask = None, None, None

from src.denoiser.base import DenoiserInput, LossAndNllOutput
from src.denoiser.diffusion_config import create_attn_mask
from src.denoiser.mdlm import MDLM, MDLMConfig


class BD3LMConfig(MDLMConfig):
    """Configuration class for BD3LM models."""

    model_type = "bd3lm"
    auto_map = {
        "AutoConfig": "diffusion.BD3LMConfig",
        "AutoModel": "diffusion.BD3LM",
        "AutoModelForMaskedLM": "diffusion.BD3LM",
    }

    def __init__(
        self,
        block_size: Optional[int] = None,
        eval_block_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.eval_block_size = (
            eval_block_size if eval_block_size is not None else block_size
        )


class BD3LM(MDLM):
    """Denoiser class for BD3LM models."""

    config_class = BD3LMConfig

    def __init__(
        self,
        config: BD3LMConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, **kwargs)

    # noinspection PyUnusedLocal
    @staticmethod
    def _block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        block_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        del b, h

        # Indicate whether token belongs to xt or x0:
        xt_flag_q = (q_idx >= seq_length).bool()
        xt_flag_kv = (kv_idx >= seq_length).bool()

        # Compute block indices
        block_q = torch.where(
            xt_flag_q, (q_idx - seq_length) // block_size, q_idx // block_size
        )
        block_kv = torch.where(
            xt_flag_kv, (kv_idx - seq_length) // block_size, kv_idx // block_size
        )
        # **1. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q > block_kv) & ~xt_flag_kv & xt_flag_q

        # **2. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q == block_kv) & (xt_flag_q == xt_flag_kv)

        # **3. Block-Causal Mask (M_BC) **
        block_causal = (block_q >= block_kv) & ~xt_flag_kv & ~xt_flag_q

        # **3. Combine Masks **
        return block_diagonal | offset_block_causal | block_causal

    def _create_static_mask(self) -> None:
        static_mask = self.generate_static_mask()

        if self.config.attn_backend == "flex_attention":
            self.static_attention_mask = static_mask
        else:
            self.register_buffer(
                "static_attention_mask",
                static_mask,
            )
            self.skip_params_for_push.append("static_attention_mask")

    def generate_static_mask(self) -> Union[torch.Tensor, BlockMask]:
        if self.config.attn_backend == "flex_attention":
            mask = partial(
                self._block_mask,
                block_size=(
                    self.config.block_size
                    if self.training
                    else self.config.eval_block_size
                ),
                seq_length=self.config.length,
            )
            return create_block_mask(
                mask,
                B=None,
                H=None,
                Q_LEN=self.config.length * 2,
                KV_LEN=self.config.length * 2,
            )
        else:
            static_mask = self._block_mask(
                b=None,
                h=None,
                q_idx=torch.arange(self.config.length * 2)[:, None],
                kv_idx=torch.arange(self.config.length * 2)[None, :],
                block_size=self.config.block_size,
                seq_length=self.config.length,
            )
            return static_mask

    def update_static_mask(
        self,
        new_static_mask: Union[torch.Tensor, BlockMask],
    ) -> None:
        self.static_attention_mask.copy_(new_static_mask)

    def _ensure_no_unmasked_blocks(
        self,
        input_ids: torch.LongTensor,
        xt: torch.LongTensor,
        context_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        pad_length = self.config.block_size - (xt.shape[1] % self.config.block_size)
        if pad_length > 0:
            xt = F.pad(xt, (0, pad_length), value=self.pad_token_id)
            context_mask = F.pad(context_mask, (0, pad_length), value=0)
        n_blocks = xt.shape[1] // self.config.block_size
        # If context overlaps w/block, ignore it
        blocks_without_masks = ((xt == self.mask_token_id) + context_mask).reshape(
            -1, n_blocks, self.config.block_size
        ).sum(dim=-1) == 0
        if blocks_without_masks.sum() > 0:
            num_remasks_per_block = torch.randint(
                0,
                self.config.block_size,
                blocks_without_masks.shape,
                device=xt.device,
            )
            rand = torch.rand(xt.shape[0], xt.shape[1], device=xt.device)
            perm_indices = torch.argsort(
                rand.view(xt.shape[0], n_blocks, self.config.block_size),
                stable=True,
                dim=-1,
            )
            remask_indices = perm_indices <= num_remasks_per_block[..., None]
            xt = torch.where(
                remask_indices.view(xt.shape[0], xt.shape[1])
                * blocks_without_masks.repeat_interleave(self.config.block_size, dim=1),
                self.mask_token_id,
                xt,
            )
            if self.config.keep_clean_bos:
                xt[..., 0] = input_ids[..., 0]
        if pad_length > 0:
            xt = xt[:, :-pad_length]
            context_mask = context_mask[:, :-pad_length]
        return xt

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
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
                (
                    input_ids.shape[1] // self.config.block_size
                    if self.training
                    else self.config.eval_block_size
                ),
                device=input_ids.device,
            ).repeat_interleave(
                (
                    self.config.block_size
                    if self.training
                    else self.config.eval_block_size
                ),
                dim=-1,
            )
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]
        noise_mask = context_mask | ~(attention_mask.bool())
        if getattr(self.config, "mdlm_loss_scale", False):
            eps = 1e-3
            t = 1 - alpha_t
            sigma = -torch.log1p(-(1 - eps) * t)
            p = 1 - torch.exp(-sigma)
            xt = self._sample_q_xt(x0=input_ids, alpha_t=1 - p, mask=noise_mask)
        else:
            xt = self._sample_q_xt(x0=input_ids, alpha_t=alpha_t, mask=noise_mask)
        # Ensure each block has at least 1 masked token
        if self.training or self.config.block_size == 1:
            xt = self._ensure_no_unmasked_blocks(
                input_ids,
                xt,
                noise_mask,
            )
        if self.config.attn_backend == "sdpa":
            decoder_attention_mask = (
                self.static_attention_mask[None, ...]
                & attention_mask.repeat(1, 2)[:, None, :]
                & attention_mask.repeat(1, 2)[..., None]
            )[:, None, ...]  # Make attention mask 4D
            decoder_attention_mask = self._preprocess_attention_mask(
                decoder_attention_mask, dtype=torch.float
            )
        elif self.config.attn_backend == "flex_attention":
            if context_mask.any():
                raise NotImplementedError(
                    "flex_attention with context_mask not implemented yet."
                )
            elif attention_mask is not None and (attention_mask != 1).any():
                padding_mask = create_attn_mask(
                    attention_mask.bool().repeat(2, 2).bool()
                )
                dec_masks = [
                    partial(self._block_mask, block_size=self.config.block_size),
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
            raise ValueError("Unknown backbone backend")
        backbone_input_ids = torch.cat((input_ids, xt), dim=-1)
        position_ids = (
            torch.arange(input_ids.shape[1]).repeat(2).to(input_ids.device)[None, :]
        )
        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask)
        return DenoiserInput(
            xt=backbone_input_ids,  # type: ignore
            x0=input_ids,
            attention_mask=decoder_attention_mask,  # type: ignore
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
            backbone_kwargs={
                "cache_position": position_ids[0],
                "position_ids": position_ids,
            },
        )

    def _crop_kv_cache_left(self, past_key_values: Any, drop: int) -> Any:
        """
        Drop `drop` tokens from the *left/oldest* side of the KV cache.
        Works with common DynamicCache-like implementations that store per-layer
        key/value tensors in `key_cache` / `value_cache` lists.
        Falls back to no-op if structure is unknown.
        """
        if drop <= 0 or past_key_values is None:
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

            key_cache[i] = k[..., drop:, :]
            value_cache[i] = v[..., drop:, :]
        return past_key_values

    def _prepare_inputs_inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context: Optional[torch.LongTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        cache: Optional[Dict[str, Any]] = None,
        return_updated_cache: bool = False,
        **backbone_kwargs: Dict[str, Any],
    ) -> Tuple[DenoiserInput, Union[Dict[str, Any], None]]:
        device = input_ids.device if input_ids is not None else context.device
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        if context is not None:
            if input_ids is not None:
                input_ids = torch.cat([context, input_ids], dim=-1)
            else:
                input_ids = context
        cache_length = self._get_past_key_values_seq_length(past_key_values)
        full_seq_length = cache_length + input_ids.shape[-1]
        # --- crop KV cache if we would exceed model context ---
        if full_seq_length > self.config.length:
            overflow = full_seq_length - self.config.length
            past_key_values = self._crop_kv_cache_left(past_key_values, overflow)
            cache_length = cache_length - overflow
            full_seq_length = cache_length + input_ids.shape[-1]
        # subset of block-causal mask
        decoder_attention_mask = self.static_attention_mask[
            None,
            None,
            cache_length:full_seq_length,
            :full_seq_length,
        ]  # Make attention mask 4D
        decoder_attention_mask = self._preprocess_attention_mask(
            decoder_attention_mask, dtype=torch.float
        )
        position_ids = torch.arange(cache_length, full_seq_length).to(device)[None, :]
        return (
            DenoiserInput(
                xt=input_ids,
                attention_mask=decoder_attention_mask,
                context_mask=context_mask,
                past_key_values=past_key_values,
                backbone_kwargs={
                    "position_ids": position_ids,
                }
                | backbone_kwargs,
            ),
            cache,
        )

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        input_length = denoiser_inputs.x0.shape[1]
        model_output = model_output[:, input_length : input_length * 2, ...]
        denoiser_inputs.xt = denoiser_inputs.xt[:, input_length : input_length * 2]
        return super()._compute_loss(
            model_output=model_output,  # type: ignore
            denoiser_inputs=denoiser_inputs,
            **kwargs,
        )
