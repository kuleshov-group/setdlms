from typing import Any, Literal

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DynamicCache,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from src.backbone.custom_modeling_qwen3 import CustomQwen3ForCausalLM
from src.backbone.dit import (
    esolm_bidirectional_mask,
    esolm_causal_mask,
    esolm_mixed2_mask,
    esolm_mixed_mask,
    esolm_sequential_block_mask,
)

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None

AUTO_MODEL_CLS = {
    "AutoModel": AutoModel,
    "AutoModelForCausalLM": AutoModelForCausalLM,
    "AutoModelForMaskedLM": AutoModelForMaskedLM,
}


class AutoModelFromPreTrained(nn.Module):
    """Simple wrapper class that enables using AutoModel from pre-trained."""

    def __init__(
        self,
        automodel_cls: Literal[
            "AutoModel",
            "AutoModelForCausalLM",
            "AutoModelForMaskedLM",
        ],
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        num_layers: int = -1,
        keep_top_layers: bool = False,
        reinit_model: bool = False,
        use_causal_mask: bool = False,
        is_esolm_backbone: bool = False,
        **automodel_init_kwargs,
    ):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.is_esolm_backbone = is_esolm_backbone
        self._esolm_past_key_values = None
        if reinit_model:
            auto_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                num_hidden_layers=num_layers,
                trust_remote_code=trust_remote_code,
                **automodel_init_kwargs,
            )
            self.model = CustomQwen3ForCausalLM(auto_config)
            # self.model = AUTO_MODEL_CLS[automodel_cls].from_config(auto_config)
        else:
            # self.model = AUTO_MODEL_CLS[automodel_cls].from_pretrained(
            #     pretrained_model_name_or_path,
            #     trust_remote_code=trust_remote_code,
            #     **automodel_init_kwargs,
            # )
            self.model = CustomQwen3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **automodel_init_kwargs,
            )
            num_layers = (
                len(self.model.model.layers) if num_layers == -1 else num_layers
            )
            if keep_top_layers:
                self.model.model.layers = self.model.model.layers[-num_layers:]
            else:
                self.model.model.layers = self.model.model.layers[:num_layers]
        if hasattr(self.model, "set_esolm_backbone"):
            self.model.set_esolm_backbone(is_esolm_backbone)
        else:
            self.model.is_esolm_backbone = is_esolm_backbone

    @staticmethod
    def _normalize_attention_mask(
        attention_mask: torch.FloatTensor | BlockMask | dict[str, Any] | None,
    ) -> torch.FloatTensor | BlockMask | None:
        if isinstance(attention_mask, dict):
            if "full_attention" in attention_mask:
                return attention_mask["full_attention"]
            if "sliding_attention" in attention_mask:
                return attention_mask["sliding_attention"]
            raise ValueError(
                "Unsupported attention-mask mapping; expected `full_attention` "
                "or `sliding_attention`."
            )
        return attention_mask

    @staticmethod
    def _preprocess_attention_mask(attention_mask: torch.Tensor) -> torch.FloatTensor:
        min_dtype = torch.finfo(torch.float32).min
        return torch.where(attention_mask.bool(), 0.0, min_dtype).to(torch.float32)

    @staticmethod
    def _apply_attention_mask(mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        expanded = attention_mask.bool()
        return mask & expanded[:, None, None, :] & expanded[:, None, :, None]

    def _build_diffusion_attention_mask(
        self,
        xt: torch.LongTensor,
        attention_mask: torch.Tensor,
        mask_token_id: int,
        attn_mode: str,
    ) -> torch.FloatTensor:
        seq_len = xt.shape[1]
        q_idx = torch.arange(seq_len, device=xt.device)[None, :, None]
        kv_idx = torch.arange(seq_len, device=xt.device)[None, None, :]
        cutoffs = torch.sum(xt != mask_token_id, dim=1)
        if attn_mode == "causal":
            mask = esolm_causal_mask(q_idx, kv_idx)
        elif attn_mode == "bidirectional":
            mask = esolm_bidirectional_mask(q_idx, kv_idx)
        elif attn_mode == "mixed":
            mask = esolm_mixed_mask(q_idx, kv_idx, cutoffs)
        elif attn_mode == "mixed2":
            mask = esolm_mixed2_mask(q_idx, kv_idx, cutoffs)
        else:
            raise ValueError(f"Unsupported EsoLM attention mode: {attn_mode}")
        mask = self._apply_attention_mask(mask[:, None], attention_mask)
        return self._preprocess_attention_mask(mask)

    def _build_sequential_attention_mask(
        self,
        zt_and_x0: torch.LongTensor,
        attention_mask: torch.Tensor,
        mask_token_id: int,
        attn_mode: str,
    ) -> torch.FloatTensor:
        if zt_and_x0.shape[1] % 2 != 0:
            raise ValueError("EsoLM sequential inputs must contain two concatenated halves.")
        seq_len = zt_and_x0.shape[1] // 2
        zt = zt_and_x0[:, :seq_len]
        q_idx = torch.arange(seq_len * 2, device=zt_and_x0.device)[None, :, None]
        kv_idx = torch.arange(seq_len * 2, device=zt_and_x0.device)[None, None, :]
        base_mask = esolm_sequential_block_mask(q_idx=q_idx, kv_idx=kv_idx, seq_len=seq_len)
        if attn_mode == "causal":
            mask = base_mask[:, None]
        elif attn_mode == "mixed":
            cutoffs = torch.sum(zt != mask_token_id, dim=1)
            block_prefix_lm = (
                (seq_len <= q_idx)
                & (q_idx < seq_len + cutoffs[:, None, None])
                & (seq_len <= kv_idx)
                & (kv_idx < seq_len + cutoffs[:, None, None])
            )
            mask = (base_mask | block_prefix_lm)[:, None]
        else:
            raise ValueError(f"Unsupported EsoLM sequential attention mode: {attn_mode}")
        mask = self._apply_attention_mask(mask, attention_mask)
        return self._preprocess_attention_mask(mask)

    def _forward_esolm(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | BlockMask | dict[str, Any] | None = None,
        sort_index: torch.LongTensor | None = None,
        sequential_input: bool = False,
        diffusion_attn_mode: str = "bidirectional",
        sequential_attn_mode: str = "causal",
        mask_token_id: int | None = None,
        past_key_values: DynamicCache | None = None,
        sigma: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast | BaseModelOutputWithPast:
        del sigma
        if sort_index is None:
            raise ValueError("EsoLM forward requires `sort_index`.")
        if mask_token_id is None:
            raise ValueError("EsoLM forward requires `mask_token_id`.")
        attention_mask = self._normalize_attention_mask(attention_mask)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if sequential_input:
            position_ids = torch.cat([sort_index, sort_index], dim=1)
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim != 4:
                attention_mask = self._build_sequential_attention_mask(
                    zt_and_x0=input_ids,
                    attention_mask=attention_mask,
                    mask_token_id=mask_token_id,
                    attn_mode=sequential_attn_mode,
                )
            output_length = input_ids.shape[1] // 2
        else:
            position_ids = sort_index
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim != 4:
                attention_mask = self._build_diffusion_attention_mask(
                    xt=input_ids,
                    attention_mask=attention_mask,
                    mask_token_id=mask_token_id,
                    attn_mode=diffusion_attn_mode,
                )
            output_length = None

        model_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        if output_length is not None and getattr(model_output, "logits", None) is not None:
            model_output.logits = model_output.logits[:, :output_length]
        return model_output

    @staticmethod
    def _crop_past_key_values(
        past_key_values: DynamicCache | Any | None,
        keep_length: int,
    ) -> DynamicCache | Any | None:
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "crop"):
            past_key_values.crop(keep_length)
            return past_key_values
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            for i in range(len(past_key_values)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][
                    ..., :keep_length, :
                ]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][
                    ..., :keep_length, :
                ]
            return past_key_values
        return past_key_values

    def reset_kv_cache(self) -> None:
        self._esolm_past_key_values = None

    @staticmethod
    def reset_sorted_rotary_cache() -> None:
        return None

    @torch.no_grad()
    def forward_sample(
        self,
        zt: torch.LongTensor,
        sort_idx: torch.LongTensor,
        last_k_start: int | None = None,
        curr_k_start: int | None = None,
        curr_k_end: int | None = None,
        past_key_values: DynamicCache | None = None,
    ) -> CausalLMOutputWithPast:
        if last_k_start is None:
            last_k_start = 0
        if curr_k_start is None:
            curr_k_start = last_k_start
        if curr_k_end is None:
            curr_k_end = zt.shape[1]

        cache = past_key_values if past_key_values is not None else self._esolm_past_key_values
        if cache is None and DynamicCache is not None:
            cache = DynamicCache()
        cache = self._crop_past_key_values(cache, last_k_start)
        token_chunk = zt[:, last_k_start:curr_k_end]
        position_ids = sort_idx[:, last_k_start:curr_k_end]
        model_output = self.model(
            token_chunk,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        self._esolm_past_key_values = getattr(model_output, "past_key_values", cache)
        num_clean = curr_k_start - last_k_start
        if getattr(model_output, "logits", None) is not None:
            model_output.logits = model_output.logits[:, num_clean:, :]
        return model_output

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | BlockMask | None = None,
        position_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        fix_cache_length: bool = False,  # False for AR, True for diffusion models
        return_updated_cache=False,
        **kwargs,
    ) -> CausalLMOutputWithPast | BaseModelOutputWithPast:
        sort_index = kwargs.pop("sort_index", None)
        if sort_index is not None:
            return self._forward_esolm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sort_index=sort_index,
                past_key_values=past_key_values,
                **kwargs,
            )
        prev_cache_len = None
        attention_mask = self._normalize_attention_mask(attention_mask)
        if past_key_values is not None and fix_cache_length:
            prev_cache_len = [
                past_key_values[i][0].shape[-2]  # type: ignore
                for i in range(len(past_key_values))
            ]
        if self.use_causal_mask:
            attention_mask = None  # None --> enforces use of causal mask
        model_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            **kwargs,
        )
        if return_updated_cache:
            return model_output
        if (
            prev_cache_len is not None
            and model_output.get("past_key_values", None) is not None
        ):
            # DynamicCache extends along sequence dimension by default;
            # truncate back to original cache len
            for i, cache_len in enumerate(prev_cache_len):
                model_output.past_key_values.key_cache[i] = (
                    model_output.past_key_values.key_cache[i][..., :cache_len, :]
                )
                model_output.past_key_values.value_cache[i] = (
                    model_output.past_key_values.value_cache[i][..., :cache_len, :]
                )
        return model_output
