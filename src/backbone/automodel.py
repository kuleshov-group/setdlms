from typing import Literal

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
        **automodel_init_kwargs,
    ):
        super().__init__()
        self.use_causal_mask = use_causal_mask
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
            self.model = AUTO_MODEL_CLS[automodel_cls].from_pretrained(
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
        prev_cache_len = None
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
            return BaseModelOutputWithPast(past_key_values=model_output.past_key_values)
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
