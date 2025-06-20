from typing import Literal

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from src.backbone.custom_modeling_qwen3 import CustomQwen3ForCausalLM

AUTO_MODEL_CLS = {
    "AutoModel": AutoModel,
    "AutoModelForCausalLM": AutoModelForCausalLM,
    "AutoModelForMaskedLM": AutoModelForMaskedLM,
    "CustomQwen3ForCausalLM": CustomQwen3ForCausalLM,
}


class AutoModelFromPreTrained(nn.Module):
    """Simple wrapper class that enables using AutoModel from pre-trained."""

    def __init__(
        self,
        automodel_cls: Literal[
            "AutoModel",
            "AutoModelForCausalLM",
            "AutoModelForMaskedLM",
            "CustomQwen3ForCausalLM",
        ],
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        num_layers: int = -1,
        keep_top_layers: bool = False,
        reinit_model: bool = False,
        **automodel_init_kwargs,
    ):
        super().__init__()
        if reinit_model:
            auto_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                num_hidden_layers=num_layers,
                trust_remote_code=trust_remote_code,
                **automodel_init_kwargs,
            )
            self.model = AUTO_MODEL_CLS[automodel_cls].from_config(auto_config)
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
        self, input_ids: torch.LongTensor, return_updated_cache=False, **kwargs
    ) -> CausalLMOutputWithPast | BaseModelOutputWithPast:
        if return_updated_cache:
            return BaseModelOutputWithPast(
                past_key_values=self.model(input_ids, **kwargs).past_key_values
            )
        return self.model(input_ids, **kwargs)
