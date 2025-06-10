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
        keep_bottom_n_layers: int = -1,
        reinit_model: bool = False,
        **kwargs,
    ):
        super().__init__()
        if reinit_model:
            auto_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
            self.model = AUTO_MODEL_CLS[automodel_cls].from_config(auto_config)
        else:
            self.model = AUTO_MODEL_CLS[automodel_cls].from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        keep_bottom_n_layers = (
            len(self.model.model.layers)
            if keep_bottom_n_layers == -1
            else keep_bottom_n_layers
        )
        self.model.model.layers = self.model.model.layers[:keep_bottom_n_layers]

    def forward(
        self, input_ids: torch.LongTensor, return_past_key_values=False, **kwargs
    ) -> torch.FloatTensor | DynamicCache:
        if return_past_key_values:
            return self.model(input_ids, **kwargs).past_key_values
        return self.model(input_ids, **kwargs)
