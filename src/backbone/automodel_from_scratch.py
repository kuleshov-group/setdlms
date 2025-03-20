from typing import Literal

from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

AUTO_MODEL_CLS = {
    "AutoModel": AutoModel,
    "AutoModelForCausalLM": AutoModelForCausalLM,
    "AutoModelForMaskedLM": AutoModelForMaskedLM,
}


class AutoModelFromScratch(nn.Module):
    """Simple wrapper class that enables using AutoModels initialized from scratch."""

    def __init__(
        self,
        automodel_cls: Literal[
            "AutoModel", "AutoModelForCausalLM", "AutoModelForMaskedLM"
        ],
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super(AutoModelFromScratch, self).__init__()
        auto_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )
        self.model = AUTO_MODEL_CLS[automodel_cls].from_config(auto_config)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)
