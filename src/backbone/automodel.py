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
    """Simple wrapper class that enables using AutoModel initialized from scratch."""

    def __init__(
        self,
        automodel_cls: Literal[
            "AutoModel", "AutoModelForCausalLM", "AutoModelForMaskedLM"
        ],
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__()
        auto_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )
        self.model = AUTO_MODEL_CLS[automodel_cls].from_config(auto_config)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)


class AutoModelFromPreTrained(nn.Module):
    """Simple wrapper class that enables using AutoModel from pre-trained."""

    def __init__(
        self,
        automodel_cls: Literal[
            "AutoModel", "AutoModelForCausalLM", "AutoModelForMaskedLM"
        ],
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model = AUTO_MODEL_CLS[automodel_cls].from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
