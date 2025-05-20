import copy
from typing import Any, Tuple

import torch
from torch import Tensor
from transformers import (
    PreTrainedTokenizer,
    StoppingCriteriaList,
)

from src.denoiser.denoiser import (
    Denoiser,
    DenoiserConfig,
    DenoiserInput,
    LossAndNllOutput,
)


class ARConfig(DenoiserConfig):
    """Configuration class for autoregressive (AR) models."""

    model_type = "ar"
    auto_map = {
        "AutoConfig": "denoiser.ARConfig",
        "AutoModel": "denoiser.AR",
        "AutoModelForCausalLM": "denoiser.AR",
    }

    def __init__(
        self,
        length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        noise_config: None = None,
        time_conditioned_backbone: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            length=length,
            backbone_config=backbone_config,
            noise_config=noise_config,
            tokenization_config=tokenization_config,
            time_conditioned_backbone=time_conditioned_backbone,
            **kwargs,
        )


class AR(Denoiser):
    """Denoiser class for autoregressive (AR) models."""

    config_class = ARConfig

    def __init__(
        self,
        config: ARConfig,
    ):
        super().__init__(config)

    def _prepare_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
        t: Tensor | None = None,
        past_key_values: Tensor | None = None,
    ) -> DenoiserInput:
        # Prepare inputs for autoregressive model
        labels = copy.deepcopy(input_ids[..., 1:])[..., None]
        input_ids = input_ids[..., :-1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.shape != input_ids.shape:
            attention_mask = attention_mask[..., :-1]
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)
        else:
            context_mask = context_mask[..., :-1]
        return DenoiserInput(
            xt=input_ids,
            x0=labels,
            attention_mask=attention_mask,
            context_mask=context_mask,
            tokens_mask=attention_mask * (1 - context_mask),
            past_key_values=past_key_values,
        )

    def _compute_loss(
        self, model_output: Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        # Shift labels
        loss = -torch.gather(model_output, -1, denoiser_inputs.x0).squeeze(-1)

        nlls = loss * denoiser_inputs.tokens_mask
        count = denoiser_inputs.tokens_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return LossAndNllOutput(loss=token_nll, nlls=nlls)

    def generate(  # TODO: clean up signature and docstring
        self,
        max_length: int | None = None,
        batch_size: int | None = None,
        disable_cache: bool | None = None,
        device: str | None = None,
        context: Tensor | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        disable_pbar: bool = False,
        **kwargs: Any,
    ) -> Tuple[Tensor, int]:
        len_penalty = kwargs.pop("len_penalty", 1.0)
        regulation_start = kwargs.pop("regulation_start", None)
        repetition_penalty = kwargs.pop("repetition_penalty", None)
        exponential_decay_length_penalty = (
            (regulation_start, len_penalty) if len_penalty != 1.0 else None
        )
        outputs = self.backbone.model.generate(
            input_ids=context,
            attention_mask=torch.ones_like(context),
            max_new_tokens=max_length - context.shape[-1],
            stopping_criteria=stopping_criteria,
            repetition_penalty=repetition_penalty,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            top_k=kwargs.pop("top_k", None),  # None implies greedy decoding
            **kwargs,
        )

        if tokenizer is not None:
            print(tokenizer.batch_decode(outputs))
        # Decode output
        return outputs, -1
