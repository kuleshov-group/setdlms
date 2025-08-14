import copy
from typing import Any, Dict, Tuple

import torch
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput

from src.denoiser.base import (
    Denoiser,
    DenoiserConfig,
    DenoiserInput,
    LossAndNllOutput,
)


class ARConfig(DenoiserConfig):
    """Configuration class for autoregressive (AR) models."""

    model_type = "ar"
    auto_map = {
        "AutoConfig": "ar.ARConfig",
        "AutoModel": "ar.AR",
        "AutoModelForCausalLM": "ar.AR",
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
        **kwargs,
    ):
        super().__init__(config, **kwargs)

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
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
        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask)
        return DenoiserInput(
            xt=input_ids,  # type: ignore
            x0=labels,  # type: ignore
            attention_mask=attention_mask,
            context_mask=context_mask,
            tokens_mask=tokens_mask,
            past_key_values=past_key_values,
        )

    def _prepare_inputs_inference(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        context: torch.LongTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        cache: Dict[str, Any] | None = None,
        **backbone_kwargs: Any,
    ) -> Tuple[DenoiserInput, Dict[str, Any]]:
        pass  # Not used

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        # Shift labels
        loss = -torch.gather(model_output, -1, denoiser_inputs.x0).squeeze(-1)

        nlls = loss * denoiser_inputs.tokens_mask
        count = denoiser_inputs.tokens_mask.sum(dim=-1)

        batch_nll = nlls.sum(dim=-1)
        token_nll = (batch_nll / count).mean()

        return LossAndNllOutput(loss=token_nll, nlls=nlls)  # type: ignore

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.LongTensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        disable_pbar: bool | None = None,  # not used; compatability w/other denoisers
        **kwargs,
    ) -> GenerateOutput | torch.LongTensor:
        outputs = self.backbone.model.generate(
            inputs=inputs,
            attention_mask=torch.ones_like(inputs),
            generation_config=generation_config,
            logits_processor=logits_processor,
            # TODO: debug: passing EOS stopping criteria generates EOS right away?
            # stopping_criteria=stopping_criteria,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            # TODO: Can we pass this in `generation_config`?
            # eos_token_id=None,  # Uncomment for t-put runs; prevents stopping at EOS
            **kwargs,
        )

        if tokenizer is not None:
            print(tokenizer.batch_decode(outputs))
        # Decode output
        return outputs
