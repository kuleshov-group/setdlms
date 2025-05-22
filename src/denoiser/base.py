import inspect
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import hydra.utils
import torch
from transformers import (
    AutoTokenizer,
    DynamicCache,
    GenerationConfig,
    LogitsProcessorList,
    PretrainedConfig,
    PreTrainedModel,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import ModelOutput

# Add the local directory (enables hydra.utils.instantiate for local imports)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

# Local imports not used, but added here so that HF push_to_hub adds them to model repo
# noinspection PyUnresolvedReferences
from src.backbone.automodel import AutoModelFromPreTrained  # noqa: F401
from src.backbone.encoder_decoder import LLMasEncoderDecoder  # noqa: F401
from src.noise_schedule.noise_schedules import (  # noqa: F401
    CosineNoise,
    ExponentialNoise,
    LinearNoise,
    LogarithmicNoise,
)


@dataclass
class DenoiserInput(OrderedDict):
    """Input to the denoiser model."""

    xt: torch.LongTensor  # (B, L) token_ids
    x0: Optional[torch.LongTensor] = None  # (B, L) token_ids (not used in gen.)
    # 1 / True indicates attention applies; 0 / False indicates ignore (e.g., padding)
    attention_mask: Optional[torch.FloatTensor] = None
    # 1 / True indicates token is part of context; 0 / False indicates token should be
    # generated / predicted
    context_mask: Optional[torch.FloatTensor] = None
    # 1 / True indicates token contributes to loss; 0 / False indicates otherwise;
    # for most use cases, this should be `= attention_mask & ~context_mask`
    tokens_mask: Optional[torch.FloatTensor] = None  # (B, L)
    t: Optional[torch.FloatTensor] = None  # (B,)
    alpha_t: Optional[torch.FloatTensor] = None  # (B,) | (B, 1) | (B, 1, 1)
    alpha_t_prime: Optional[torch.FloatTensor] = None  # (B,) | (B, 1) | (B, 1, 1)
    past_key_values: Optional[torch.FloatTensor] = None  # (B, ctx_len, D)
    # Placeholder in case future experiments require different inputs
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossAndNllOutput(OrderedDict):
    """Loss output for denoiser models."""

    loss: torch.FloatTensor
    nlls: torch.FloatTensor


@dataclass
class DenoiserOutput(ModelOutput):
    """Output of the denoiser model."""

    denoiser_output: torch.FloatTensor
    logits: Optional[torch.FloatTensor] = None
    tokens_mask: Optional[torch.FloatTensor] = None  # Which tokens contribute to loss
    past_key_values: Optional[Cache] = None
    loss: Optional[torch.FloatTensor] = None
    nlls: Optional[torch.FloatTensor] = None
    # Placeholder in case future models produce different outputs
    output_kwargs: Optional[Dict[str, Any]] = None


class DenoiserConfig(PretrainedConfig):
    """Configuration class for Denoiser models.

    This class is used to initialize the model and contains all the necessary
    parameters for the model's architecture.
    """

    model_type = "denoiser"

    def __init__(
        self,
        max_length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        noise_config: dict[str, Any] | None = None,
        sampler_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        time_conditioned_backbone: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for v in [
            "vocab_size",
            "mask_token_id",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "pad_vocab_size_multiple",
        ]:
            if tokenization_config is not None and (
                getattr(self, v, None) is None or v in tokenization_config
            ):
                setattr(self, v, tokenization_config.get(v, None))
            else:
                setattr(self, v, None)
        self.backbone_config = backbone_config
        self.noise_config = noise_config
        self.sampler_config = sampler_config
        self.max_length = max_length
        self.time_conditioned_backbone = time_conditioned_backbone


class Denoiser(ABC, PreTrainedModel):
    """Abstract base class for denoising models.

    This class defines the interface for AR, Diffusion, and Flow-based parametrizations.
    """

    config_class = DenoiserConfig

    def __init__(
        self,
        config: DenoiserConfig,
    ):
        """
        Initialize the Denoiser with a configuration and optional dataset type.

        Parameters:
            config (Any): Configuration object for the model.
        """
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.backbone = hydra.utils.instantiate(config.backbone_config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            trust_remote_code=True,
        )
        self.noise_schedule = (
            hydra.utils.instantiate(config.noise_config)
            if config.noise_config is not None
            else None
        )
        self.time_conditioned_backbone = (
            config.time_conditioned_backbone
            if config.time_conditioned_backbone is not None
            else "noise" in inspect.getfullargspec(self.backbone.forward).args
        )

    @abstractmethod
    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
    ) -> DenoiserInput:
        """
        Prepare inputs for the model.

        Parameters:
            input_ids (LongTensor): Input tensor to the model.
            attention_mask (Optional[FloatTensor]): Attention mask for the model.
            t (Optional[FloatTensor]): Time step for the model.
            past_key_values (Optional[Cache]): Past key values for the model.
        Returns:
            Denoiser inputs.
        """
        raise NotImplementedError("Denoiser subclasses must implement _prepare_inputs")

    @abstractmethod
    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        """
        Compute the loss for the denoising model.

        Parameters:
            model_output (FloatTensor): Output tensor from self.forward.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            LossAndNllOutput: loss (FloatTensor) and nlls (FloatTensor).
        """
        raise NotImplementedError("Denoiser subclasses must implement _compute_loss")

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        """
        Forward pass for the denoiser model returns probabilities over denoised
        sequence.

        Some classes may need to override this method.

        Parameters:
            backbone_output (FloatTensor): Output tensor from the backbone model.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            Model outputs (FloatTensor).
        """
        return torch.log_softmax(backbone_output, dim=-1)  # type: ignore

    def _backbone_forward(
        self,
        denoiser_inputs: DenoiserInput,
        return_past_key_values: bool = False,
        **backbone_kwargs: Any,
    ) -> torch.FloatTensor | Cache:
        """Forward pass for the backbone model (should return logits).

        Some classes may need to override this method.

        Parameters:
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.
            return_past_key_values (bool): If True, return past_key_values instead of
                logits.

        Returns:
            Backbone output (logits) or cache.
        """
        if self.time_conditioned_backbone:
            return self.backbone(
                denoiser_inputs.xt,
                attention_mask=denoiser_inputs.attention_mask,
                noise=denoiser_inputs.alpha_t,
                return_past_key_values=return_past_key_values,
                **denoiser_inputs.backbone_kwargs,
                **backbone_kwargs,
            )
        return self.backbone(
            denoiser_inputs.xt,
            attention_mask=denoiser_inputs.attention_mask,
            return_past_key_values=return_past_key_values,
            **denoiser_inputs.backbone_kwargs,
            **backbone_kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        compute_loss: bool | None = True,
        **kwargs,
    ) -> DenoiserOutput:
        """
        Perform a forward pass through the denoising model and
        (optionally) compute the loss.

        Parameters:
            input_ids (LongTensor): Input tensor to the model.
            attention_mask (Optional[FloatTensor]): Attention mask for the model.
            context_mask (Optional[FloatTensor]): Indicator for context tokens.
            t (Optional[FloatTensor]): Denoising time step for the model.
            past_key_values (Optional[Cache]): KV cache.
            compute_loss (Optional[bool]): Flag to compute loss.

        Returns:
            DenoiserOutput
        """
        denoiser_inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            t=t,
        )

        backbone_output = self._backbone_forward(denoiser_inputs, **kwargs)
        new_past_key_values = getattr(backbone_output, "past_key_values", None)
        if hasattr(backbone_output, "logits"):
            backbone_output = backbone_output.logits
        denoiser_output = self._forward(
            backbone_output,
            denoiser_inputs,
            **kwargs,
        )

        if compute_loss:
            loss_and_nll = self._compute_loss(
                model_output=denoiser_output, denoiser_inputs=denoiser_inputs, **kwargs
            )
            loss = loss_and_nll.loss
            nlls = loss_and_nll.nlls
        else:
            loss, nlls = None, None

        return DenoiserOutput(
            denoiser_output=denoiser_output,
            logits=backbone_output,
            past_key_values=new_past_key_values,
            tokens_mask=denoiser_inputs.tokens_mask,
            loss=loss,
            nlls=nlls,
        )

    @staticmethod
    def _sample_categorical(categorical_probs, do_sample=True):
        """Helper function to sample from a categorical distribution."""
        # TODO: for greedy, can we skip fp64 casting?
        categorical_probs = categorical_probs.to(torch.float64)
        if not do_sample:
            return categorical_probs.argmax(dim=-1)
        gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()).to(
            categorical_probs.dtype
        )
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def _prepare_inputs_inference(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        context: torch.LongTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> DenoiserInput:
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        if context is not None:
            if input_ids is not None:
                if context_mask is None:
                    context_mask = torch.cat(
                        [torch.ones_like(context), torch.zeros_like(input_ids)], dim=-1
                    )
                input_ids = torch.cat([context, input_ids], dim=-1)
            else:
                input_ids = context
                context_mask = torch.ones_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
        )

    @staticmethod
    def _get_past_key_values_seq_length(past_key_values: DynamicCache):
        seq_length = 0
        for i in range(len(past_key_values)):
            if past_key_values[i][0].shape[0] > 0:  # type: ignore
                seq_length = max(
                    past_key_values[i][0].shape[-2],  # type: ignore
                    seq_length,
                )
        return seq_length

    def update_past_key_values(
        self,
        inputs: torch.LongTensor,
        past_key_values: Cache | None = None,
        **backbone_kwargs: Any,
    ) -> DynamicCache:
        """
        Cache the key-value pairs for the context.
        Args:
            inputs (torch.LongTensor): The context tensor.
            past_key_values (DynamicCache | None): Previous key-value cache.
        Returns:
            DynamicCache: Cached key-value pairs.
        """
        context_input = self._prepare_inputs_inference(
            input_ids=inputs,
            past_key_values=past_key_values,
        )
        past_key_values = self._backbone_forward(
            context_input,
            use_cache=True,
            past_key_values=past_key_values,
            return_past_key_values=True,
            **backbone_kwargs,
        )
        return past_key_values

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
        **kwargs: Any,
    ) -> GenerateOutput | torch.LongTensor:
        """Generates sample from denoising model.
        Follows signature of transformers.GenerationMixin.
        """
        raise NotImplementedError("Denoiser subclasses must implement generate")
