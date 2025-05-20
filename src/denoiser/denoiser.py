import inspect
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra.utils
import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import ModelOutput

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )
except ImportError:
    BlockMask, create_block_mask, flex_attention = None, None, None

# Add the local directory (enables hydra.utils.instantiate for local imports)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

# Local imports not used, but added here so that HF push_to_hub adds them to model repo
# noinspection PyUnresolvedReferences
from src.backbone.dit import DIT  # noqa: F401
from src.backbone.encoder_decoder import LLMasEncoderDecoder  # noqa: F401
from src.noise_schedule.noise_schedules import (  # noqa: F401
    CosineNoise,
    ExponentialNoise,
    LinearNoise,
    LogarithmicNoise,
)

# TODO: Consider remove loss weighting for MDLM / BD3LM


@dataclass
class DenoiserInput(OrderedDict):
    """Input to the denoiser model."""

    xt: Tensor  # (B, L) Tensor of token_ids
    x0: Optional[Tensor] = None  # (B, L) Tensor of token_ids (not used in gen.)
    # 1 / True indicates attention applies; 0 / False indicates ignore (e.g., padding)
    attention_mask: Optional[Tensor] = None
    # 1 / True indicates token is part of context; 0 / False indicates token should be
    # generated / predicted
    context_mask: Optional[Tensor] = None
    # 1 / True indicates token contributes to loss; 0 / False indicates otherwise;
    # for most use cases, this should be `= attention_mask & ~context_mask`
    tokens_mask: Optional[Tensor] = None  # (B, L)
    t: Optional[Tensor] = None  # (B,)
    alpha_t: Optional[Tensor] = None  # (B,) | (B, 1) | (B, 1, 1)
    alpha_t_prime: Optional[Tensor] = None  # (B,) | (B, 1) | (B, 1, 1)
    past_key_values: Optional[Tensor] = None  # (B, ctx_len, D)
    # Placeholder in case future experiments require different inputs
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossAndNllOutput(OrderedDict):
    """Loss output for denoiser models."""

    loss: Tensor
    nlls: Tensor


@dataclass
class DenoiserOutput(ModelOutput):
    """Output of the denoiser model."""

    denoiser_output: Tensor
    logits: Optional[Tensor] = None
    tokens_mask: Optional[Tensor] = None  # Which tokens contribute to loss
    past_key_values: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    nlls: Optional[Tensor] = None
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
        length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        noise_config: dict[str, Any] | None = None,
        sampler_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        time_conditioned_backbone: bool | None = None,
        keep_clean_bos: bool | None = None,  # Whether to enforce un-noised BOS token
        # Logits @ position i predicts token @ position i+1 (as in AR models)
        shift_logits: bool | None = None,
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
        self.length = length
        self.time_conditioned_backbone = time_conditioned_backbone
        self.keep_clean_bos = keep_clean_bos
        self.shift_logits = shift_logits


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
        self.sampler_config = (
            hydra.utils.instantiate(config.sampler_config)
            if config.sampler_config is not None
            else None
        )

    @abstractmethod
    def _prepare_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
        t: Tensor | None = None,
        past_key_values: Tensor | None = None,
    ) -> DenoiserInput:
        """
        Prepare inputs for the model.

        Parameters:
            input_ids (Tensor): Input tensor to the model.
            attention_mask (Optional[Tensor]): Attention mask for the model.
            t (Optional[Tensor]): Time step for the model.
            past_key_values (Optional[Tensor]): Past key values for the model.
        Returns:
            Denoiser inputs.
        """
        raise NotImplementedError("Denoiser subclasses must implement _prepare_inputs")

    @abstractmethod
    def _compute_loss(
        self, model_output: Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        """
        Compute the loss for the denoising model.

        Parameters:
            model_output (Tensor): Output tensor from self.forward.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            LossAndNllOutput: loss (Tensor) and nlls (Tensor).
        """
        raise NotImplementedError("Denoiser subclasses must implement _compute_loss")

    def _forward(
        self,
        backbone_output: Tensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> Tensor:
        """
        Forward pass for the denoiser model returns probabilities over denoised
        sequence.

        Some classes may need to override this method.

        Parameters:
            backbone_output (Tensor): Output tensor from the backbone model.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            Model outputs (Tensor).
        """
        return torch.log_softmax(backbone_output, dim=-1)

    def _backbone_forward(self, denoiser_inputs: DenoiserInput, **kwargs: Any):
        """Forward pass for the backbone model (should return logits).

        Some classes may need to override this method.

        Parameters:
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            Backbone output (Tensor).
        """
        if self.time_conditioned_backbone:
            return self.backbone(
                denoiser_inputs.xt,
                attention_mask=denoiser_inputs.attention_mask,
                noise=denoiser_inputs.alpha_t,
                **denoiser_inputs.backbone_kwargs,
                **kwargs,
            )
        return self.backbone(
            denoiser_inputs.xt,
            attention_mask=denoiser_inputs.attention_mask,
            **denoiser_inputs.backbone_kwargs,
            **kwargs,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
        t: Tensor | None = None,
        past_key_values: Tensor | None = None,
        compute_loss: bool | None = True,
        **kwargs,
    ) -> DenoiserOutput:
        """
        Perform a forward pass through the denoising model and
        (optionally) compute the loss.

        Parameters:
            input_ids (Tensor): Input tensor to the model.
            attention_mask (Optional[Tensor]): Attention mask for the model.
            context_mask (Optional[Tensor]): Indicator for context tokens.
            t (Optional[Tensor]): Denoising time step for the model.
            past_key_values (Optional[Tensor]): KV cache.
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

    def _sample_categorical(self, categorical_probs):
        """Helper function to sample from a categorical distribution."""
        categorical_probs = categorical_probs.to(torch.float64)
        if self.sampler_config.greedy:
            return categorical_probs.argmax(dim=-1)
        gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()).to(
            categorical_probs.dtype
        )
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def update_kv_cache(
        self,
        context: Tensor,
        past_key_values: DynamicCache | None = None,
        **kwargs: Any,
    ) -> DynamicCache:
        """
        Cache the key-value pairs for the context.
        Args:
            model (torch.nn.Module): The model to use for caching.
            context (Tensor): The context tensor.
            past_key_values (DynamicCache | None): Previous key-value pairs.
        Returns:
            DynamicCache: Cached key-value pairs.
        """
        context_input = self._prepare_inputs_inference(
            input_ids=None,
            context=context,
            past_key_values=past_key_values,
        )
        past_key_values = self._backbone_forward(
            context_input,
            use_cache=True,
            past_key_values=past_key_values,
        )
        return past_key_values

    @abstractmethod
    def generate(  # TODO: clean up signature and docstring
        self,
        max_seq_length: int,
        batch_size: int | None = None,
        device: str | None = None,
        context: Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, int]:
        """Generates sample from denoising model.
        # TODO: will need to enable infilling / starting from partially noised sequences

        Args:
            batch_size (int): Batch size.
            max_seq_length (int): Maximum sequence length.
            device (str, optional): Device to use for computation.
                Defaults to None, which will select cuda (if available).
            disable_cache (bool, optional): Whether to disable caching.
                Defaults to False.
            context (Tensor, optional): Optional prompt tensor
        Returns:
            Tensor: Generated samples of token_ids (B, L).
            int: Total number of function evaluations (NFEs).
        """
        raise NotImplementedError("Denoiser subclasses must implement generate")
