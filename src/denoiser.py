import copy
import inspect
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import hydra.utils
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
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
from src.backbone.encoder_decoder import LlamaAsEncoderDecoder  # noqa: F401
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

    xt: torch.Tensor  # (B, L) Tensor of token_ids
    x0: Optional[torch.Tensor] = None  # (B, L) Tensor of token_ids (not used in gen.)
    # 1 / True indicates attention applies; 0 / False indicates ignore (e.g., padding)
    attention_mask: Optional[torch.Tensor] = None
    # 1 / True indicates token is part of context; 0 / False indicates token should be
    # generated / predicted
    context_mask: Optional[torch.Tensor] = None
    # 1 / True indicates token contributes to loss; 0 / False indicates otherwise;
    # for most use cases, this should be `= attention_mask & ~context_mask`
    tokens_mask: Optional[torch.Tensor] = None  # (B, L)
    t: Optional[torch.Tensor] = None  # (B,)
    alpha_t: Optional[torch.Tensor] = None  # (B,) | (B, 1) | (B, 1, 1)
    alpha_t_prime: Optional[torch.Tensor] = None  # (B,) | (B, 1) | (B, 1, 1)
    past_key_values: Optional[torch.Tensor] = None  # (B, ctx_len, D)
    # Placeholder in case future experiments require different inputs
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossAndNllOutput(OrderedDict):
    """Loss output for denoiser models."""

    loss: torch.Tensor
    nlls: torch.Tensor


@dataclass
class DenoiserOutput(ModelOutput):
    """Output of the denoiser model."""

    denoiser_output: torch.Tensor
    logits: Optional[torch.Tensor] = None
    tokens_mask: Optional[torch.Tensor] = None  # Which tokens contribute to loss
    past_key_values: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    nlls: Optional[torch.Tensor] = None
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
        self.sampler = (
            hydra.utils.instantiate(config.sampler_config)
            if config.sampler_config is not None
            else None
        )

    @abstractmethod
    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
    ) -> DenoiserInput:
        """
        Prepare inputs for the model.

        Parameters:
            input_ids (torch.Tensor): Input tensor to the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the model.
            t (Optional[torch.Tensor]): Time step for the model.
            past_key_values (Optional[torch.Tensor]): Past key values for the model.
        Returns:
            Denoiser inputs.
        """
        raise NotImplementedError("Denoiser subclasses must implement _prepare_inputs")

    @abstractmethod
    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        """
        Compute the loss for the denoising model.

        Parameters:
            model_output (torch.Tensor): Output tensor from self.forward.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            LossAndNllOutput: loss (torch.Tensor) and nlls (torch.Tensor).
        """
        raise NotImplementedError("Denoiser subclasses must implement _compute_loss")

    def _forward(
        self,
        backbone_output: torch.Tensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the denoiser model returns probabilities over denoised
        sequence.

        Some classes may need to override this method.

        Parameters:
            backbone_output (torch.Tensor): Output tensor from the backbone model.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            Model outputs (torch.Tensor).
        """
        return torch.log_softmax(backbone_output, dim=-1)

    def _backbone_forward(self, denoiser_inputs: DenoiserInput, **kwargs: Any):
        """Forward pass for the backbone model (should return logits).

        Some classes may need to override this method.

        Parameters:
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            Backbone output (torch.Tensor).
        """
        if self.time_conditioned_backbone:
            return self.backbone(
                denoiser_inputs.xt,
                attention_mask=denoiser_inputs.attention_mask,
                noise=denoiser_inputs.alpha_t,
                past_key_values=denoiser_inputs.past_key_values,
                **denoiser_inputs.backbone_kwargs,
                **kwargs,
            )
        return self.backbone(
            denoiser_inputs.xt,
            attention_mask=denoiser_inputs.attention_mask,
            past_key_values=denoiser_inputs.past_key_values,
            **denoiser_inputs.backbone_kwargs,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        compute_loss: bool | None = True,
        **kwargs,
    ) -> DenoiserOutput:
        """
        Perform a forward pass through the denoising model and
        (optionally) compute the loss.

        Parameters:
            input_ids (torch.Tensor): Input tensor to the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the model.
            context_mask (Optional[torch.Tensor]): Indicator for context tokens.
            t (Optional[torch.Tensor]): Denoising time step for the model.
            past_key_values (Optional[torch.Tensor]): KV cache.
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
    def _sample_categorical(categorical_probs):
        """Helper function to sample from a categorical distribution."""
        categorical_probs = categorical_probs.to(torch.float64)
        gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()).to(
            categorical_probs.dtype
        )
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    @abstractmethod
    def generate(  # TODO: clean up signature and docstring
        self,
        max_seq_len: int,
        num_steps: int,
        nucleus_p: float = 1.0,
        batch_size: int | None = None,
        eps: float = 1e-5,
        device: str | None = None,
        disable_cache: bool = False,
        prompt: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generates sample from denoising model.
        # TODO: will need to enable infilling / starting from partially noised sequences

        Args:
            batch_size (int): Batch size.
            max_seq_len (int): Maximum sequence length.
            num_steps (int): Number of sampling steps.
            nucleus_p (float, optional): Nucleus sampling probability.
                Defaults to 1.0 (i.e., no nucleus sampling)
            eps (float, optional): Minimum value for t. Defaults to 1e-5.
            device (str, optional): Device to use for computation.
                Defaults to None, which will select cuda (if available).
            disable_cache (bool, optional): Whether to disable caching.
                Defaults to False.
            prompt (torch.Tensor, optional): Optional prompt tensor
        Returns:
            torch.Tensor: Generated samples of token_ids (B, L).
        """
        raise NotImplementedError


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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
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
        return DenoiserInput(
            xt=input_ids,
            x0=labels,
            attention_mask=attention_mask,
            context_mask=context_mask,
            tokens_mask=attention_mask * (1 - context_mask),
            past_key_values=past_key_values,
        )

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        # Shift labels
        loss = -torch.gather(model_output, -1, denoiser_inputs.x0).squeeze(-1)

        nlls = loss * denoiser_inputs.tokens_mask
        count = denoiser_inputs.tokens_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return LossAndNllOutput(loss=token_nll, nlls=nlls)

    def generate(
        self,
        max_seq_len: int,
        nucleus_p: float = 1.0,
        batch_size: int | None = None,
        device: str | None = None,
        disable_cache: bool = False,
        input_ids: torch.Tensor | None = None,
        input_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # TODO implement ar sampler
        input_attention_mask = (
            torch.ones((batch_size, 1), device=device)
            if input_ids is None
            else input_attention_mask
        )
        input_ids = (
            torch.ones((batch_size, 1), device=device) * self.bos_token_id
            if input_ids is None
            else input_ids
        )
        generated = torch.empty((input_ids.shape[0], max_seq_len), device=device)
        max_seq_len = max_seq_len - input_ids.shape[-1]
        past_key_values = None
        for i in range(max_seq_len):
            denoiser_output = self.forward(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                past_key_values=past_key_values,
                compute_loss=False,
            )
            past_key_values = denoiser_output.past_key_values
            log_probs = denoiser_output.denoiser_output
            if nucleus_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(
                    log_probs[:, -1, :], descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_probs, dim=-1), dim=-1
                )
                top_p_mask = cumulative_probs <= nucleus_p
                top_p_mask[..., 0] = True
                nucleus_probs = torch.zeros_like(log_probs[:, -1, :])
                nucleus_probs.scatter_(
                    -1,
                    sorted_indices,
                    torch.nn.functional.softmax(
                        sorted_probs * top_p_mask.float(), dim=-1
                    ),
                )
                log_probs[:, -1, :] = nucleus_probs.log()
            input_ids = self._sample_categorical(log_probs[:, -1, :])
            generated[:, i] = input_ids
        return generated


class D3PMConfig(DenoiserConfig):
    """Configuration class for D3PM models."""

    model_type = "d3pm"
    auto_map = {
        "AutoConfig": "denoiser.D3PMConfig",
        "AutoModel": "denoiser.D3PM",
        "AutoModelForMaskedLM": "denoiser.D3PM",
    }

    def __init__(
        self,
        T: int = 1000,
        diffusion_type: Literal["absorbing", "uniform"] = "absorbing",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diffusion_type = diffusion_type
        self.T = T


class D3PM(Denoiser):
    """Denoiser class for D3PM models.

    This class implements the Denoiser interface for D3PM models.
    """

    config_class = D3PMConfig

    def __init__(self, config: D3PMConfig):
        super().__init__(config)
        self.T = config.T
        self.diffusion_type = config.diffusion_type

    def _sample_q_xt(
        self,
        x0: torch.Tensor,
        alpha_t: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the pre-defined forward / noising process.

        Parameters:
            x0 (torch.Tensor): Signal / data sample;
                can potentially include context tokens.
            alpha_t (torch.Tensor): Amount of signal to retain.
            context_mask (torch.Tensor): Indicator of context tokens (to remain
                unchanged).
        """
        move_indices = torch.rand(*x0.shape, device=x0.device) < (1.0 - alpha_t)
        if self.diffusion_type == "absorbing":
            xt = torch.where(
                (move_indices * (1 - context_mask)).bool(), self.mask_token_id, x0
            )
            if self.config.keep_clean_bos:
                xt = torch.where(x0 == self.bos_token_id, x0, xt)
            return xt
        if self.diffusion_type == "uniform":
            xt = torch.randint(0, self.vocab_size, x0.shape, device=x0.device)
            xt = torch.where(context_mask.bool(), x0, xt)
            if self.config.keep_clean_bos:
                xt = torch.where(x0 == self.bos_token_id, x0, xt)
            return xt
        raise NotImplementedError(
            f"Diffusion type '{self.diffusion_type}' not implemented."
        )

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
    ):
        # Prepare inputs for D3PM model
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)

        if t is None:
            t = torch.rand(input_ids.shape[0], device=input_ids.device)
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]
        xt = self._sample_q_xt(
            x0=input_ids,
            alpha_t=alpha_t,
            context_mask=context_mask,
        )

        return DenoiserInput(
            xt=xt,
            x0=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            tokens_mask=attention_mask * (1 - context_mask),
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
        )

    def _prepare_inputs_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
    ):
        attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        alpha_t, _ = self.noise_schedule(t)
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            tokens_mask=attention_mask * (1 - context_mask),
            t=t,
            alpha_t=alpha_t,
            past_key_values=past_key_values,
        )

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        raise NotImplementedError

    def _sample_prior(self, device, batch_size, length):
        """Samples from prior / limiting distribution."""
        if self.diffusion_type == "absorbing_state":
            return self.mask_token_id * torch.ones(
                (batch_size, length), dtype=torch.int64, device=device
            )
        if self.diffusion_type == "uniform":
            return torch.randint(
                0,
                self.vocab_size,
                (batch_size, length),
                device=device,
                dtype=torch.int64,
            )
        raise NotImplementedError(
            f"Diffusion type '{self.diffusion_type}' not implemented."
        )

    def _compute_posterior(
        self,
        x: torch.Tensor,
        xt: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_s: torch.Tensor,
    ) -> torch.Tensor:
        """Computes posterior / approximate posterior q(x_s | x_t, x),
            where x represents clean sequence (as one-hots) or the output of the
            denoising model.

        Args:
            x (torch.Tensor): True (one-hot) / predicted clean signal (B, L, V).
            xt (torch.Tensor): Noised signal at time t (B, L).
            alpha_t (torch.Tensor): Noise schedule parameter at time t (B, 1, 1).
            alpha_s (torch.Tensor): Noise schedule parameter at time s (B, 1, 1).
        """
        if self.diffusion_type == "absorbing_state":
            q_xs = x * (alpha_s - alpha_t)
            q_xs[..., self.mask_token_id] = 1 - alpha_s[..., 0]
            q_xs /= 1 - alpha_t
            return q_xs

        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = torch.nn.functional.one_hot(x, self.vocab_size)
        limiting_distribution = torch.ones_like(xt_one_hot) / self.vocab_size
        if self.diffusion_type == "uniform":
            return (
                alpha_t * self.vocab_size * x * xt_one_hot
                + (alpha_ts - alpha_t) * xt_one_hot
                + d_alpha * x
                + (1 - alpha_ts) * (1 - alpha_s) * limiting_distribution
            ) / (
                alpha_t * self.vocab_size * torch.gather(x, -1, xt[..., None])
                + (1 - alpha_t)
            )
        raise NotImplementedError(
            f"Diffusion type {self.diffusion_type} not implemented."
        )

    def generate(
        self,
        batch_size: int,
        max_seq_len: int,
        num_steps: int,
        eps: float = 1e-5,
        device: str | None = None,
        disable_cache: bool = False,
        prompt: torch.Tensor | None = None,
        block_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        samples, NFEs = self.sampler(
            model=self,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            eps=eps,
            device=device,
            disable_cache=disable_cache,
            num_steps=num_steps,
            context=prompt,
            block_size=block_size,
        )
        if prompt is not None:
            samples = torch.cat([prompt, samples], dim=1)
        return samples, NFEs


class MDLMConfig(D3PMConfig):
    """Configuration class for MDLM models."""

    model_type = "mdlm"
    auto_map = {
        "AutoConfig": "denoiser.MDLMConfig",
        "AutoModel": "denoiser.MDLM",
        "AutoModelForMaskedLM": "denoiser.MDLM",
    }


class MDLM(D3PM):
    """Denoiser class for MDLM models."""

    config_class = MDLMConfig

    def __init__(self, config: MDLMConfig):
        super().__init__(config)
        self.neg_infinity = -1e12

    def _forward(
        self, backbone_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs
    ) -> torch.Tensor:
        if self.config.shift_logits:
            backbone_output = backbone_output[:, :-1, ...]
        # Zero-mask probability
        mask = (
            torch.arange(backbone_output.shape[-1], device=backbone_output.device)
            == self.mask_token_id
        ).view(1, 1, -1)  # unsqueeze for broadcast to (batch, seq_len, vocab_size)
        log_probs = torch.where(
            mask, backbone_output + self.neg_infinity, backbone_output
        )
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
        # Copy-over unmasked: For the log_probs of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        xt = denoiser_inputs.xt
        if self.config.shift_logits:
            xt = xt[..., 1:]
        unmasked_indices = xt != self.mask_token_id
        log_probs[unmasked_indices] = self.neg_infinity
        log_probs[unmasked_indices, xt[unmasked_indices]] = 0
        return log_probs

    def _forward_inference(
        self, backbone_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs
    ) -> torch.Tensor:
        # Zero-mask probability
        mask = (
            torch.arange(backbone_output.shape[-1], device=backbone_output.device)
            == self.mask_token_id
        ).view(1, 1, -1)  # unsqueeze for broadcast to (batch, seq_len, vocab_size)
        log_probs = torch.where(
            mask, backbone_output + self.neg_infinity, backbone_output
        )
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
        # Copy-over unmasked: For the log_probs of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        xt = denoiser_inputs.xt
        if self.config.shift_logits:
            # only apply carry-over for tokens except the last
            # (the last token predicts a token not in the context)
            xt = F.pad(xt[..., 1:], (0, 1), value=self.mask_token_id)
        unmasked_indices = xt != self.mask_token_id
        log_probs[unmasked_indices] = self.neg_infinity
        log_probs[unmasked_indices, xt[unmasked_indices]] = 0
        log_probs = log_probs[~denoiser_inputs.context_mask.bool()].view(
            log_probs.shape[0], -1, log_probs.shape[-1]
        )
        return log_probs

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        if self.config.shift_logits:
            denoiser_inputs.x0 = denoiser_inputs.x0[..., 1:]
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask[..., 1:]
            if denoiser_inputs.t.ndim > 1:
                denoiser_inputs.alpha_t = denoiser_inputs.alpha_t[..., 1:]
                denoiser_inputs.alpha_t_prime = denoiser_inputs.alpha_t_prime[..., 1:]

        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=denoiser_inputs.x0[:, :, None]
        ).squeeze(-1)

        loss = (
            log_p_theta * denoiser_inputs.alpha_t_prime / (1 - denoiser_inputs.alpha_t)
        )
        nlls = loss * denoiser_inputs.tokens_mask
        count = denoiser_inputs.tokens_mask.sum()
        batch_nll = nlls.sum()
        token_nll = batch_nll / count
        return LossAndNllOutput(loss=token_nll, nlls=nlls)


class BD3LMConfig(MDLMConfig):
    """Configuration class for BD3LM models."""

    model_type = "bd3lm"
    auto_map = {
        "AutoConfig": "denoiser.BD3LMConfig",
        "AutoModel": "denoiser.BD3LM",
        "AutoModelForMaskedLM": "denoiser.BD3LM",
    }

    def __init__(
        self,
        block_size: int | None = None,
        attn_backend: str = "sdpa",
        backbone_is_decoder_only: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.attn_backend = attn_backend
        # Determines whether inputs / masks are concatenated or separate for enc-dec
        self.backbone_is_decoder_only = backbone_is_decoder_only


class BD3LM(MDLM):
    """Denoiser class for BD3LM models."""

    config_class = BD3LMConfig

    def __init__(self, config: BD3LMConfig):
        super().__init__(config)
        self._create_static_mask()

    @staticmethod
    def _encoder_block_mask(
        q_idx: torch.Tensor,  # (L, 1)
        kv_idx: torch.Tensor,  # (1, L)
        b: int | None = None,
        h: int | None = None,
        block_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q_idx (torch.Tensor): Query indices.
            kv_idx (torch.Tensor): Key indices
            b (Optional: int): batch size
            h (Optional: int): number of heads
            block_size (Optional: int): Defines the block structure.

        Returns:
            Encoder block-causal attention mask.
        """

        del b, h

        # Compute block indices
        block_q = q_idx // block_size
        block_kv = kv_idx // block_size

        return block_q >= block_kv

    @staticmethod
    def _decoder_block_mask(
        q_idx: torch.Tensor,  # (L, 1)
        kv_idx: torch.Tensor,  # (1, 2 * L)
        b: int | None = None,  # needed for compat. with flex_attention
        h: int | None = None,  # needed for compat. with flex_attention
        block_size: int | None = None,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        del b, h

        # Indicate whether token belongs to xt or x0:
        x0_flag_q = (q_idx >= seq_len).bool()
        x0_flag_kv = (kv_idx >= seq_len).bool()

        # Compute block indices
        block_q = torch.where(
            x0_flag_q, (q_idx - seq_len) // block_size, q_idx // block_size
        )
        block_kv = torch.where(
            x0_flag_kv, (kv_idx - seq_len) // block_size, kv_idx // block_size
        )

        # **1. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

        # **2. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q

        # **3. Combine Masks **
        return block_diagonal | offset_block_causal

    @staticmethod
    def _block_mask(
        q_idx: torch.Tensor,  # (B, 2 * L)
        kv_idx: torch.Tensor | None = None,  # needed for compat. with flex_attention
        b: int | None = None,  # needed for compat. with flex_attention
        h: int | None = None,  # needed for compat. with flex_attention
        block_size: int | None = None,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """
        Constructs the specialized block diffusion attention mask for training
        composed of three masks:
        - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
        - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
        - **Block Causal Mask (M_BC)**: Attention to update x0

        Args:
            q_idx (torch.Tensor): Query indices.
            kv_idx (Optional: torch.Tensor): Key indices
            b (Optional: int): batch size
            h (Optional: int): number of heads
            block_size (Optional: int): Defines the block structure.
            seq_len (Optional: int): Total sequence length.

        Returns:
            Attention mask.
        """

        del b, h, kv_idx

        # Indicate whether token belongs to xt or x0:
        #   xt, x0 are concatenated to create 2N x 2N tensor
        x0_flag_q = (q_idx >= seq_len).bool()
        x0_flag_kv = x0_flag_q

        # Compute block indices
        block_q = torch.where(
            x0_flag_q, (q_idx - seq_len) // block_size, q_idx // block_size
        )
        block_kv = block_q

        # **1. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

        # **2. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q

        # **3. Block-Causal Mask (M_BC) **
        block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

        # **4. Combine Masks **
        return block_diagonal | offset_block_causal | block_causal

    def _create_static_mask(self) -> None:
        if self.config.backbone_is_decoder_only:
            if self.config.attn_backend == "flex_attention":
                static_mask = create_block_mask(
                    partial(
                        self._block_mask,
                        block_size=self.config.block_size,
                        seq_len=self.config.length,
                    ),
                    B=None,
                    H=None,
                    Q_LEN=self.config.length * 2,
                    KV_LEN=self.config.length * 2,
                )
            else:
                static_mask = self._block_mask(
                    q_idx=torch.arange(self.config.length * 2)[:, None],
                    block_size=self.config.block_size,
                    seq_len=self.config.length,
                )
            self.register_buffer(
                "static_attention_mask",
                static_mask,
            )
        else:
            if self.config.attn_backend == "flex_attention":
                encoder_static_mask = create_block_mask(
                    partial(
                        self._encoder_block_mask,
                        block_size=self.config.block_size,
                    ),
                    B=None,
                    H=None,
                    Q_LEN=self.config.length,
                    KV_LEN=self.config.length,
                )
                decoder_static_mask = create_block_mask(
                    partial(
                        self._decoder_block_mask,
                        block_size=self.config.block_size,
                        seq_len=self.config.length,
                    ),
                    B=None,
                    H=None,
                    Q_LEN=self.config.length,
                    KV_LEN=self.config.length * 2,
                )
            else:
                encoder_static_mask = self._encoder_block_mask(
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length)[None, :],
                    block_size=self.config.block_size,
                )
                decoder_static_mask = self._decoder_block_mask(
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length * 2)[None, :],
                    block_size=self.config.block_size,
                    seq_len=self.config.length,
                )
            self.register_buffer(
                "encoder_static_attention_mask",
                encoder_static_mask,
            )
            self.register_buffer(
                "decoder_static_attention_mask",
                decoder_static_mask,
            )

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)

        if t is None:
            t = torch.rand(
                input_ids.shape[0],
                input_ids.shape[1] // self.config.block_size,
                device=input_ids.device,
            ).repeat_interleave(self.config.block_size, dim=-1)
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]
        xt = self._sample_q_xt(x0=input_ids, alpha_t=alpha_t, context_mask=context_mask)

        if self.config.backbone_is_decoder_only:
            # TODO: check attention mask is correct
            decoder_attention_mask = (
                self.decoder_static_attention_mask[None, ...]
                & attention_mask.repeat(1, 2)[:, None, :]
                & attention_mask[..., None]
            )
            return DenoiserInput(
                xt=xt,
                x0=input_ids,
                attention_mask=decoder_attention_mask,
                tokens_mask=attention_mask * (1 - context_mask),
                t=t,
                alpha_t=alpha_t,
                alpha_t_prime=alpha_t_prime,
            )
        else:
            decoder_attention_mask = (
                self.decoder_static_attention_mask[None, ...]
                & attention_mask.repeat(1, 2)[:, None, :]
                & attention_mask[..., None]
            )
            encoder_attention_mask = (
                self.encoder_static_attention_mask[None, ...]
                & attention_mask[:, None, :]
                & attention_mask[..., None]
            )
            return DenoiserInput(
                xt=xt,
                x0=input_ids,
                attention_mask=decoder_attention_mask,
                tokens_mask=attention_mask * (1 - context_mask),
                t=t,
                alpha_t=alpha_t,
                alpha_t_prime=alpha_t_prime,
                backbone_kwargs={
                    "encoder_input_ids": input_ids,
                    "encoder_attention_mask": encoder_attention_mask,
                },
            )

    def _prepare_inputs_inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
    ):
        xt = input_ids[~context_mask.bool()].view(input_ids.shape[0], -1)
        x0 = input_ids[context_mask.bool()].view(input_ids.shape[0], -1)
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        if self.config.backbone_is_decoder_only:
            raise NotImplementedError(
                "Inference for decoder-only BD3LM is not implemented yet."
            )
        else:
            # full attention to xt and context
            decoder_attention_mask = torch.ones(
                (xt.shape[0], xt.shape[1], context_mask.shape[1]),
                device=xt.device,
                dtype=attention_mask.dtype,
            )
            # block-causal attention within x0
            encoder_attention_mask = (
                self.encoder_static_attention_mask[None, : x0.shape[1], : x0.shape[1]]
            ).to(attention_mask.dtype)

            return DenoiserInput(
                xt=xt,
                x0=x0,
                context_mask=context_mask,
                attention_mask=decoder_attention_mask,
                tokens_mask=attention_mask,
                t=t,
                alpha_t=alpha_t,
                alpha_t_prime=alpha_t_prime,
                backbone_kwargs={
                    "encoder_input_ids": x0,
                    "encoder_attention_mask": encoder_attention_mask,
                    "position_ids": torch.arange(
                        context_mask.shape[1], device=xt.device
                    ).expand((xt.shape[0], -1)),
                },
            )

    def _forward_inference(
        self, backbone_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs
    ) -> torch.Tensor:
        # Zero-mask probability
        mask = (
            torch.arange(backbone_output.shape[-1], device=backbone_output.device)
            == self.mask_token_id
        ).view(1, 1, -1)  # unsqueeze for broadcast to (batch, seq_len, vocab_size)
        log_probs = torch.where(
            mask, backbone_output + self.neg_infinity, backbone_output
        )
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
        # Copy-over unmasked: For the log_probs of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        xt = denoiser_inputs.xt
        if self.config.shift_logits:
            # only apply carry-over for tokens except the last
            # (the last token predicts a token not in the context)
            xt = F.pad(xt[..., 1:], (0, 1), value=self.mask_token_id)
        unmasked_indices = xt != self.mask_token_id
        log_probs[unmasked_indices] = self.neg_infinity
        log_probs[unmasked_indices, xt[unmasked_indices]] = 0
        if self.config.backbone_is_decoder_only:
            log_probs = log_probs[~denoiser_inputs.context_mask.bool()].view(
                log_probs.shape[0], -1, log_probs.shape[-1]
            )
        return log_probs


# TODO
# class UDLM(D3PM):


# TODO
# class SEDD(Denoiser):


# TODO
# class DFM(Denoiser):
