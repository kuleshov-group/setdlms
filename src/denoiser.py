import copy
import inspect
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import hydra.utils
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
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

    def update_kv_cache(
        self,
        context: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        **kwargs: Any,
    ) -> DynamicCache:
        """
        Cache the key-value pairs for the context.
        Args:
            model (torch.nn.Module): The model to use for caching.
            context (torch.Tensor): The context tensor.
            past_key_values (DynamicCache | None): Previous key-value pairs.
        Returns:
            DynamicCache: Cached key-value pairs.
        """
        context_input = self._prepare_inputs_inference(
            input_ids=context,
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
        max_seq_len: int,
        batch_size: int | None = None,
        device: str | None = None,
        context: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, int]:
        """Generates sample from denoising model.
        # TODO: will need to enable infilling / starting from partially noised sequences

        Args:
            batch_size (int): Batch size.
            max_seq_len (int): Maximum sequence length.
            device (str, optional): Device to use for computation.
                Defaults to None, which will select cuda (if available).
            disable_cache (bool, optional): Whether to disable caching.
                Defaults to False.
            context (torch.Tensor, optional): Optional prompt tensor
        Returns:
            torch.Tensor: Generated samples of token_ids (B, L).
            int: Total number of function evaluations (NFEs).
        """
        raise NotImplementedError("Denoiser subclasses must implement generate")


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
    ) -> Tuple[torch.Tensor, int]:
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
                xt[..., 0] = x0[..., 0]
            return xt
        if self.diffusion_type == "uniform":
            xt = torch.randint(0, self.vocab_size, x0.shape, device=x0.device)
            xt = torch.where(context_mask.bool(), x0, xt)
            if self.config.keep_clean_bos:
                xt[..., 0] = x0[..., 0]
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
        past_key_values: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        raise NotImplementedError

    def _sample_prior(self, device, batch_size, length):
        """Samples from prior / limiting distribution."""
        if self.diffusion_type == "absorbing":
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
        if self.diffusion_type == "absorbing":
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

    def _sample_generation_timesteps(
        self, max_seq_len: int | None = None, device: str | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """
        Sample timesteps for the diffusion process.
        Args:
            eps (float): Small value to avoid division by zero.
            num_steps (int): Number of timesteps to sample.
            device (str | None): Device to use for sampling.
        Returns:
            torch.Tensor: Sampled timesteps.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.sampler_config.first_hitting:
            if max_seq_len is None:
                raise ValueError("max_seq_len must be provided for first hitting.")
            timesteps = torch.tensor([1.0])
            for i in range(max_seq_len, 0, -1):
                u = torch.rand(1)
                next_t = timesteps[-1] * u ** (1 / i)
                timesteps = torch.cat((timesteps, next_t), dim=0)
            return timesteps[1:].to(device)
        timesteps = torch.linspace(
            1,
            self.sampler_config.min_t,
            self.sampler_config.num_steps + 1,
            device=device,
        )
        return timesteps

    def _logit_transform(self, logits: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Transform logits using various techniques.
        Args:
            logits (torch.Tensor): Logits to transform.
        Returns:
            torch.Tensor: Transformed logits.
        """
        if self.sampler_config.top_p < 1.0:
            p = self.sampler_config.top_p
            sorted_probs, sorted_indices = logits.sort(dim=-1, descending=True)
            cum_probs = sorted_probs.cumsum(dim=-1)
            nucleus_mask = cum_probs <= p
            nucleus_mask[..., 0] = 1
            sorted_probs = sorted_probs * nucleus_mask
            logits.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
            logits /= logits.sum(-1, keepdim=True)
        return logits

    def _maybe_remask(
        self,
        xs: torch.Tensor,
        q_xs: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Remask the sampled sequence based on different strategies.
        Args:
            xs (torch.Tensor): Sampled sequence.
            q_xs (torch.Tensor): Posterior distribution.
            xt (torch.Tensor): Masked sequence.
            mask_token_id (int): Mask token ID.
        Returns:
            torch.Tensor: Remasked tokens.
        """
        # TODO implement remdm
        if self.config.shift_logits:
            xt = xt[:, 1:]

        if self.sampler_config.first_hitting:
            # uniformly select an index (among masked tokens)
            num_masked = (xt == self.mask_token_id).sum(-1)

            if self.sampler_config.low_confidence_remasking:
                # select the index with the highest confidence
                xs_q = q_xs.gather(-1, xs[..., None]).squeeze(-1)
                xs_q[xt != self.mask_token_id] = 0
                ind = xs_q.argmax(dim=-1)
            else:
                ind = torch.randint(0, num_masked.item(), (xs.shape[0],))
                ind = (xt == self.mask_token_id).nonzero()[ind, 1]

            unmask_flag = torch.arange(xt.shape[-1], device=xt.device) == ind[None, :]
            # if a token is already unmasked, don't apply remasking
            unmask_flag = unmask_flag | (xt != self.mask_token_id)

            # remask tokens not selected
            xs[~unmask_flag] = self.mask_token_id

        return xs

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

        if self.diffusion_type == "absorbing":
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

    def _generate_unconditional(  # TODO add CBG and CFG generation
        self,
        alpha_t: torch.Tensor,
        alpha_s: torch.Tensor,
        denoiser_inputs: DenoiserInput | None = None,
        cache: Dict[str, torch.Tensor] | None = None,
        past_key_values: DynamicCache | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if cache is None:
            # pad context with masks to match the training context
            # (improves mdlm quality)
            pad_len = 0
            if (
                self.sampler_config.pad_context
                and denoiser_inputs.xt.shape[-1] < self.config.length
            ):
                # pad with masks
                pad_len = self.config.length - denoiser_inputs.xt.shape[-1]
                denoiser_inputs.xt = F.pad(
                    denoiser_inputs.xt,
                    pad=(0, pad_len),
                    mode="constant",
                    value=self.mask_token_id,  # TODO could also use pad, check
                )
                denoiser_inputs.backbone_kwargs["position_ids"] = torch.arange(
                    denoiser_inputs.xt.shape[-1], device=denoiser_inputs.xt.device
                )[None, :]
            backbone_output = self._backbone_forward(
                denoiser_inputs,
                past_key_values=past_key_values,
            )
            # remove padding
            if pad_len > 0:
                backbone_output = backbone_output[:, :-pad_len]
                denoiser_inputs.xt = denoiser_inputs.xt[:, :-pad_len]
                denoiser_inputs.backbone_kwargs["position_ids"] = torch.arange(
                    denoiser_inputs.xt.shape[-1], device=denoiser_inputs.xt.device
                )[None, :]
            if isinstance(backbone_output, ModelOutput) and hasattr(
                backbone_output, "logits"
            ):
                backbone_output = backbone_output.logits
            log_x_theta = self._forward(
                backbone_output,
                denoiser_inputs,
            )  # should be the log(x_\theta) with the shape of (B, Seq, Vocab)
            x_theta = log_x_theta.exp()
            # TODO add from old implementation
            # x_theta = self._logit_transform(x_theta, **kwargs)
        else:
            x_theta = cache["x_theta"]
        cache = {"x_theta": x_theta}
        if self.sampler_config.use_x0_pred:
            return x_theta, cache
        q_xs = self._compute_posterior(x_theta, denoiser_inputs.xt, alpha_t, alpha_s)
        return q_xs, cache

    def generate(  # TODO: clean up signature and docstring
        self,
        max_length: int | None = None,
        batch_size: int | None = None,
        disable_cache: bool | None = None,
        device: str | None = None,
        context: torch.Tensor | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, int]:
        max_length = (
            max_length if max_length is not None else self.sampler_config.max_length
        )
        batch_size = (
            batch_size if batch_size is not None else self.sampler_config.batch_size
        )
        disable_cache = (
            disable_cache
            if disable_cache is not None
            else self.sampler_config.disable_cache
        )
        device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        max_blocks = max_length // self.sampler_config.block_size
        if context is not None:
            assert context.shape[-1] + max_length <= self.config.length
            accumulated_samples = context.to(device)
        else:
            accumulated_samples = torch.empty(
                (batch_size, 0), dtype=torch.int64, device=device
            )

        total_NFEs = 0
        # cache kvs of context
        past_key_values = None
        if context is not None and self.sampler_config.kv_caching:
            past_key_values = self.update_kv_cache(
                context=context,
                past_key_values=past_key_values,
            )
        block_size = self.sampler_config.block_size
        block_pbar = tqdm(range(max_blocks), desc="Sampling blocks", leave=False)
        for _ in block_pbar:
            block_NFEs = 0
            xt = self._sample_prior(
                device=device,
                batch_size=batch_size,
                length=block_size,
            )
            if self.config.shift_logits and context is not None:
                xt = torch.cat((accumulated_samples[:, -1:], xt), dim=-1)
                accumulated_samples = accumulated_samples[:, :-1]

            timesteps = self._sample_generation_timesteps(
                max_seq_len=block_size, device=device
            )
            step_bar = tqdm(timesteps, desc="T", total=timesteps.shape[0], leave=False)
            dt = (1 - self.sampler_config.min_t) / len(timesteps)
            cache = None

            for t in step_bar:
                if cache is None:
                    block_NFEs += 1
                    total_NFEs += 1
                # t is 0-dim tensor, reshape to (1, 1, 1) for broadcasting
                alpha_t, _ = self.noise_schedule(t)
                alpha_s, _ = self.noise_schedule(t - dt)
                alpha_t = alpha_t[None, None, None]
                alpha_s = alpha_s[None, None, None]

                input_ids = xt
                denoiser_inputs = self._prepare_inputs_inference(
                    input_ids=input_ids,
                    context=accumulated_samples,
                    past_key_values=past_key_values,
                )

                q_xs, cache = self._generate_unconditional(
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                    denoiser_inputs=denoiser_inputs,
                    cache=cache,
                    xt=xt,
                    past_key_values=past_key_values,
                )

                xs = self._sample_categorical(q_xs)
                xs = self._maybe_remask(xs, q_xs, xt)
                if self.config.shift_logits:
                    xs = torch.cat((xt[:, :1], xs), dim=-1)

                block_pbar.set_postfix(
                    NFEs=total_NFEs,
                    block_NFEs=block_NFEs,
                    prob_check=(q_xs.sum() / xt.numel()).item(),
                    nan_check=bool(q_xs.isnan().sum() > 0),
                )

                if not torch.allclose(xs, xt) or not disable_cache:
                    cache = None
                xt = xs
            accumulated_samples = torch.cat((accumulated_samples, xt), dim=-1)

            if tokenizer is not None:
                print(tokenizer.batch_decode(accumulated_samples))
            if self.sampler_config.kv_caching:
                past_key_values = self.update_kv_cache(
                    context=xt,
                    past_key_values=past_key_values,
                )
        return accumulated_samples, total_NFEs


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
        if not self.training:
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask * (
                denoiser_inputs.x0 != self.pad_token_id
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
        if config.attn_backend == "flex_attention":
            self.static_attention_mask = None
            self.encoder_static_attention_mask = None
        self._create_static_mask()

    @staticmethod
    def _encoder_block_mask(
        b,
        h,
        q_idx,
        kv_idx,
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
        b,
        h,
        q_idx,
        kv_idx,
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
        # **1. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q == block_kv) & x0_flag_kv & ~x0_flag_q

        # **2. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q > block_kv) & (x0_flag_q == x0_flag_kv)

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
        # TODO

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
        block_diagonal = (block_q > block_kv) & (x0_flag_q == x0_flag_kv)

        # **2. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q == block_kv) & x0_flag_kv & ~x0_flag_q

        # **3. Block-Causal Mask (M_BC) **
        block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

        # **4. Combine Masks **
        return block_diagonal | offset_block_causal | block_causal

    def _create_static_mask(self) -> None:
        assert self.config.attn_backend != "flex_attention", (
            "FlexAttention not supported yet"
        )
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
            if self.config.attn_backend == "flex_attention":
                self.static_attention_mask = static_mask
            else:
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
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length)[None, :],
                    block_size=self.config.block_size,
                )
                decoder_static_mask = self._decoder_block_mask(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length * 2)[None, :],
                    block_size=self.config.block_size,
                    seq_len=self.config.length,
                )
            if self.config.attn_backend == "flex_attention":
                self.encoder_static_attention_mask = encoder_static_mask
                self.static_attention_mask = decoder_static_mask
            else:
                self.register_buffer(
                    "encoder_static_attention_mask",
                    encoder_static_mask,
                )
                self.register_buffer(
                    "static_attention_mask",
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
                self.static_attention_mask[None, ...]
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
                self.static_attention_mask[None, ...]
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
        context: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        **kwargs: Any,
    ):
        batch_size = input_ids.shape[0]
        if self.config.backbone_is_decoder_only:
            raise NotImplementedError(
                "Inference for decoder-only BD3LM is not implemented yet."
            )
        else:
            position_ids = None
            if past_key_values is not None:
                full_seq_len = past_key_values.get_seq_length() + input_ids.shape[1]
                encoder_attention_mask = None
                position_ids = torch.arange(
                    past_key_values.get_seq_length(), full_seq_len
                ).to(input_ids.device)[None, :]
            else:
                context_len = context.shape[1]
                full_seq_len = context_len + input_ids.shape[1]
                encoder_attention_mask = self.encoder_static_attention_mask[
                    None, :context_len, :context_len
                ]
                position_ids = torch.arange(context_len, full_seq_len).to(
                    input_ids.device
                )[None, :]
            decoder_attention_mask = torch.ones(
                (batch_size, input_ids.shape[1], full_seq_len),
                device=input_ids.device,
            )
            return DenoiserInput(
                xt=input_ids,
                attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                backbone_kwargs={
                    "encoder_input_ids": context,
                    "encoder_attention_mask": encoder_attention_mask,
                    "position_ids": position_ids,
                },
            )


# TODO
# class UDLM(D3PM):


# TODO
# class SEDD(Denoiser):


# TODO
# class DFM(Denoiser):
