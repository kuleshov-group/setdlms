import copy
import inspect
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from tqdm import tqdm

import hydra.utils
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class DenoiserInput(OrderedDict):
    """Input to the denoiser model."""

    x0: torch.Tensor
    xt: torch.Tensor
    attention_mask: torch.Tensor | None = None
    t: torch.Tensor | None = None
    alpha_t: torch.Tensor | None = None
    alpha_t_prime: torch.Tensor | None = None
    # Placeholder in case future experiments require different inputs
    kwargs: dict[str, Any] | None = None


@dataclass
class LossAndNllOutput(OrderedDict):
    """Loss output for denoiser models."""

    loss: torch.Tensor
    nlls: torch.Tensor


@dataclass
class DenoiserOutput(ModelOutput):
    """Output of the denoiser model."""

    model_output: torch.Tensor
    tokens_mask: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    nlls: torch.Tensor | None = None
    # Placeholder in case future models produce different outputs
    output_kwargs: dict[str, Any] | None = None


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
        tokenization_config: dict[str, Any] | None = None,
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
                not hasattr(self, v) or hasattr(tokenization_config, v)
            ):
                setattr(self, v, tokenization_config.get(v, None))
            else:
                setattr(self, v, None)
        self.backbone_config = backbone_config
        self.noise_config = noise_config
        self.length = length


class Denoiser(ABC, PreTrainedModel):
    """Abstract base class for denoising models.

    This class defines the interface for AR, Diffusion, and Flow-based parametrizations.
    """

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
        self.noise_schedule = hydra.utils.instantiate(config.noise_config)
        self.time_conditioned_backbone = (
            "noise" in inspect.getfullargspec(self.backbone.forward).args
        )

    @abstractmethod
    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
    ) -> DenoiserInput:
        """
        Prepare inputs for the model.

        Parameters:
            input_ids (torch.Tensor): Input tensor to the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the model.
            t (Optional[torch.Tensor]): Time step for the model.

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
            LossAndNllOutput: loss (torch.FloatTensor) and nlls (torch.FloatTensor).
        """
        raise NotImplementedError("Denoiser subclasses must implement _compute_loss")

    def _forward(
        self,
        backbone_output: torch.Tensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the denoiser model to be implemented by subclasses.

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
            )
        return self.backbone(
            denoiser_inputs.xt, attention_mask=denoiser_inputs.attention_mask, **kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        compute_loss: bool | None = True,
        **kwargs,
    ) -> DenoiserOutput:
        """
        Perform a forward pass through the denoising model and
        (optionally) compute the loss.

        Parameters:
            input_ids (torch.Tensor): Input tensor to the model.
            labels (Optional[torch.Tensor]): Labels for the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the model.
            compute_loss (Optional[bool]): Flag to compute loss.

        Returns:
            DenoiserOutput
        """
        t = kwargs.pop("t", None)
        denoiser_inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            t=t,
        )

        backbone_output = self._backbone_forward(denoiser_inputs, **kwargs)
        if isinstance(backbone_output, ModelOutput) and hasattr(
            backbone_output, "logits"
        ):
            backbone_output = backbone_output.logits
        model_output = self._forward(
            backbone_output,
            denoiser_inputs,
            **kwargs,
        )

        if compute_loss:
            loss_and_nll = self._compute_loss(
                model_output=model_output, denoiser_inputs=denoiser_inputs, **kwargs
            )
            loss = loss_and_nll.loss
            nlls = loss_and_nll.nlls
        else:
            loss, nlls = None, None
        return DenoiserOutput(
            model_output=model_output,
            tokens_mask=denoiser_inputs.attention_mask,
            loss=loss,
            nlls=nlls,
        )

    @abstractmethod
    def generate_samples(self):  # TODO: clean up signature and docstring
        """Generate samples starting from noise.

        # TODO: will need to enable infilling / starting from partially noised sequences
        """
        pass


class AR(Denoiser):
    """Denoiser class for autoregressive (AR) models."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.time_conditioned_backbone = False

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
    ) -> DenoiserInput:
        # Prepare inputs for autoregressive model
        labels = copy.deepcopy(input_ids[..., 1:])

        input_ids = input_ids[..., :-1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)

        return DenoiserInput(
            x0=labels,
            xt=input_ids,
            attention_mask=attention_mask,
        )

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        # Shift labels
        loss = -torch.gather(model_output, -1, denoiser_inputs.x0).squeeze(-1)

        nlls = loss * denoiser_inputs.attention_mask
        count = denoiser_inputs.attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return LossAndNllOutput(loss=token_nll, nlls=nlls)

    def generate_samples(self):
        pass  # TODO


class D3PMConfig(DenoiserConfig):
    """Configuration class for D3PM models."""

    model_type = "d3pm"
    auto_map = {
        "AutoConfig": "denoiser.D3PMConfig",
        "AutoModel": "denoiser.D3PM",
    }

    def __init__(
        self,
        length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        noise_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        T: int = 1000,
        **kwargs,
    ):
        super().__init__(
            length, backbone_config, noise_config, tokenization_config, **kwargs
        )
        self.T = T


class D3PM(Denoiser):
    """Denoiser class for D3PM models.

    This class implements the Denoiser interface for D3PM models.
    """

    config_class = D3PMConfig

    def __init__(self, config: D3PMConfig):
        super().__init__(config)
        self.T = config.T

    def _sample_q_xt(self, x0: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
    ):
        # Prepare inputs for D3PM model
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)

        if t is None:
            t = torch.rand(input_ids.shape[0], device=input_ids.device)
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        if alpha_t.ndim == 1:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]
        xt = self._sample_q_xt(input_ids, alpha_t)

        return DenoiserInput(
            x0=input_ids,
            xt=xt,
            attention_mask=attention_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
        )

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        dt = 1 / self.T
        t = kwargs.get("t", None)

        if torch.is_tensor(t):
            t = t[:, None]
            assert t.ndim == 2
            t = t.clamp(0.0, 1.0 - 1e-4)
        alpha_t = 1 - t + torch.zeros_like(denoiser_inputs.x0)
        alpha_s = 1 - (t - dt) + torch.zeros_like(denoiser_inputs.x0)

        log_x_theta_at_x0 = torch.gather(
            model_output, -1, denoiser_inputs.x0[:, :, None]
        ).squeeze(-1)
        log_x_theta_at_m = model_output[:, :, self.mask_token_id]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

        L_vb_masked = term_1_coef * (term_1_log_nr - term_1_log_dr) + term_2_coef * (
            term_2_log_nr - term_2_log_dr
        )

        L_vb = L_vb_masked * (denoiser_inputs.xt == self.mask_token_id)
        loss = self.T * L_vb
        nlls = loss * denoiser_inputs.attention_mask
        count = denoiser_inputs.attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return LossAndNllOutput(loss=token_nll, nlls=nlls)
    
    def _sample_prior(self, device, batch_size, length):
        raise NotImplementedError
    
    def _compute_posterior(self, log_x_theta, alpha_t, alpha_s):
        raise NotImplementedError
    
    def _sample_categorical(self, categorical_probs):
        categorical_probs = categorical_probs.to(torch.float64)
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log()).to(categorical_probs.dtype)
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def generate_samples(self, device, batch_size, length, num_steps, nucleus_p=1.0, eps=1e-5):
        xt = self._sample_prior(device, batch_size, length)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        pbar = tqdm(range(num_steps), desc='Sampling', leave=False)
        NFEs = 0
        cache = None
        for i in pbar:
            if cache is None:
                t = timesteps[i]
                if self.T > 0:
                    t = (t * self.T).to(torch.int)
                    t = t / self.T
                    t += (1 / self.T)
                if cache is None:
                    NFEs += 1
                # alpha_t and alpha_s should be scalars
                alpha_t, _ = self.noise_schedule(t)
                alpha_s, _ = self.noise_schedule(t - dt)
                # prepare backbone inputs
                attention_mask = torch.ones_like(xt, dtype=torch.float)
                denoiser_inputs = DenoiserInput(
                    xt=xt,
                    attention_mask=attention_mask,
                    alpha_t=alpha_t
                )
                backbone_output = self._backbone_forward(denoiser_inputs)
                if isinstance(backbone_output, ModelOutput) and hasattr(
                    backbone_output, "logits"
                ):
                    backbone_output = backbone_output.logits
                log_x_theta = self._forward(
                    backbone_output,
                    denoiser_inputs,
                ) # should be the log(x_\theta) with the shape of (B, Seq, Vocab)
                x_theta = log_x_theta.exp()
                if nucleus_p < 1:
                    sorted_probs, sorted_indices = torch.sort(x_theta, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    top_p_mask = cumulative_probs <= nucleus_p
                    top_p_mask[..., 0] = True
                    nucleus_probs = sorted_probs * top_p_mask
                    nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
                    x_theta = torch.zeros_like(x_theta).scatter_(-1, sorted_indices, nucleus_probs)
            else:
                x_theta = cache
            q_xs = self._compute_posterior(x_theta, alpha_t, alpha_s)
            xs = self._sample_categorical(q_xs)
            pbar.set_postfix(NFEs=NFEs, prob_check=(q_xs.sum() / xt.numel()).item(), nan_check=bool(q_xs.isnan().sum() > 0))
            cache = x_theta
            if (not torch.allclose(xs, xt)):
                cache = None
            xt = xs
        return xt        


class MDLMConfig(D3PMConfig):
    """Configuration class for MDLM models."""

    model_type = "mdlm"
    auto_map = {
        "AutoConfig": "denoiser.MDLMConfig",
        "AutoModel": "denoiser.MDLM",
    }


class MDLM(D3PM):
    """Denoiser class for MDLM models."""

    config_class = MDLMConfig

    def __init__(self, config: MDLMConfig):
        super().__init__(config)
        self.neg_infinity = -1e12

    def _sample_q_xt(self, x0: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        move_indices = torch.rand(*x0.shape, device=x0.device) < (1.0 - alpha_t)
        return torch.where(move_indices, self.mask_token_id, x0)

    def _forward(
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
        unmasked_indices = denoiser_inputs.xt != self.mask_token_id
        log_probs[unmasked_indices] = self.neg_infinity
        log_probs[unmasked_indices, denoiser_inputs.xt[unmasked_indices]] = 0
        return log_probs

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=denoiser_inputs.x0[:, :, None]
        ).squeeze(-1)

        loss = (
            -log_p_theta * denoiser_inputs.alpha_t_prime / (1 - denoiser_inputs.alpha_t)
        )

        nlls = loss * denoiser_inputs.attention_mask
        count = denoiser_inputs.attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count
        return LossAndNllOutput(loss=token_nll, nlls=nlls)
    
    def _sample_prior(self, device, batch_size, length):
        return self.mask_token_id * torch.ones((batch_size, length), dtype=torch.int64, device=device)
    
    def _compute_posterior(self, x_theta, alpha_t, alpha_s):
        q_xs = x_theta * (alpha_s - alpha_t) / (1 - alpha_t)
        q_xs[:, :, self.mask_token_id] = (1 - alpha_s) / (1 - alpha_t)
        return q_xs

# TODO
# class UDLM(D3PM):


# TODO
# class SEDD(Denoiser):

    
# TODO
# class DFM(Denoiser):
