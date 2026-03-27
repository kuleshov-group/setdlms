import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import LogitsProcessorList, PreTrainedTokenizer, StoppingCriteriaList
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import ExponentialDecayLengthPenalty

from src.denoiser.base import Denoiser, DenoiserConfig, DenoiserInput, LossAndNllOutput
from src.denoiser.diffusion_config import (
    DiffusionGenerationConfig,
    DiffusionGenerationOutput,
)


class MDLMConfig(DenoiserConfig):
    """Configuration class for MDLM models."""

    model_type = "mdlm"
    auto_map = {
        "AutoConfig": "diffusion.MDLMConfig",
        "AutoModel": "diffusion.MDLM",
        "AutoModelForMaskedLM": "diffusion.MDLM",
    }

    def __init__(
        self,
        keep_clean_bos: Optional[bool] = None,  # Whether to enforce un-noised BOS token
        papl_alpha: float = 0.0,
        papl_tau: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if papl_alpha < 0:
            raise ValueError(f"`papl_alpha` must be non-negative, got {papl_alpha}.")
        if papl_tau <= 0:
            raise ValueError(f"`papl_tau` must be > 0, got {papl_tau}.")
        self.keep_clean_bos = keep_clean_bos
        self.papl_alpha = papl_alpha
        self.papl_tau = papl_tau


class MDLM(Denoiser):
    """Denoiser class for MDLM models.

    This class implements the Denoiser interface for MDLM models.
    """

    config_class = MDLMConfig

    def __init__(
        self,
        config: MDLMConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, **kwargs)
        self._create_static_mask()
        self.neg_infinity = -1e12

    def _create_static_mask(self) -> None:
        static_mask = self.generate_static_mask()
        self.register_buffer(
            "static_attention_mask",
            static_mask,
        )
        self.skip_params_for_push.append("static_attention_mask")

    def generate_static_mask(self) -> torch.Tensor:
        static_mask = torch.ones(
            self.config.length, self.config.length, dtype=torch.bool
        )
        return static_mask

    def update_static_mask(self, new_static_mask: torch.Tensor) -> None:
        self.static_attention_mask.copy_(new_static_mask)

    def _sample_q_xt(
        self,
        x0: torch.LongTensor,
        alpha_t: torch.FloatTensor,
        mask: torch.FloatTensor,
    ) -> torch.LongTensor:
        """Sample from the pre-defined forward / noising process.

        Parameters:
            x0 (Tensor): Signal / data sample;
                can potentially include context tokens.
            alpha_t (Tensor): Amount of signal to retain.
            mask (Tensor): Indicator of tokens (to remain
                unchanged).
        """
        move_indices = torch.rand(*x0.shape, device=x0.device) < (1.0 - alpha_t)
        xt = torch.where((move_indices * (1 - mask)).bool(), self.mask_token_id, x0)
        if self.config.keep_clean_bos:
            xt[..., 0] = x0[..., 0]
        return xt  # type: ignore

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        # Prepare inputs for D3PM model
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)

        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
            context_mask = context_mask.to(torch.int)

        if t is None:
            t = torch.rand(input_ids.shape[0], device=input_ids.device)
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]
        noise_mask = context_mask | ~(attention_mask.bool())
        if getattr(self.config, "mdlm_loss_scale", False):
            eps = 1e-3
            t = 1 - alpha_t
            sigma = -torch.log1p(-(1 - eps) * t)
            p = 1 - torch.exp(-sigma)
            xt = self._sample_q_xt(x0=input_ids, alpha_t=1 - p, mask=noise_mask)
        else:
            xt = self._sample_q_xt(x0=input_ids, alpha_t=alpha_t, mask=noise_mask)
        if self.config.keep_clean_bos:
            xt[..., 0] = input_ids[..., 0]
        if (
            context_mask is not None
            and context_mask.sum() == 0
            and (attention_mask == 1).all()
        ):
            processed_attention_mask = None
        else:
            processed_attention_mask = (
                self.static_attention_mask[None, ...]
                & attention_mask[:, None, :]
                & attention_mask[..., None]
            )[:, None, ...]  # Make attention mask 4D
            processed_attention_mask = self._preprocess_attention_mask(
                processed_attention_mask, dtype=torch.float
            )
        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask)
        return DenoiserInput(
            xt=xt,
            x0=input_ids,
            attention_mask=processed_attention_mask,
            context_mask=context_mask,
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
        )

    def _prepare_inputs_inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context: Optional[torch.LongTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        cache: Optional[Dict[str, Any]] = None,
        **backbone_kwargs: Any,
    ) -> Tuple[DenoiserInput, Dict[str, Any]]:
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        if attention_mask is None:
            cache_length = self._get_past_key_values_seq_length(past_key_values)
            full_seq_length = cache_length + input_ids.shape[-1]
            attention_mask = torch.ones(
                (input_ids.shape[0], 1, input_ids.shape[1], full_seq_length),
                device=input_ids.device,
            )  # Make attention mask 4D
            attention_mask = self._preprocess_attention_mask(
                attention_mask, dtype=torch.float
            )
        return (
            DenoiserInput(
                xt=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                backbone_kwargs=backbone_kwargs | {"use_cache": False},
            ),
            cache,
        )

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs,
    ) -> torch.FloatTensor:
        # Zero-mask probability
        backbone_output[..., self.mask_token_id] = self.neg_infinity
        log_probs = backbone_output - torch.logsumexp(
            backbone_output, dim=-1, keepdim=True
        )
        # Copy-over unmasked: For the log_probs of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        xt = denoiser_inputs.xt
        unmasked_indices = xt != self.mask_token_id
        log_probs[unmasked_indices] = self.neg_infinity
        log_probs[unmasked_indices, xt[unmasked_indices]] = 0
        return log_probs  # type: ignore

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=denoiser_inputs.x0[:, :, None]
        ).squeeze(-1)
        if self.config.keep_clean_bos and not self.training:
            log_p_theta = log_p_theta[:, 1:]
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask[:, 1:]
            denoiser_inputs.alpha_t_prime = denoiser_inputs.alpha_t_prime[:, 1:]
            denoiser_inputs.alpha_t = denoiser_inputs.alpha_t[:, 1:]
            denoiser_inputs.xt = denoiser_inputs.xt[:, 1:]
        block_size = getattr(self.config, "block_size", denoiser_inputs.x0.shape[-1])
        papl_alpha = float(getattr(self.config, "papl_alpha", 0.0))
        masked_tokens = (denoiser_inputs.xt == self.mask_token_id).int()
        masked_positions = masked_tokens.bool() & denoiser_inputs.tokens_mask.bool()

        # below is needed to reproduce mdlm/sedd numbers with models from sahoo et al
        # (numerical imprecision computing probs under loglinear schedule)
        loss_scale = 1.0
        if getattr(self.config, "mdlm_loss_scale", False):
            eps = 1e-3
            t = 1 - denoiser_inputs.alpha_t
            sigma = -torch.log1p(-(1 - eps) * t)
            dsigma = (1 - eps) / (1 - (1 - eps) * t)
            loss_scale = dsigma / torch.expm1(sigma)
        elif block_size > 1 or getattr(self.config, "train_on_nelbo", False):
            loss_scale = -(
                denoiser_inputs.alpha_t_prime / (1 - denoiser_inputs.alpha_t)
            )
        nlls = -log_p_theta * denoiser_inputs.tokens_mask * loss_scale

        if self.training or block_size == 1:
            batch_nll = -(log_p_theta * denoiser_inputs.tokens_mask).sum(dim=-1)
        else:
            batch_nll = nlls.sum(dim=-1)

        papl_active = (
            self.training
            and not getattr(self.config, "train_on_nelbo", False)
            and papl_alpha > 0
        )
        other_loss_terms = {
            "masked_tokens": masked_tokens,
            "log_p_theta": -log_p_theta * denoiser_inputs.tokens_mask,
        }
        if papl_active:
            token_nll, papl_metrics = self._papl_loss(
                target_log_probs=log_p_theta,
                masked_positions=masked_positions,
            )
            other_loss_terms.update(
                {
                    "papl_avg_n_masked": papl_metrics["avg_n_masked"],
                    "papl_avg_planner_entropy": papl_metrics["avg_planner_entropy"],
                    "papl_avg_correct_prob_on_masked": papl_metrics[
                        "avg_correct_prob_on_masked"
                    ],
                    "papl_enabled": torch.ones((), device=log_p_theta.device, dtype=torch.int),
                }
            )
        elif (
            self.training and not getattr(self.config, "train_on_nelbo", False)
        ) or block_size == 1:
            # Average over masked tokens during training
            batch_nll = -(log_p_theta * denoiser_inputs.tokens_mask).sum(dim=-1)
            count = masked_tokens.sum(
                dim=-1
            )  # override count to be masked tokens
            token_nll = torch.where(
                count > 0, batch_nll / count, torch.zeros_like(batch_nll)
            ).mean()
        else:
            # NELBO; average over response tokens
            count = denoiser_inputs.tokens_mask.sum(dim=-1)
            token_nll = torch.where(
                count > 0, batch_nll / count, torch.zeros_like(batch_nll)
            ).mean()
        return LossAndNllOutput(
            loss=token_nll,  # type: ignore
            nlls=nlls,
            other_loss_terms=other_loss_terms,
        )

    def _papl_params(self) -> tuple[float, float]:
        papl_alpha = float(getattr(self.config, "papl_alpha", 0.0))
        papl_tau = float(getattr(self.config, "papl_tau", 1.0))
        if papl_alpha < 0:
            raise ValueError(
                f"`papl_alpha` must be non-negative, got {papl_alpha}."
            )
        if papl_tau <= 0:
            raise ValueError(f"`papl_tau` must be > 0, got {papl_tau}.")
        return papl_alpha, papl_tau

    @staticmethod
    def _papl_metrics(
        planner_weights: torch.FloatTensor,
        masked_positions: torch.BoolTensor,
        target_log_probs: torch.FloatTensor,
        eps: float = 1e-8,
    ) -> dict[str, torch.FloatTensor]:
        masked_float = masked_positions.to(target_log_probs.dtype)
        n_masked = masked_float.sum(dim=-1).clamp_min(1)
        safe_probs = torch.where(
            masked_positions,
            planner_weights.clamp_min(eps),
            torch.ones_like(planner_weights),
        )
        avg_planner_entropy = (
            -(planner_weights * safe_probs.log()).sum(dim=-1)
        ).mean()
        if masked_positions.any():
            avg_correct_prob_on_masked = target_log_probs.exp()[masked_positions].mean()
        else:
            avg_correct_prob_on_masked = target_log_probs.new_zeros(())
        return {
            "avg_n_masked": n_masked.mean(),
            "avg_planner_entropy": avg_planner_entropy,
            "avg_correct_prob_on_masked": avg_correct_prob_on_masked,
        }

    def _papl_loss(
        self,
        target_log_probs: torch.FloatTensor,
        masked_positions: torch.BoolTensor,
    ) -> tuple[torch.FloatTensor, dict[str, torch.FloatTensor]]:
        papl_alpha, papl_tau = self._papl_params()
        target_nll = -target_log_probs
        masked_nll = target_nll * masked_positions.to(target_nll.dtype)
        detached_scores = (target_log_probs.detach() / papl_tau).masked_fill(
            ~masked_positions, float("-inf")
        )
        planner_weights = torch.zeros_like(target_log_probs)
        has_masked = masked_positions.any(dim=-1)
        if has_masked.any():
            planner_weights[has_masked] = F.softmax(
                detached_scores[has_masked], dim=-1
            )
        planner_weights = torch.where(
            masked_positions, planner_weights, torch.zeros_like(planner_weights)
        )
        n_masked = masked_positions.sum(dim=-1).clamp_min(1).to(target_nll.dtype)
        base_weight = n_masked.reciprocal().unsqueeze(-1)
        weights = base_weight * (1.0 + papl_alpha * planner_weights)
        loss = (weights * masked_nll).sum(dim=-1).mean()
        return loss, self._papl_metrics(
            planner_weights=planner_weights,
            masked_positions=masked_positions,
            target_log_probs=target_log_probs,
        )

    @torch.no_grad()
    def _compute_sampling_lengths(
        self,
        generation_config: DiffusionGenerationConfig,
        input_length: int,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[int, int]:
        if max_length is None:
            if hasattr(generation_config, "max_length"):
                max_length = generation_config.max_length
            else:
                max_length = self.max_length
        if max_new_tokens is None:
            if max_length is not None:
                max_new_tokens = max_length - input_length
            else:
                if hasattr(generation_config, "max_new_tokens"):
                    max_new_tokens = generation_config.max_new_tokens
                else:
                    max_new_tokens = max_length - input_length
        return max_length, max_new_tokens

    @torch.no_grad()
    def _compute_max_blocks_and_pad_input(
        self,
        inputs: torch.LongTensor,
        generation_config: DiffusionGenerationConfig,
        max_new_tokens: Optional[int] = None,
        block_size: int = None,
        is_infill_task: bool = False,
        mdlm_inference: bool = False,
    ) -> Tuple[torch.LongTensor, int, Optional[int]]:
        pad_length = None
        if is_infill_task:
            if generation_config.align_inputs_to_blocks:
                if mdlm_inference:
                    pad_length = inputs.shape[-1] % self.config.length
                    if pad_length > 0:
                        pad_length = self.config.length - pad_length
                else:
                    pad_length = inputs.shape[-1] % block_size
                    if pad_length > 0:
                        pad_length = block_size - pad_length
                inputs = F.pad(inputs, (0, pad_length), value=self.mask_token_id)
                mask_tokens = inputs == self.mask_token_id
                if pad_length > 0:
                    mask_tokens[:, -pad_length:] = False
                mask_tokens = mask_tokens.view(-1, block_size)
                max_blocks = (mask_tokens.max(dim=-1).values == 1).sum()
            else:
                max_blocks = math.ceil(
                    (inputs == self.mask_token_id).sum() / block_size
                )
            block_size = min(block_size, inputs.shape[-1])
        else:
            max_blocks = math.ceil(max_new_tokens / block_size)
        return inputs, max_blocks, pad_length

    def _sample_prior(
        self,
        inputs: torch.LongTensor,
        generation_config: DiffusionGenerationConfig,
        batch_size: int,
        max_new_tokens: int,
        block_size: int,
        is_infill_task: bool,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> torch.LongTensor:
        mdlm_inference = (
            getattr(self.config, "block_size", self.config.length) == self.config.length
        )
        if is_infill_task:
            num_mask_tokens = (inputs == self.mask_token_id).sum()
            if mdlm_inference:
                pad_length = inputs.shape[-1] % self.config.length
                if pad_length > 0:
                    pad_length = self.config.length - pad_length
            else:
                pad_length = inputs.shape[-1] % block_size
                if pad_length > 0:
                    pad_length = block_size - pad_length
            inputs = F.pad(
                inputs, (0, pad_length), value=generation_config.pad_token_id
            )
            mask_tokens = inputs == self.mask_token_id
            if pad_length > 0:
                mask_tokens[:, -pad_length:] = False
            mask_tokens = mask_tokens.view(-1, block_size)
            block_size = min(block_size, inputs.shape[-1])
        else:
            num_mask_tokens = max_new_tokens

        # Sample max generation length tensor from prior
        if is_infill_task:
            masked_tensor = inputs
        else:
            masked_tensor = self.mask_token_id * torch.ones(
                (batch_size, num_mask_tokens), dtype=torch.int64, device=device
            )
            if inputs is not None:
                masked_tensor = torch.cat([inputs, masked_tensor], dim=-1)
            if mdlm_inference and masked_tensor.shape[-1] < self.config.length:
                masked_tensor = F.pad(
                    masked_tensor,
                    (0, self.config.length - masked_tensor.shape[-1]),
                    value=generation_config.pad_token_id,
                )
        return masked_tensor

    def _compute_posterior(
        self,
        x: Union[torch.FloatTensor, torch.LongTensor],
        xt: torch.LongTensor,
        alpha_t: torch.FloatTensor,
        alpha_s: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Computes posterior / approximate posterior q(x_s | x_t, x),
            where x represents clean sequence (as one-hots) or the output of the
            denoising model.

        Args:
            x (Tensor): True (one-hot) / predicted clean signal (B, L, V).
            xt (Tensor): Noised signal at time t (B, L).
            alpha_t (Tensor): Noise schedule parameter at time t (B, 1, 1).
            alpha_s (Tensor): Noise schedule parameter at time s (B, 1, 1).
        """
        q_xs = x * (alpha_s[..., None] - alpha_t[..., None])
        q_xs[..., self.mask_token_id] = 1 - alpha_s
        # removed in mdlm:
        q_xs = torch.where(alpha_t[..., None] != 1, q_xs / (1 - alpha_t[..., None]), x)
        return q_xs  # type: ignore

    def _sample_generation_timesteps(
        self,
        generation_config: DiffusionGenerationConfig,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float64,
        first_hitting_times: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Sample timesteps for diffusion generation process."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if max_length is None:
            max_length = generation_config.max_new_tokens
        sampling_strategy = generation_config.sampling_strategy
        if generation_config.first_hitting and sampling_strategy == "posterior":
            if first_hitting_times is None:
                first_hitting_times = self.noise_schedule.compute_first_hitting_times(
                    batch_size=1, length=max_length, device=device, dtype=dtype
                )[0]
            return first_hitting_times.sort(descending=True).values
        num_steps = (
            generation_config.num_steps + 1
            if sampling_strategy == "posterior"
            else min(generation_config.num_steps + 1, max_length + 1)
        )
        return torch.linspace(  # type: ignore
            1.0,
            0.0,
            num_steps,
            device=device,
            dtype=dtype,
        )[:-1]

    def _nucleus_sample(self, p_x0: torch.FloatTensor, p: float):
        if p >= 1.0:
            return p_x0

        if getattr(self.config, "block_size", None) is not None:
            p_x0_ = p_x0[:, -self.config.block_size :].clone()
        else:
            p_x0_ = p_x0.clone()

        sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)

        # remove tokens after the first one that pushes cumulative prob above p
        sorted_mask = cum_probs >= p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        filtered_sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)

        filtered = torch.zeros_like(p_x0_)
        filtered.scatter_(-1, sorted_indices, filtered_sorted_probs)

        filtered = filtered / filtered.sum(dim=-1, keepdim=True)

        if getattr(self.config, "block_size", None) is not None:
            out = p_x0.clone()
            out[:, -self.config.block_size :] = filtered
            return out
        else:
            return filtered

    def _generate_unconditional(
        self,
        generation_config: DiffusionGenerationConfig,
        t: torch.FloatTensor,
        next_t: torch.FloatTensor,
        denoiser_inputs: Optional[DenoiserInput] = None,
        cache: Optional[Dict[str, Any]] = None,
        running_generation: Optional[torch.LongTensor] = None,
        inputs_offset: Optional[int] = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
        sample_indices: Optional[Tuple[int, int]] = None,
        input_indices: Optional[Tuple[int, int]] = None,
        return_updated_cache: bool = False,
        cache_len: Optional[int] = None,
        window_size: int = 0,
        block_size: int = 0,
        **kwargs: Any,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.FloatTensor], Dict[str, Any]]:
        cache = cache if cache is not None else {}
        backbone_output = self._backbone_forward(
            denoiser_inputs,
            fix_cache_length=(
                True if not return_updated_cache else False
            ),  # Do not let kv cache grow on each forward call
            **cache,
            **kwargs,
        )
        if isinstance(backbone_output, torch.Tensor):
            logits = backbone_output
        else:
            backbone_output = {k: v for k, v in backbone_output.items()}
            logits = backbone_output.pop("logits")
            cache = cache | backbone_output
        if cache_len is not None:
            logits = logits[:, cache_len:]
            denoiser_inputs.xt = denoiser_inputs.xt[:, cache_len:]
        elif sample_indices is not None and input_indices is not None:
            logits = logits[:, sample_indices - input_indices[0], :]
            denoiser_inputs.xt = denoiser_inputs.xt[
                ..., sample_indices - input_indices[0]
            ]
        else:
            logits = logits[:, sample_indices - sample_indices[0], :]
            denoiser_inputs.xt = denoiser_inputs.xt[
                ..., sample_indices - sample_indices[0]
            ]  # truncate any extra padding tokens

        if logits_processor is not None and len(logits_processor) > 0:
            log_x_theta = logits
            sample_idx = (
                sample_indices[0] if sample_indices.ndim == 2 else sample_indices
            )
            for lp in logits_processor:
                for j in range(log_x_theta.shape[1]):
                    if isinstance(lp, ExponentialDecayLengthPenalty):
                        log_x_theta[:, j] = lp(
                            input_ids=running_generation[..., : sample_idx[j]],
                            scores=log_x_theta[:, j],
                        )
                    else:
                        log_x_theta[:, j] = lp(
                            input_ids=running_generation,
                            scores=log_x_theta[:, j],
                        )
            # renormalize
            log_x_theta[..., self.mask_token_id] = self.neg_infinity
            log_x_theta = log_x_theta - torch.logsumexp(
                log_x_theta, dim=-1, keepdim=True
            )
        else:
            log_x_theta = self._forward(logits, denoiser_inputs, **kwargs)

        x_theta = log_x_theta.exp()

        # nucleus sampling
        if generation_config.nucleus_p < 1.0:
            x_theta = self._nucleus_sample(x_theta, generation_config.nucleus_p)

        sampling_strategy = generation_config.sampling_strategy
        if sampling_strategy == "posterior":
            assert t is not None and next_t is not None, (
                "t and next_t must be provided for posterior sampling."
            )
            alpha_t, _ = self.noise_schedule(
                t[None, None].repeat(
                    denoiser_inputs.xt.shape[0], denoiser_inputs.xt.shape[1]
                )
            )
            alpha_s, _ = self.noise_schedule(
                next_t[None, None].repeat(
                    denoiser_inputs.xt.shape[0], denoiser_inputs.xt.shape[1]
                )
            )
            q_xs = self._compute_posterior(
                x_theta, denoiser_inputs.xt, alpha_t, alpha_s
            )
            # removed in mdlm (from removing denominator)
            # assert abs(
            #     (q_xs.sum() / (denoiser_inputs.xt.numel())).item() - 1.0
            # ) < 1e-6, ("Posterior probabilities not summing to 1.")
            assert q_xs.isnan().sum().item() == 0, "NaN found in the posterior."
            xs = self._sample_categorical(q_xs, generation_config.do_sample)
            output = torch.where(
                (denoiser_inputs.xt != self.mask_token_id).bool(),  # type: ignore
                denoiser_inputs.xt,
                xs,
            )
        elif sampling_strategy == "predict_and_noise":
            # Predict
            xs = self._sample_categorical(x_theta, generation_config.do_sample)
            xs_probs = x_theta.gather(-1, xs[..., None]).squeeze(dim=-1)
            output = xs.clone()

            # Noise
            est_noise_indices_next = (next_t * block_size).round().to(torch.int)
            est_noise_indices_curr = (t * block_size).round().to(torch.int)
            num_to_decode = est_noise_indices_curr - est_noise_indices_next
            num_noise_indices = denoiser_inputs.xt.shape[-1] - num_to_decode
            if generation_config.confidence_based_noising:
                conf = x_theta.gather(-1, xs[..., None]).squeeze(-1)
                conf = torch.where(  # already decoded tokens have 'inf' confidence
                    (denoiser_inputs.xt == self.mask_token_id).bool(),  # type: ignore
                    conf,
                    torch.inf,
                )
                noise_indices = conf.argsort(dim=-1)[..., :num_noise_indices]
            elif generation_config.confidence_margin_based_noising:
                top2 = torch.topk(x_theta, k=2, dim=-1).values  # shape: (B, L, 2)
                conf = (top2[..., 0] - top2[..., 1]).abs()
                conf = torch.where(  # already decoded tokens have 'inf' confidence
                    (denoiser_inputs.xt == self.mask_token_id).bool(),  # type: ignore
                    conf,
                    torch.inf,
                )
                num_clean_indices = (denoiser_inputs.xt != self.mask_token_id).sum(
                    -1
                ) + num_to_decode
                noise_indices = conf.argsort(dim=-1)[..., : -num_clean_indices[0]]
            else:
                # Always decode the most confident token
                conf = x_theta.gather(-1, xs[..., None]).squeeze(-1)
                conf = torch.where(  # already decoded tokens have 'inf' confidence
                    (denoiser_inputs.xt == self.mask_token_id).bool(),  # type: ignore
                    conf,
                    torch.inf,
                )
                num_clean_indices = (denoiser_inputs.xt != self.mask_token_id).sum(
                    -1
                ) + num_to_decode
                noise_indices = conf.argsort(dim=-1)[..., : -num_clean_indices[0]]
            output[..., noise_indices] = self.mask_token_id
            output = torch.where(
                xs_probs >= generation_config.confidence_threshold, xs, output
            )
            output = torch.where(
                denoiser_inputs.xt == self.mask_token_id, output, denoiser_inputs.xt
            )
        else:
            raise NotImplementedError(
                f"Sampling strategy {sampling_strategy} not implemented."
            )
        return output, cache  # type: ignore

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[DiffusionGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        return_dict_in_generate: Optional[bool] = False,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs: Any,
    ) -> torch.LongTensor:
        # Setup sampling variables
        if generation_config is None:
            assert getattr(self, "generation_config", None) is not None, (
                "Generation config must be provided if not present in the model."
            )
            generation_config = self.generation_config
        input_length = inputs.shape[-1] if inputs is not None else 0
        max_length, max_new_tokens = self._compute_sampling_lengths(
            generation_config=generation_config,
            input_length=input_length,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )
        block_size = generation_config.block_size
        is_infill_task = self.mask_token_id in inputs
        mdlm_inference = (
            getattr(self.config, "block_size", self.config.length) == self.config.length
        )

        # Compute max blocks, maybe pad input
        inputs, max_blocks, pad_length = self._compute_max_blocks_and_pad_input(
            inputs,
            generation_config,
            max_new_tokens,
            block_size,
            is_infill_task,
            mdlm_inference,
        )
        accumulated_samples = self._sample_prior(
            inputs=inputs,
            batch_size=batch_size,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            is_infill_task=is_infill_task,
            device=device,
        )

        cache = None
        blocks_to_cache_flag = (
            generation_config.align_inputs_to_blocks and input_length >= block_size
        ) or not generation_config.align_inputs_to_blocks
        precomputed_cache_flag = (
            generation_config.use_cache and input_length > 0 and blocks_to_cache_flag
        )
        if precomputed_cache_flag:
            cache = self.update_cache(
                inputs=(
                    inputs[:, : block_size * (input_length // block_size)][
                        :, -self.config.length :
                    ]
                    if generation_config.align_inputs_to_blocks
                    else inputs[:, -self.config.length :]
                ),
                cache={},
            )

        if is_infill_task:
            inputs_offset = (
                (accumulated_samples == self.mask_token_id)[0].nonzero().min()
            )
            first_mask_token_idx = inputs_offset
            last_mask_token_idx = (accumulated_samples != self.mask_token_id).float()[
                :, first_mask_token_idx:
            ].argmax(dim=-1)[0] + first_mask_token_idx
        else:
            inputs_offset = input_length
            first_mask_token_idx = input_length
            last_mask_token_idx = input_length + max_new_tokens
        if generation_config.align_inputs_to_blocks:
            inputs_offset = (
                block_size * (inputs_offset // block_size) if inputs_offset > 0 else 0
            )

        total_NFEs = 0
        timesteps = self._sample_generation_timesteps(  # Re-use in every block
            generation_config, max_length=block_size, device=device
        )
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        disable_pbar = rank != 0 or kwargs.get("disable_pbar", False)

        block_pbar = tqdm(
            range(max_blocks),
            desc="Blocks",
            leave=True,
            disable=disable_pbar,
        )
        num_tokens_generated_per_step = []
        inf_budget_per_step = []
        sample_indices = None
        input_indices = None
        if mdlm_inference:
            if inputs_offset <= self.config.length:
                start_input_idx = 0
                end_input_idx = min(self.config.length, accumulated_samples.shape[-1])
            else:
                end_input_idx = max(inputs_offset + 32, self.config.length)
                start_input_idx = end_input_idx - self.config.length
            start_sample_idx = inputs_offset
            end_sample_idx = min(start_sample_idx + block_size, end_input_idx)
            if pad_length is not None:
                end_sample_idx = min(end_sample_idx, self.config.length - pad_length)
        for block_id in block_pbar:
            block_NFEs = 0
            # Compute indices for the current block
            if mdlm_inference:
                if block_id > 0:
                    start_sample_idx += block_size
                    end_sample_idx += block_size
                if start_sample_idx >= self.config.length:
                    end_input_idx = end_sample_idx
                    start_input_idx = end_input_idx - self.config.length
                xt = accumulated_samples[:, start_input_idx:end_input_idx]
                end_input_idx = min(end_input_idx, accumulated_samples.shape[-1])
                end_sample_idx = min(end_sample_idx, end_input_idx)
                if pad_length is not None:
                    end_sample_idx = min(
                        end_sample_idx, self.config.length - pad_length
                    )
                sample_indices = torch.arange(start_sample_idx, end_sample_idx).to(
                    device
                )
                input_indices = (start_input_idx, end_input_idx)
            elif generation_config.use_cache:
                xt = accumulated_samples[
                    :,
                    inputs_offset + (block_id * block_size) : inputs_offset
                    + ((block_id + 1) * block_size),
                ]
                end_sample_idx = inputs_offset + ((block_id + 1) * block_size)
                if pad_length is not None and pad_length > 0:
                    end_sample_idx = min(
                        end_sample_idx, accumulated_samples.shape[-1] - pad_length
                    )
                sample_indices = torch.arange(
                    inputs_offset + (block_id * block_size), end_sample_idx
                ).to(device)
                input_indices = sample_indices
            else:
                xt = accumulated_samples[
                    :, : inputs_offset + ((block_id + 1) * block_size)
                ]
                end_sample_idx = inputs_offset + ((block_id + 1) * block_size)
                if pad_length is not None and pad_length > 0:
                    end_sample_idx = min(
                        end_sample_idx, accumulated_samples.shape[-1] - pad_length
                    )
                sample_indices = torch.arange(
                    inputs_offset + (block_id * block_size),
                    min(
                        inputs_offset + ((block_id + 1) * block_size),
                        accumulated_samples.shape[-1],
                    ),
                )
                input_indices = (0, inputs_offset + ((block_id + 1) * block_size))

            if self.mask_token_id not in xt:
                continue
            if sample_indices.shape[-1] == 0:
                break

            step_pbar = tqdm(
                timesteps,
                desc="T",
                total=timesteps.shape[0],
                leave=False,
                disable=disable_pbar,
            )
            context = (
                accumulated_samples[:, : (block_id * block_size) + inputs_offset]
                if not generation_config.use_cache
                else None
            )
            for i, t in enumerate(step_pbar):
                # Used for logit processing
                block_NFEs += 1
                total_NFEs += 1
                denoiser_inputs, cache = self._prepare_inputs_inference(
                    input_ids=xt,
                    context=context,
                    cache=cache if generation_config.use_cache else None,
                )
                next_t = (
                    timesteps[i + 1]
                    if i < timesteps.shape[-1] - 1
                    else timesteps[-1] * 0
                )
                generation_output = self._generate_unconditional(
                    generation_config=generation_config,
                    block_size=block_size,
                    t=t,
                    next_t=next_t,
                    denoiser_inputs=denoiser_inputs,
                    cache=cache,
                    running_generation=(
                        accumulated_samples[:, first_mask_token_idx:last_mask_token_idx]
                        if not is_infill_task
                        else accumulated_samples[:, : input_indices[-1] + 1]
                    ),  # type: ignore
                    inputs_offset=inputs_offset,
                    logits_processor=logits_processor,
                    tokenizer=tokenizer,
                    sample_indices=sample_indices,
                    input_indices=input_indices,
                    **kwargs,
                )

                xs, cache = generation_output
                block_pbar.set_postfix(
                    NFEs=total_NFEs,
                    block_NFEs=block_NFEs,
                )
                num_tokens_generated_per_step.append(
                    (xs != denoiser_inputs.xt).sum().item()
                )
                if generation_config.compute_inf_budget:
                    inf_budget = (denoiser_inputs.xt == self.mask_token_id).sum().item()
                    inf_budget_per_step.append(inf_budget)
                if input_indices is not None:
                    xt[..., sample_indices - input_indices[0]] = xs
                    if (
                        xt[..., sample_indices - input_indices[0]] == self.mask_token_id
                    ).sum().item() == 0:
                        break
                else:
                    xt = xs
                if input_indices is not None:
                    accumulated_samples.scatter_(
                        dim=-1,
                        index=sample_indices[None, :],
                        src=xt[..., sample_indices - input_indices[0]],
                    )
                else:
                    accumulated_samples.scatter_(
                        dim=-1, index=sample_indices[None, :], src=xt[:, -block_size:]
                    )
                if ((xt == self.mask_token_id).sum().item() == 0) or (
                    pad_length is not None
                    and pad_length > 0
                    and mdlm_inference
                    and (xt[:, :-pad_length] == self.mask_token_id).sum().item() == 0
                ):
                    if generation_config.compute_inf_budget:
                        remaining_steps = timesteps.shape[0] - len(inf_budget_per_step)
                        inf_budget_per_step.extend([0] * remaining_steps)
                    break
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=accumulated_samples[  # type: ignore
                        :,
                        : sample_indices[-1] + 1,
                    ],
                    scores=None,  # type: ignore
                )
                if torch.any(is_done):
                    accumulated_samples = accumulated_samples[
                        :,
                        : sample_indices[-1] + 1,
                    ]
                    break
            if (
                generation_config.use_cache
                and getattr(self.config, "block_size", self.config.length)
                < self.config.length
            ):
                cache = self.update_cache(
                    inputs=xt,
                    cache=cache,
                )
        parallelism_factor = (
            sum(num_tokens_generated_per_step) / len(num_tokens_generated_per_step)
            if len(num_tokens_generated_per_step) > 0
            else 0
        )
        inf_budget = None
        if generation_config.compute_inf_budget:
            inf_budget = sum(inf_budget_per_step) / len(inf_budget_per_step)
        accumulated_samples = accumulated_samples[
            accumulated_samples != self.mask_token_id
        ].unsqueeze(0)
        if return_dict_in_generate:
            return DiffusionGenerationOutput(
                sequences=accumulated_samples,
                parallelism_factor=parallelism_factor,
                inf_budget=inf_budget,
                inf_budgets=inf_budget_per_step,
            )
        return accumulated_samples  # type: ignore


class SEDD(MDLM):
    """Denoiser class for SEDD models."""

    def __init__(
        self,
        config: MDLMConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, **kwargs)

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        denoiser_inputs = super()._prepare_inputs(
            input_ids, attention_mask, context_mask, t, past_key_values
        )

        def _sigma_from_t(t, eps=1e-3):
            return -torch.log1p(-(1 - eps) * t)

        sigma_max = _sigma_from_t(torch.tensor(1.0)).to(denoiser_inputs.alpha_t.device)
        sigma = torch.min(_sigma_from_t(1 - denoiser_inputs.alpha_t), sigma_max)
        denoiser_inputs.backbone_kwargs["sigma"] = sigma[:, 0]
        return denoiser_inputs

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs,
    ) -> torch.FloatTensor:
        logits = backbone_output

        # sigma = - torch.log(denoiser_inputs.alpha_t)
        def _sigma_from_t(t, eps=1e-3):
            return -torch.log1p(-(1 - eps) * t)

        sigma_max = _sigma_from_t(torch.tensor(1.0)).to(denoiser_inputs.alpha_t.device)
        sigma = torch.min(_sigma_from_t(1 - denoiser_inputs.alpha_t), sigma_max)
        esigm1_log = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(logits.dtype)
        )
        # logits shape
        # (batch_size, diffusion_model_input_length, vocab_size)
        logits = logits - esigm1_log[..., None] - math.log(logits.shape[-1] - 1)
        # The below scatter operation sets the log score
        # for the input word to 0.
        logits = torch.scatter(
            logits, -1, denoiser_inputs.xt[..., None], torch.zeros_like(logits[..., :1])
        )
        return logits

    def _score_entropy(self, log_score, sigma, xt, x0):
        """Computes the SEDD loss.

        Args:
        log_score: float torch.Tensor with shape (batch_size,
            diffusion_model_input_length, vocab_size),
            log score, output of the denoising network.
        xt: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        x0: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        sigma: float torch.Tensor with shape (batch_size, 1).

        Returns:
        loss with shape (batch_size, diffusion_model_input_length)
        """
        masked_indices = xt == self.mask_token_id

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_score[masked_indices], -1, words_that_were_masked[..., None]
        ).squeeze(-1)
        score = log_score[masked_indices].exp()
        if self.mask_token_id == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_token_id].sum(dim=-1) + score[
                :, self.mask_token_id + 1 :
            ].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(*xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return entropy

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        if self.config.keep_clean_bos and not self.training:
            model_output = model_output[:, 1:]
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask[:, 1:]
            denoiser_inputs.alpha_t_prime = denoiser_inputs.alpha_t_prime[:, 1:]
            denoiser_inputs.alpha_t = denoiser_inputs.alpha_t[:, 1:]
            denoiser_inputs.xt = denoiser_inputs.xt[:, 1:]
            denoiser_inputs.x0 = denoiser_inputs.x0[:, 1:]

        def _sigma_from_t(t, eps=1e-3):
            return -torch.log1p(-(1 - eps) * t)

        sigma_max = _sigma_from_t(torch.tensor(1.0)).to(denoiser_inputs.alpha_t.device)
        sigma = torch.min(_sigma_from_t(1 - denoiser_inputs.alpha_t), sigma_max)
        # sigma = - torch.log(denoiser_inputs.alpha_t)
        # dsigma = 1 / denoiser_inputs.alpha_t
        dsigma = (1 / (1 - denoiser_inputs.alpha_t)) * torch.expm1(sigma)
        nlls = (
            dsigma
            * self._score_entropy(
                model_output, sigma, denoiser_inputs.xt, denoiser_inputs.x0
            )
            * denoiser_inputs.tokens_mask
        )

        batch_nll = nlls.sum(dim=-1)

        # NELBO; average over response tokens
        count = denoiser_inputs.tokens_mask.sum(dim=-1)
        token_nll = torch.where(
            count > 0, batch_nll / count, torch.zeros_like(batch_nll)
        ).mean()
        return LossAndNllOutput(
            loss=token_nll,  # type: ignore
            nlls=nlls,
        )

    def generate(self, *args, **kwargs):
        return NotImplementedError("SEDD generation not implemented yet.")
