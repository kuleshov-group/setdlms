import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import LogitsProcessorList, PreTrainedTokenizer, StoppingCriteriaList
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import (
    ExponentialDecayLengthPenalty,
    MinNewTokensLengthLogitsProcessor,
)

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keep_clean_bos = keep_clean_bos


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
        if getattr(self.config, "keep_clean_bos", False):
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
        if getattr(self.config, "keep_clean_bos", False):
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
        if getattr(self.config, "keep_clean_bos", False) and not self.training:
            log_p_theta = log_p_theta[:, 1:]
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask[:, 1:]
            denoiser_inputs.alpha_t_prime = denoiser_inputs.alpha_t_prime[:, 1:]
            denoiser_inputs.alpha_t = denoiser_inputs.alpha_t[:, 1:]
            denoiser_inputs.xt = denoiser_inputs.xt[:, 1:]
        block_size = getattr(self.config, "block_size", denoiser_inputs.x0.shape[-1])
        masked_tokens = (denoiser_inputs.xt == self.mask_token_id).int()

        # Needed to reproduce MDLM numbers with models from Sahoo et al.
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

        other_loss_terms = {
            "masked_tokens": masked_tokens,
            "log_p_theta": -log_p_theta * denoiser_inputs.tokens_mask,
        }
        if (
            self.training and not getattr(self.config, "train_on_nelbo", False)
        ) or block_size == 1:
            # Average over masked tokens during training
            batch_nll = -(log_p_theta * denoiser_inputs.tokens_mask).sum(dim=-1)
            count = masked_tokens.sum(dim=-1)  # override count to be masked tokens
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

    @staticmethod
    def _visible_infill_context(
        accumulated_samples: torch.LongTensor,
        mask_token_id: int,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        visible_context_mask = accumulated_samples != int(mask_token_id)
        if pad_token_id is not None:
            visible_context_mask = visible_context_mask & (
                accumulated_samples != int(pad_token_id)
            )
        if accumulated_samples.shape[0] == 1:
            return accumulated_samples[visible_context_mask].view(1, -1)

        visible_rows = [
            row[row_mask] for row, row_mask in zip(accumulated_samples, visible_context_mask)
        ]
        max_visible = max(row.numel() for row in visible_rows)
        fill_value = pad_token_id if pad_token_id is not None else mask_token_id
        visible_context = accumulated_samples.new_full(
            (accumulated_samples.shape[0], max_visible),
            int(fill_value),
        )
        for row_idx, row in enumerate(visible_rows):
            visible_context[row_idx, : row.numel()] = row
        return visible_context

    @staticmethod
    def _length_penalty_prefix_lengths(
        accumulated_samples: torch.LongTensor,
        sample_indices: torch.LongTensor,
        target_start_idx: int | torch.Tensor,
        mask_token_id: int,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        if accumulated_samples.ndim != 2:
            raise ValueError(
                "accumulated_samples must have shape [batch, sequence]."
            )
        if sample_indices.ndim == 1:
            candidate_indices = sample_indices.unsqueeze(0).expand(
                accumulated_samples.shape[0], -1
            )
        elif sample_indices.ndim == 2:
            candidate_indices = sample_indices
            if candidate_indices.shape[0] == 1 and accumulated_samples.shape[0] > 1:
                candidate_indices = candidate_indices.expand(
                    accumulated_samples.shape[0], -1
                )
            elif candidate_indices.shape[0] != accumulated_samples.shape[0]:
                raise ValueError(
                    "sample_indices batch size must match accumulated_samples."
                )
        else:
            raise ValueError("sample_indices must be rank 1 or rank 2.")

        target_start = (
            int(target_start_idx.item())
            if hasattr(target_start_idx, "item")
            else int(target_start_idx)
        )
        seq_len = accumulated_samples.shape[-1]
        target_start = max(0, min(target_start, seq_len))
        candidate_indices = candidate_indices.to(
            device=accumulated_samples.device, dtype=torch.long
        ).clamp(min=0, max=seq_len)

        visible = accumulated_samples != mask_token_id
        if pad_token_id is not None:
            visible = visible & (accumulated_samples != pad_token_id)
        positions = torch.arange(seq_len, device=accumulated_samples.device)
        visible = visible & (positions.unsqueeze(0) >= target_start)
        exclusive_prefix_counts = F.pad(visible.to(torch.long).cumsum(dim=-1), (1, 0))
        return torch.gather(exclusive_prefix_counts, 1, candidate_indices)


    @staticmethod
    def _relative_sample_positions(
        sample_indices: torch.LongTensor,
        input_indices: torch.LongTensor | tuple[int, int] | tuple[torch.Tensor, torch.Tensor],
        local_len: int,
    ) -> torch.LongTensor:
        if isinstance(input_indices, tuple):
            start = input_indices[0]
            if hasattr(start, "item"):
                start = int(start.item())
            rel = sample_indices - int(start)
        elif torch.is_tensor(input_indices):
            flat_input_indices = input_indices.reshape(-1).to(sample_indices.device)
            flat_sample_indices = sample_indices.reshape(-1).to(sample_indices.device)
            if (
                flat_input_indices.shape == flat_sample_indices.shape
                and torch.equal(flat_input_indices, flat_sample_indices)
            ):
                rel = torch.arange(
                    flat_sample_indices.numel(),
                    device=sample_indices.device,
                    dtype=sample_indices.dtype,
                )
            else:
                rel = flat_sample_indices - flat_input_indices[0]
                if rel.numel() and (rel.min() < 0 or rel.max() >= local_len):
                    if flat_input_indices.numel() == 0:
                        raise ValueError("input_indices is empty.")
                    lookup = torch.searchsorted(flat_input_indices, flat_sample_indices)
                    lookup = lookup.clamp(max=flat_input_indices.numel() - 1)
                    if not torch.equal(flat_input_indices[lookup], flat_sample_indices):
                        raise ValueError(
                            "sample_indices are not contained in input_indices: "
                            f"sample_min={int(flat_sample_indices.min().item())}, "
                            f"sample_max={int(flat_sample_indices.max().item())}, "
                            f"input_min={int(flat_input_indices.min().item())}, "
                            f"input_max={int(flat_input_indices.max().item())}"
                        )
                    rel = lookup.to(sample_indices.dtype)
        else:
            raise TypeError(f"Unsupported input_indices type: {type(input_indices)}")

        rel = rel.to(device=sample_indices.device, dtype=torch.long)
        if rel.numel() and (rel.min() < 0 or rel.max() >= local_len):
            raise ValueError(
                "sample_indices fall outside the local model-output window: "
                f"rel_min={int(rel.min().item())}, "
                f"rel_max={int(rel.max().item())}, local_len={local_len}"
            )
        return rel

    def _can_use_fused_block_cache(
        self,
        generation_config: DiffusionGenerationConfig,
        is_infill_task: bool,
        mdlm_inference: bool,
        block_size: int,
    ) -> bool:
        fused_block_cache = getattr(generation_config, "fused_block_cache", None)
        if isinstance(fused_block_cache, str):
            normalized = fused_block_cache.strip().lower()
            if normalized in {"auto", "none", "null"}:
                fused_block_cache = None
            elif normalized in {"1", "true", "yes", "on"}:
                fused_block_cache = True
            elif normalized in {"0", "false", "no", "off"}:
                fused_block_cache = False
        if fused_block_cache is None:
            fused_block_cache = getattr(self.config, "model_type", None) == "bd3lm"
        if not fused_block_cache:
            return False
        if not getattr(generation_config, "use_cache", False):
            return False
        if is_infill_task or mdlm_inference:
            return False
        if getattr(self.config, "block_size", self.config.length) >= self.config.length:
            return False
        if block_size <= 0 or 2 * block_size > self.config.length:
            return False
        return isinstance(getattr(self, "static_attention_mask", None), torch.Tensor)

    def _build_fused_block_cache_attention_mask(
        self,
        batch_size: int,
        cache_len: int,
        prefix_len: int,
        decode_len: int,
        device: torch.device,
    ) -> torch.FloatTensor:
        full_len = cache_len + prefix_len + decode_len
        local_len = prefix_len + decode_len
        static_attention_mask = self.static_attention_mask
        if static_attention_mask.device != device:
            static_attention_mask = static_attention_mask.to(device)
        if (
            static_attention_mask.shape[-2] < full_len
            or static_attention_mask.shape[-1] < full_len
        ):
            raise ValueError(
                "static attention mask is too short for fused block cache: "
                f"mask_shape={tuple(static_attention_mask.shape)}, full_len={full_len}"
            )
        attention_mask = torch.zeros(
            (batch_size, 1, local_len, full_len),
            dtype=torch.bool,
            device=device,
        )
        if prefix_len > 0:
            attention_mask[:, :, :prefix_len, : cache_len + prefix_len] = (
                static_attention_mask[
                    None,
                    None,
                    cache_len : cache_len + prefix_len,
                    : cache_len + prefix_len,
                ]
            )
        if decode_len > 0:
            attention_mask[:, :, prefix_len:, :full_len] = static_attention_mask[
                None,
                None,
                cache_len + prefix_len : full_len,
                :full_len,
            ]
        return self._preprocess_attention_mask(attention_mask, dtype=torch.float)

    @staticmethod
    def _crop_past_key_values_left(past_key_values: Any, drop: int) -> Any:
        if drop <= 0 or past_key_values is None:
            return past_key_values
        if not (
            hasattr(past_key_values, "key_cache")
            and hasattr(past_key_values, "value_cache")
        ):
            raise TypeError("DynamicCache-like structure not found")
        key_cache = getattr(past_key_values, "key_cache")
        value_cache = getattr(past_key_values, "value_cache")
        for i in range(len(past_key_values)):
            k = key_cache[i]
            v = value_cache[i]
            if k is not None:
                key_cache[i] = k[..., drop:, :]
            if v is not None:
                value_cache[i] = v[..., drop:, :]
        return past_key_values

    @staticmethod
    def _crop_cache_to_length(
        cache: Optional[Dict[str, Any]],
        keep_len: int,
    ) -> Optional[Dict[str, Any]]:
        if cache is None or "past_key_values" not in cache:
            return cache
        past_key_values = cache.get("past_key_values")
        if past_key_values is None:
            return cache
        if hasattr(past_key_values, "crop"):
            past_key_values.crop(keep_len)
            return cache
        if not (
            hasattr(past_key_values, "key_cache")
            and hasattr(past_key_values, "value_cache")
        ):
            raise TypeError("DynamicCache-like structure not found")
        key_cache = getattr(past_key_values, "key_cache")
        value_cache = getattr(past_key_values, "value_cache")
        for i in range(len(past_key_values)):
            k = key_cache[i]
            v = value_cache[i]
            if k is not None:
                key_cache[i] = k[..., :keep_len, :]
            if v is not None:
                value_cache[i] = v[..., :keep_len, :]
        return cache

    def _trim_cache_for_fused_block(
        self,
        cache: Optional[Dict[str, Any]],
        fused_input_len: int,
    ) -> tuple[Dict[str, Any], int]:
        cache = cache if cache is not None else {}
        past_key_values = cache.get("past_key_values", DynamicCache())
        cache_len = self._get_past_key_values_seq_length(past_key_values)
        max_cache_len = self.config.length - fused_input_len
        if max_cache_len < 0:
            raise ValueError(
                "fused block cache input exceeds model context: "
                f"fused_input_len={fused_input_len}, length={self.config.length}"
            )
        overflow = max(cache_len - max_cache_len, 0)
        if overflow > 0:
            crop_left = getattr(self, "_crop_kv_cache_left", None)
            if crop_left is not None:
                past_key_values = crop_left(past_key_values, overflow)
            else:
                past_key_values = self._crop_past_key_values_left(
                    past_key_values, overflow
                )
            cache["past_key_values"] = past_key_values
            cache_len -= overflow
        return cache, cache_len

    def _generate_unconditional(
        self,
        generation_config: DiffusionGenerationConfig,
        t: torch.FloatTensor,
        next_t: torch.FloatTensor,
        denoiser_inputs: Optional[DenoiserInput] = None,
        cache: Optional[Dict[str, Any]] = None,
        running_generation: Optional[torch.LongTensor] = None,
        repetition_penalty_context: Optional[torch.LongTensor] = None,
        inputs_offset: Optional[int] = 0,
        logits_processor_inputs_offset: Optional[int] = None,
        length_penalty_prefix_lengths: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        sample_indices: Optional[Tuple[int, int]] = None,
        input_indices: Optional[Tuple[int, int]] = None,
        return_updated_cache: bool = False,
        cache_len: Optional[int] = None,
        window_size: int = 0,
        block_size: int = 0,
        confidence_state: Optional[Dict[str, torch.Tensor]] = None,
        active_decode_len: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.FloatTensor], Dict[str, Any]]:
        cache = cache if cache is not None else {}
        backbone_cache = {
            k: v
            for k, v in cache.items()
            if k
            not in {
                "_setdlm_kv_cache_position_ids",
                "_setdlm_kv_cache_semantically_cropped",
            }
        }
        backbone_output = self._backbone_forward(
            denoiser_inputs,
            fix_cache_length=(
                True if not return_updated_cache else False
            ),  # Do not let kv cache grow on each forward call
            **backbone_cache,
            **kwargs,
        )
        if isinstance(backbone_output, torch.Tensor):
            logits = backbone_output
        else:
            backbone_output = {k: v for k, v in backbone_output.items()}
            logits = backbone_output.pop("logits", None)
            if logits is None:
                raise ValueError("Backbone output must include logits.")
            cache = cache | backbone_output
        model_output = logits
        prefix_lengths_cover_full_window = (
            length_penalty_prefix_lengths is not None
            and length_penalty_prefix_lengths.shape[-1] == model_output.shape[1]
        )
        active_prefix_len = 0
        if cache_len is not None:
            active_prefix_len = (
                int(cache_len.item()) if hasattr(cache_len, "item") else int(cache_len)
            )
            model_output = model_output[:, active_prefix_len:]
            denoiser_inputs.xt = denoiser_inputs.xt[:, active_prefix_len:]
            if (
                length_penalty_prefix_lengths is not None
                and active_prefix_len > 0
                and prefix_lengths_cover_full_window
            ):
                length_penalty_prefix_lengths = length_penalty_prefix_lengths[
                    ..., active_prefix_len:
                ]
        elif active_decode_len is not None:
            pass
        elif sample_indices is not None and input_indices is not None:
            relative_sample_indices = self._relative_sample_positions(
                sample_indices=sample_indices,
                input_indices=input_indices,
                local_len=model_output.shape[1],
            )
            model_output = model_output[:, relative_sample_indices, :]
            denoiser_inputs.xt = denoiser_inputs.xt[..., relative_sample_indices]
        else:
            model_output = model_output[:, sample_indices - sample_indices[0], :]
            denoiser_inputs.xt = denoiser_inputs.xt[
                ..., sample_indices - sample_indices[0]
            ]  # truncate any extra padding tokens

        if active_decode_len is not None:
            active_decode_len = (
                int(active_decode_len.item())
                if hasattr(active_decode_len, "item")
                else int(active_decode_len)
            )
            model_output = model_output[:, :active_decode_len]
            denoiser_inputs.xt = denoiser_inputs.xt[..., :active_decode_len]
            if length_penalty_prefix_lengths is not None:
                length_penalty_prefix_lengths = length_penalty_prefix_lengths[
                    ..., :active_decode_len
                ]
            active_position_len = active_prefix_len + active_decode_len
            for key in ("position_ids", "permutation_order"):
                value = denoiser_inputs.backbone_kwargs.get(key)
                if value is not None and value.shape[-1] > active_position_len:
                    denoiser_inputs.backbone_kwargs[key] = value[
                        ..., :active_position_len
                    ]
        logits = model_output

        if logits_processor is not None and len(logits_processor) > 0:
            log_x_theta = logits
            sample_idx = (
                sample_indices[0] if sample_indices.ndim == 2 else sample_indices
            )
            repetition_processor_input_ids = (
                repetition_penalty_context
                if repetition_penalty_context is not None
                else running_generation
            )
            processor_running_generation = running_generation
            length_penalty_prefix_lengths_for_processor = None
            if length_penalty_prefix_lengths is not None:
                length_penalty_prefix_lengths_for_processor = (
                    length_penalty_prefix_lengths.to(
                        device=log_x_theta.device, dtype=torch.long
                    )
                )
                if length_penalty_prefix_lengths_for_processor.ndim == 1:
                    length_penalty_prefix_lengths_for_processor = (
                        length_penalty_prefix_lengths_for_processor.unsqueeze(0)
                    )
                if (
                    length_penalty_prefix_lengths_for_processor.shape[0] == 1
                    and log_x_theta.shape[0] > 1
                ):
                    length_penalty_prefix_lengths_for_processor = (
                        length_penalty_prefix_lengths_for_processor.expand(
                            log_x_theta.shape[0], -1
                        )
                    )
                if (
                    length_penalty_prefix_lengths_for_processor.shape[0]
                    == log_x_theta.shape[0]
                    and length_penalty_prefix_lengths_for_processor.shape[1]
                    > log_x_theta.shape[1]
                ):
                    length_penalty_prefix_lengths_for_processor = (
                        length_penalty_prefix_lengths_for_processor[
                            ..., : log_x_theta.shape[1]
                        ]
                    )
                if length_penalty_prefix_lengths_for_processor.shape != log_x_theta.shape[:2]:
                    raise ValueError(
                        "length_penalty_prefix_lengths must have shape "
                        "[batch, decode_len]."
                    )
            target_relative_sample_idx = sample_idx
            processor_inputs_offset = (
                inputs_offset
                if logits_processor_inputs_offset is None
                else logits_processor_inputs_offset
            )
            if processor_inputs_offset is not None:
                inputs_offset_value = (
                    int(processor_inputs_offset.item())
                    if hasattr(processor_inputs_offset, "item")
                    else int(processor_inputs_offset)
                )
                if inputs_offset_value > 0:
                    target_relative_sample_idx = sample_idx - inputs_offset_value
                    # Some callers pass the full absolute sequence as context, while
                    # SetDLM/MDLM seq2seq pass only the target span. In both cases the
                    # length penalty should see generated target tokens only.
                    max_sample_idx = int(sample_idx.max().item())
                    if processor_running_generation.shape[-1] > max_sample_idx:
                        processor_running_generation = processor_running_generation[
                            ..., inputs_offset_value:
                        ]
            for lp in logits_processor:
                if isinstance(lp, MinNewTokensLengthLogitsProcessor):
                    eos_token_id = getattr(lp, "eos_token_id", None)
                    if isinstance(eos_token_id, torch.Tensor):
                        lp.eos_token_id = eos_token_id.to(device=log_x_theta.device)
                for j in range(log_x_theta.shape[1]):
                    if isinstance(lp, (ExponentialDecayLengthPenalty, MinNewTokensLengthLogitsProcessor)):
                        if length_penalty_prefix_lengths_for_processor is not None:
                            prefix_lengths = length_penalty_prefix_lengths_for_processor[
                                :, j
                            ].clamp(
                                min=0,
                                max=processor_running_generation.shape[-1],
                            )
                            if bool(torch.all(prefix_lengths == prefix_lengths[0]).item()):
                                prefix_len = int(prefix_lengths[0].item())
                                log_x_theta[:, j] = lp(
                                    input_ids=processor_running_generation[
                                        ..., :prefix_len
                                    ],
                                    scores=log_x_theta[:, j],
                                )
                            else:
                                row_scores = []
                                for row_idx, row_prefix_len in enumerate(prefix_lengths):
                                    prefix_len = int(row_prefix_len.item())
                                    row_scores.append(
                                        lp(
                                            input_ids=processor_running_generation[
                                                row_idx : row_idx + 1, :prefix_len
                                            ],
                                            scores=log_x_theta[
                                                row_idx : row_idx + 1, j
                                            ],
                                        )
                                    )
                                log_x_theta[:, j] = torch.cat(row_scores, dim=0)
                        else:
                            prefix_len = int(
                                target_relative_sample_idx[j]
                                .clamp(
                                    min=0,
                                    max=processor_running_generation.shape[-1],
                                )
                                .item()
                            )
                            log_x_theta[:, j] = lp(
                                input_ids=processor_running_generation[..., :prefix_len],
                                scores=log_x_theta[:, j],
                            )
                    else:
                        lp_input_ids = (
                            repetition_processor_input_ids
                            if lp.__class__.__name__ == "RepetitionPenaltyLogitsProcessor"
                            else running_generation
                        )
                        log_x_theta[:, j] = lp(
                            input_ids=lp_input_ids,
                            scores=log_x_theta[:, j],
                        )
            # renormalize
            log_x_theta[..., self.mask_token_id] = self.neg_infinity
            log_x_theta = log_x_theta - torch.logsumexp(
                log_x_theta, dim=-1, keepdim=True
            )
        else:
            log_x_theta = self._forward(logits, denoiser_inputs, **kwargs)

        confidence_updates: Dict[str, torch.Tensor] = {}
        x_theta = log_x_theta.exp()

        # nucleus sampling
        if generation_config.nucleus_p < 1.0:
            x_theta = self._nucleus_sample(x_theta, generation_config.nucleus_p)

        sample_confidence = None
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
            sample_confidence = q_xs.gather(-1, xs[..., None]).squeeze(dim=-1)
            output = torch.where(
                (denoiser_inputs.xt != self.mask_token_id).bool(),  # type: ignore
                denoiser_inputs.xt,
                xs,
            )
        elif sampling_strategy == "predict_and_noise":
            # Predict
            xs = self._sample_categorical(x_theta, generation_config.do_sample)
            xs_probs = x_theta.gather(-1, xs[..., None]).squeeze(dim=-1)
            sample_confidence = xs_probs
            output = xs.clone()

            # Noise
            est_noise_indices_next = (next_t * block_size).round().to(torch.int)
            est_noise_indices_curr = (t * block_size).round().to(torch.int)
            num_to_decode = est_noise_indices_curr - est_noise_indices_next
            if generation_config.confidence_based_noising:
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
        if confidence_state is not None:
            confidence_state.clear()
            confidence_state.update(confidence_updates)
            if sample_confidence is not None:
                confidence_state["sample_confidence"] = sample_confidence.detach()
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
        accumulated_confidence = torch.full(
            accumulated_samples.shape,
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        accumulated_confidence[accumulated_samples != self.mask_token_id] = 1.0
        confidence_state: Dict[str, torch.Tensor] = {}

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
        fused_block_cache = self._can_use_fused_block_cache(
            generation_config=generation_config,
            is_infill_task=is_infill_task,
            mdlm_inference=mdlm_inference,
            block_size=block_size,
        )
        pending_cache_inputs = None

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
        logits_processor_inputs_offset = first_mask_token_idx
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
                end_sample_idx = min(
                    inputs_offset + ((block_id + 1) * block_size),
                    accumulated_samples.shape[-1],
                )
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
                if fused_block_cache and pending_cache_inputs is not None:
                    cache = self.update_cache(
                        inputs=pending_cache_inputs,
                        cache=cache,
                    )
                    pending_cache_inputs = None
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
                return_updated_cache = False
                fused_prefix_len = None
                fused_active_decode_len = None
                fused_cache_keep_len = None
                if fused_block_cache and pending_cache_inputs is not None and i == 0:
                    fused_xt = torch.cat([pending_cache_inputs, xt], dim=-1)
                    cache, fused_cache_len = self._trim_cache_for_fused_block(
                        cache=cache,
                        fused_input_len=fused_xt.shape[-1],
                    )
                    fused_prefix_len = pending_cache_inputs.shape[-1]
                    fused_active_decode_len = xt.shape[-1]
                    fused_cache_keep_len = fused_cache_len + fused_prefix_len
                    fused_attention_mask = self._build_fused_block_cache_attention_mask(
                        batch_size=fused_xt.shape[0],
                        cache_len=fused_cache_len,
                        prefix_len=fused_prefix_len,
                        decode_len=fused_active_decode_len,
                        device=fused_xt.device,
                    )
                    denoiser_inputs, cache = self._prepare_inputs_inference(
                        input_ids=fused_xt,
                        attention_mask=fused_attention_mask,
                        context=context,
                        cache=cache if generation_config.use_cache else None,
                        return_updated_cache=True,
                    )
                    return_updated_cache = True
                else:
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
                running_generation = (
                    accumulated_samples[:, first_mask_token_idx:last_mask_token_idx]
                    if not is_infill_task
                    else accumulated_samples[:, : input_indices[-1] + 1]
                )
                repetition_penalty_context = None
                if is_infill_task and getattr(
                    generation_config,
                    "infill_repetition_penalty_include_right_context",
                    False,
                ):
                    repetition_penalty_context = self._visible_infill_context(
                        accumulated_samples=accumulated_samples,
                        mask_token_id=self.mask_token_id,
                        pad_token_id=self.pad_token_id,
                    )

                length_penalty_prefix_lengths = None
                if logits_processor is not None and len(logits_processor) > 0:
                    length_penalty_prefix_lengths = self._length_penalty_prefix_lengths(
                        accumulated_samples=accumulated_samples,
                        sample_indices=sample_indices,
                        target_start_idx=first_mask_token_idx,
                        mask_token_id=self.mask_token_id,
                        pad_token_id=self.pad_token_id,
                    )

                generation_output = self._generate_unconditional(
                    generation_config=generation_config,
                    block_size=block_size,
                    t=t,
                    next_t=next_t,
                    denoiser_inputs=denoiser_inputs,
                    cache=cache,
                    running_generation=running_generation,  # type: ignore
                    repetition_penalty_context=repetition_penalty_context,
                    inputs_offset=inputs_offset,
                    logits_processor_inputs_offset=logits_processor_inputs_offset,
                    length_penalty_prefix_lengths=length_penalty_prefix_lengths,
                    logits_processor=logits_processor,
                    tokenizer=tokenizer,
                    sample_indices=sample_indices,
                    input_indices=input_indices,
                    return_updated_cache=return_updated_cache,
                    cache_len=fused_prefix_len,
                    active_decode_len=fused_active_decode_len,
                    confidence_state=confidence_state,
                    **kwargs,
                )

                xs, cache = generation_output
                if fused_cache_keep_len is not None:
                    cache = self._crop_cache_to_length(cache, fused_cache_keep_len)
                    pending_cache_inputs = None
                sample_confidence = confidence_state.get("sample_confidence")
                block_pbar.set_postfix(
                    NFEs=total_NFEs,
                    block_NFEs=block_NFEs,
                )
                num_tokens_generated_per_step.append(
                    (xs != denoiser_inputs.xt).sum().item()
                )
                if input_indices is not None:
                    relative_sample_indices = self._relative_sample_positions(
                        sample_indices=sample_indices,
                        input_indices=input_indices,
                        local_len=xt.shape[-1],
                    )
                    xt[..., relative_sample_indices] = xs
                    if (xt[..., relative_sample_indices] == self.mask_token_id).sum().item() == 0:
                        break
                else:
                    xt = xs
                if input_indices is not None:
                    sample_values = xt[..., relative_sample_indices]
                    scatter_indices = sample_indices.to(
                        device=accumulated_samples.device,
                        dtype=torch.long,
                    )[None, :]
                    accumulated_samples.scatter_(
                        dim=-1,
                        index=scatter_indices,
                        src=sample_values,
                    )
                    if sample_confidence is not None:
                        confidence_values = sample_confidence.to(accumulated_confidence)
                        confidence_values = torch.where(
                            sample_values != self.mask_token_id,
                            confidence_values,
                            torch.full_like(confidence_values, float("nan")),
                        )
                        accumulated_confidence.scatter_(
                            dim=-1,
                            index=scatter_indices,
                            src=confidence_values,
                        )
                else:
                    sample_values = xt[:, -sample_indices.shape[-1] :]
                    scatter_indices = sample_indices.to(
                        device=accumulated_samples.device,
                        dtype=torch.long,
                    )[None, :]
                    accumulated_samples.scatter_(
                        dim=-1, index=scatter_indices, src=sample_values
                    )
                    if sample_confidence is not None:
                        confidence_values = sample_confidence.to(accumulated_confidence)
                        if confidence_values.shape[-1] != sample_indices.shape[-1]:
                            confidence_values = confidence_values[
                                ..., -sample_indices.shape[-1] :
                            ]
                        confidence_values = torch.where(
                            sample_values != self.mask_token_id,
                            confidence_values,
                            torch.full_like(confidence_values, float("nan")),
                        )
                        accumulated_confidence.scatter_(
                            dim=-1,
                            index=scatter_indices,
                            src=confidence_values,
                        )
                if ((xt == self.mask_token_id).sum().item() == 0) or (
                    pad_length is not None
                    and pad_length > 0
                    and mdlm_inference
                    and (xt[:, :-pad_length] == self.mask_token_id).sum().item() == 0
                ):
                    break
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=accumulated_samples[  # type: ignore
                        :,
                        : sample_indices[-1] + 1,
                    ],
                    scores=None,  # type: ignore
                    token_confidence=accumulated_confidence[
                        :,
                        : sample_indices[-1] + 1,
                    ],
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
                if fused_block_cache:
                    pending_cache_inputs = (
                        xt.detach().clone() if block_id < max_blocks - 1 else None
                    )
                else:
                    cache = self.update_cache(
                        inputs=xt,
                        cache=cache,
                    )
        parallelism_factor = (
            sum(num_tokens_generated_per_step) / len(num_tokens_generated_per_step)
            if len(num_tokens_generated_per_step) > 0
            else 0
        )
        accumulated_samples = accumulated_samples[
            accumulated_samples != self.mask_token_id
        ].unsqueeze(0)
        if return_dict_in_generate:
            return DiffusionGenerationOutput(
                sequences=accumulated_samples,
                parallelism_factor=parallelism_factor,
            )
        return accumulated_samples  # type: ignore
