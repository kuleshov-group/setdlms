from functools import partial
from typing import Any, Dict, Literal, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import (
    DynamicCache,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import ModelOutput

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
except ModuleNotFoundError:
    BlockMask = None

from src.denoiser.denoiser import (
    Denoiser,
    DenoiserConfig,
    DenoiserInput,
    LossAndNllOutput,
)


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
        x0: Tensor,
        alpha_t: Tensor,
        context_mask: Tensor,
    ) -> Tensor:
        """Sample from the pre-defined forward / noising process.

        Parameters:
            x0 (Tensor): Signal / data sample;
                can potentially include context tokens.
            alpha_t (Tensor): Amount of signal to retain.
            context_mask (Tensor): Indicator of context tokens (to remain
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
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
        t: Tensor | None = None,
        past_key_values: Tensor | None = None,
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
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
        past_key_values: Tensor | None = None,
        **kwargs: Any,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
        )

    def _compute_loss(
        self, model_output: Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
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
        x: Tensor,
        xt: Tensor,
        alpha_t: Tensor,
        alpha_s: Tensor,
    ) -> Tensor:
        """Computes posterior / approximate posterior q(x_s | x_t, x),
            where x represents clean sequence (as one-hots) or the output of the
            denoising model.

        Args:
            x (Tensor): True (one-hot) / predicted clean signal (B, L, V).
            xt (Tensor): Noised signal at time t (B, L).
            alpha_t (Tensor): Noise schedule parameter at time t (B, 1, 1).
            alpha_s (Tensor): Noise schedule parameter at time s (B, 1, 1).
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
        self,
        max_seq_length: int | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Sample timesteps for the diffusion process.
        Args:
            eps (float): Small value to avoid division by zero.
            num_steps (int): Number of timesteps to sample.
            device (str | None): Device to use for sampling.
        Returns:
            Tensor: Sampled timesteps.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.sampler_config.first_hitting:
            if max_seq_length is None:
                raise ValueError("max_seq_length must be provided for first hitting.")
            timesteps = Tensor([1.0])
            for i in range(max_seq_length, 0, -1):
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

    def _logit_transform(
        self,
        logits: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Transform logits using various techniques.
        Args:
            logits (Tensor): Logits to transform.

        Returns:
            Tensor: Transformed logits.
        """
        if self.sampler_config.top_p < 1.0:  #  Nucleus sampling
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
        xs: Tensor,
        q_xs: Tensor,
        xt: Tensor,
    ) -> Tensor:
        """
        Remask the sampled sequence based on different strategies.
        Args:
            xs (Tensor): Sampled sequence.
            q_xs (Tensor): Posterior distribution.
            xt (Tensor): Masked sequence.
        Returns:
            Tensor: Remasked tokens.
        """
        # TODO implement remdm
        if self.config.shift_logits:
            xt = xt[:, 1:]

        if self.sampler_config.first_hitting:
            # unmask a token (among currently masked tokens)
            num_masked = (xt == self.mask_token_id).sum(-1)
            if num_masked == 0:
                return xs

            if self.sampler_config.low_confidence_remasking:
                # select the index with the highest confidence
                xs_q = q_xs.gather(-1, xs[..., None]).squeeze(-1)
                xs_q[xt != self.mask_token_id] = 0
                ind = xs_q.argmax(dim=-1)
            else:
                # uniformly select an index (among masked tokens)
                ind = torch.randint(0, num_masked.item(), (xs.shape[0],))
                ind = (xt == self.mask_token_id).nonzero()[ind, 1]

            unmask_flag = torch.arange(xt.shape[-1], device=xt.device) == ind[None, :]
            # if a token is already unmasked, don't apply remasking
            unmask_flag = unmask_flag | (xt != self.mask_token_id)

            # remask tokens not selected
            xs[~unmask_flag] = self.mask_token_id

        return xs

    def _generate_unconditional(  # TODO add CBG and CFG generation
        self,
        alpha_t: Tensor,
        alpha_s: Tensor,
        denoiser_inputs: DenoiserInput | None = None,
        cache: Dict[str, Tensor] | None = None,
        past_key_values: DynamicCache | None = None,
        context: Tensor | None = None,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        regulation_start: int = -1,
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
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
            if (
                repetition_penalty != 1.0
                and context is not None
                and context.numel() > 0
            ):
                for token_idx in range(backbone_output.shape[1]):
                    score = torch.gather(backbone_output[:, token_idx], 1, context)
                    score = torch.where(
                        score < 0,
                        score * repetition_penalty,
                        score / repetition_penalty,
                    )
                    backbone_output[:, token_idx] = backbone_output[
                        :, token_idx
                    ].scatter(1, context, score)
            if length_penalty != 1.0 and regulation_start >= 0:
                cur_len = context.shape[-1]
                if cur_len > regulation_start:
                    penalties = torch.zeros_like(backbone_output)
                    penalty_idx = cur_len - regulation_start
                    penalty = torch.abs(backbone_output[..., self.eos_token_id]) * (
                        pow(length_penalty, penalty_idx) - 1
                    )
                    penalties[..., self.eos_token_id] = penalty
                    backbone_output = backbone_output + penalties

            log_x_theta = self._forward(
                backbone_output,
                denoiser_inputs,
            )  # should be the log(x_\theta) with the shape of (B, Seq, Vocab)
            x_theta = log_x_theta.exp()
            x_theta = self._logit_transform(
                logits=x_theta,
                **kwargs,
            )
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
        context: Tensor | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        disable_pbar: bool = False,
        **kwargs: Any,
    ) -> Tuple[Tensor, int]:
        max_length = (
            max_length if max_length is not None else self.sampler_config.max_length
        )
        batch_size = (
            batch_size if batch_size is not None else self.sampler_config.batch_size
        )
        assert batch_size == 1, "Batched sampling not supported yet"
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

        block_size = self.sampler_config.block_size
        max_blocks = max_length // block_size
        total_NFEs = 0
        # cache kvs of context
        past_key_values = None
        blocks_to_cache = 0
        context_len = context.shape[-1] if context is not None else 0

        accumulated_samples = self._sample_prior(
            device=device,
            batch_size=batch_size,
            length=max_blocks * block_size,
        )

        if context is not None:
            accumulated_samples[:, :context_len] = context

        if self.sampler_config.kv_caching and context is not None:
            blocks_to_cache = context_len // block_size
            # start sampling with the last block with clean tokens
            cache_len = block_size * blocks_to_cache
            if context.shape[-1] % block_size == 0:
                blocks_to_cache -= 1
                cache_len -= block_size
            past_key_values = DynamicCache()
            past_key_values = self.update_kv_cache(
                context=accumulated_samples[:, :cache_len],
                past_key_values=past_key_values,
            )

        block_pbar = tqdm(
            range(blocks_to_cache, max_blocks),
            desc="Sampling blocks",
            leave=False,
            disable=disable_pbar,
        )
        for block_id in block_pbar:
            block_NFEs = 0
            if self.config.shift_logits:
                xt = accumulated_samples[
                    :, (block_id * block_size) : ((block_id + 1) * block_size) + 1
                ]
            else:
                xt = accumulated_samples[
                    :, (block_id * block_size) : ((block_id + 1) * block_size)
                ]
            timesteps = self._sample_generation_timesteps(
                max_seq_length=block_size, device=device
            )
            step_pbar = tqdm(
                timesteps,
                desc="T",
                total=timesteps.shape[0],
                leave=False,
                disable=disable_pbar,
            )
            dt = (1 - self.sampler_config.min_t) / len(timesteps)
            cache = None
            context_block = (
                accumulated_samples[:, : (block_id * block_size)]
                if block_id > 0
                else None
            )

            for t in step_pbar:
                if cache is None:
                    block_NFEs += 1
                    total_NFEs += 1
                # t is 0-dim tensor, reshape to (1, 1, 1) for broadcasting
                alpha_t, _ = self.noise_schedule(t)
                alpha_s, _ = self.noise_schedule(t - dt)
                alpha_t = alpha_t[None, None, None]
                alpha_s = alpha_s[None, None, None]
                denoiser_inputs = self._prepare_inputs_inference(
                    input_ids=xt,
                    context=None if self.sampler_config.kv_caching else context_block,
                    past_key_values=past_key_values,
                )

                if self.config.shift_logits:
                    context_for_next_step = accumulated_samples[
                        :, context_len : (block_id * block_size) + 1
                    ]
                else:
                    context_for_next_step = accumulated_samples[
                        :, context_len : (block_id * block_size)
                    ]
                q_xs, cache = self._generate_unconditional(
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                    denoiser_inputs=denoiser_inputs,
                    cache=cache,
                    xt=xt,
                    past_key_values=past_key_values,
                    context=context_for_next_step,
                    **kwargs,
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
            if self.config.shift_logits:
                accumulated_samples[
                    :, (block_id * block_size) + 1 : ((block_id + 1) * block_size) + 1
                ] = xt[:, 1:]
            else:
                accumulated_samples[
                    :, (block_id * block_size) : ((block_id + 1) * block_size)
                ] = xt
            if tokenizer is not None:
                print(tokenizer.batch_decode(accumulated_samples))
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=accumulated_samples[:, context_len:],
                    scores=None,  # type: ignore
                )
                if torch.any(is_done):
                    if self.config.shift_logits:
                        accumulated_samples = accumulated_samples[
                            :, : ((block_id + 1) * block_size) + 1
                        ]
                    else:
                        accumulated_samples = accumulated_samples[
                            :, : ((block_id + 1) * block_size)
                        ]
                    break
            if self.sampler_config.kv_caching:
                if self.config.shift_logits:
                    xt = xt[:, :-1]
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
        self, backbone_output: Tensor, denoiser_inputs: DenoiserInput, **kwargs
    ) -> Tensor:
        if self.config.shift_logits:
            backbone_output = backbone_output[:, :-1, ...]
        # Zero-mask probability
        mask = (
            torch.arange(backbone_output.shape[-1], device=backbone_output.device)
            == self.mask_token_id
        ).view(1, 1, -1)  # unsqueeze for broadcast to (batch, seq_length, vocab_size)
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
        self, model_output: Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
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
    ) -> Tensor:
        """
        Args:
            q_idx (Tensor): Query indices.
            kv_idx (Tensor): Key indices
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
        seq_length: int | None = None,
    ) -> Tensor:
        del b, h

        # Indicate whether token belongs to xt or x0:
        x0_flag_q = (q_idx >= seq_length).bool()
        x0_flag_kv = (kv_idx >= seq_length).bool()

        # Compute block indices
        block_q = torch.where(
            x0_flag_q, (q_idx - seq_length) // block_size, q_idx // block_size
        )
        block_kv = torch.where(
            x0_flag_kv, (kv_idx - seq_length) // block_size, kv_idx // block_size
        )
        # **1. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q == block_kv) & x0_flag_kv & ~x0_flag_q

        # **2. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q > block_kv) & (x0_flag_q == x0_flag_kv)

        # **3. Combine Masks **
        return block_diagonal | offset_block_causal

    def _create_static_mask(self) -> None:
        assert self.config.attn_backend != "flex_attention", (
            "FlexAttention not supported yet"
        )
        if self.config.backbone_is_decoder_only:
            assert self.config.attn_backend != "flex_attention", (
                "FlexAttention not supported yet"
            )
            assert self.config.block_size == self.config.length, (
                "Only MDLM supported as decoder-only"
            )
            static_mask = torch.full(
                (self.config.length, self.config.length), fill_value=True
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
                        seq_length=self.config.length,
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
                    seq_length=self.config.length,
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
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
        t: Tensor | None = None,
        past_key_values: Tensor | None = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)

        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
            context_mask = context_mask.to(torch.int)

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
            decoder_attention_mask = (
                self.static_attention_mask[None, ...]
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

    def _get_past_key_values_seq_length(self, past_key_values):
        seq_length = 0
        for i in range(len(past_key_values)):
            if past_key_values[i][0].shape[0] > 0:
                seq_length = max(past_key_values[i][0].shape[-2], seq_length)
        return seq_length

    def _prepare_inputs_inference(
        self,
        input_ids: Tensor | None = None,
        context: Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        **kwargs: Any,
    ):
        # TODO this assumes encoder-decoder backboneNone
        assert input_ids is not None or context is not None
        device = input_ids.device if input_ids is not None else context.device
        batch_size = input_ids.shape[0] if input_ids is not None else context.shape[0]
        if self.config.backbone_is_decoder_only:
            attention_mask = torch.ones(
                (batch_size, input_ids.shape[1], input_ids.shape[1]),
                device=device,
            )
            return DenoiserInput(
                xt=input_ids,
                attention_mask=attention_mask,
            )
        position_ids, encoder_position_ids = None, None
        if past_key_values is not None:
            cache_len = self._get_past_key_values_seq_length(past_key_values)
            if input_ids is not None:
                full_seq_length = cache_len + input_ids.shape[1]
                encoder_attention_mask = None
                position_ids = torch.arange(cache_len, full_seq_length).to(device)[
                    None, :
                ]
            else:
                full_seq_length = cache_len + context.shape[-1]
                encoder_attention_mask = self.encoder_static_attention_mask[
                    None, cache_len:full_seq_length, :full_seq_length
                ]
                encoder_position_ids = torch.arange(cache_len, full_seq_length).to(
                    device
                )[None, :]
        else:
            if context is not None:
                context_len = context.shape[1]
            else:
                context_len = 0
            if input_ids is not None:
                full_seq_length = context_len + input_ids.shape[1]
            else:
                full_seq_length = context_len
            encoder_attention_mask = self.encoder_static_attention_mask[
                None, :context_len, :context_len
            ]
            position_ids = torch.arange(context_len, full_seq_length).to(device)[
                None, :
            ]
        if input_ids is not None:
            # TODO for profiling index the attn mask
            decoder_attention_mask = torch.ones(
                (batch_size, input_ids.shape[1], full_seq_length),
                device=device,
            )
        else:
            decoder_attention_mask = None
        return DenoiserInput(
            xt=input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            backbone_kwargs={
                "encoder_input_ids": context,
                "encoder_position_ids": encoder_position_ids,
                "encoder_attention_mask": encoder_attention_mask,
                "position_ids": position_ids,
            },
        )


# TODO
# class UDLM(D3PM):
