from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import (
    DynamicCache,
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import ModelOutput

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
    )
except ImportError:
    BlockMask, create_block_mask = None, None


from src.denoiser.denoiser import (
    Denoiser,
    DenoiserConfig,
    DenoiserInput,
    LossAndNllOutput,
)


class DiffusionGenerationConfig(GenerationConfig):
    def __init__(
        self,
        num_steps: int = 1000,
        min_t: float = 1e-5,
        block_size: int | None = None,
        first_hitting: bool = False,
        sampling_strategy: Literal["posterior", "predict_then_noise"] = "posterior",
        confidence_based_noising: bool = False,
        use_model_output_cache: bool = True,
        **kwargs,
    ):
        """Generation config with additional parameters relevant for diffusion model
            sampling.

        Args:
            num_steps (int): Number of diffusion / iterative refinement steps.
                Defaults to 1000.
            min_t (float): Minimum time to use.
                Diffusion models use t=1 for noise and t=0 for signal.
                Setting t=0 exactly can lead to certain numerical instabilities.
                Defaults to 1e-5.
            block_size (int): Block size to use for semi-autoregressive decoding.
                Defaults to None (in which case block_size is set to max_new_tokens).
            first_hitting (bool): Whether to use first hitting sampler.
                When set to true, rather than following the diffusion time and sampling
                from posterior, which can result in no tokens changing between steps,
                e.g., for masked diffusion, we explicitly determine the next time step
                at which a token will be decoded / generated.
                Note: this will negate the `num_steps` parameter, as we will decode one
                token at a time, hence, when True, num_steps = seq_length
                (or block_size, for semi-autoregressive).
                See https://arxiv.org/abs/2409.02908 for details.
                Defaults to False.
            sampling_strategy (str): Method for transitioning between latents.
                Options:
                    - "posterior" - Compute and sample from the posterior
                        q(x_s | x_t, x_theta).
                    - "predict_then_noise" - Sample from the denoising model x_theta,
                        then add back noise to produce x_s.
                Defaults to "posterior".
            confidence_based_noising (bool): When using the "predict_then_noise"
                strategy, whether to add noise to random positions or to those that have
                the lowest probability under x_theta.
                Defaults to False.
            use_model_output_cache (bool): Whether to re-use model's output, if sequence
                is unchanged, because if xt == xs, we can simply re-use the denoising
                model's outputs and save a function evaluation.
                Relevant if model.backbone is not time/noise-conditioned.
                Defaults to True.
            kwargs: Keyword arguments passed to `GenerationConfig`.
        """
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.min_t = min_t
        # TODO: assumes we are setting max_new_tokens, which may not be the case!
        self.block_size = block_size if block_size is not None else self.max_new_tokens
        self.first_hitting = first_hitting
        self.sampling_strategy = sampling_strategy
        self.confidence_based_noising = confidence_based_noising
        self.use_model_output_cache = use_model_output_cache


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
        keep_clean_bos: bool | None = None,  # Whether to enforce un-noised BOS token
        # Logits @ position i predicts token @ position i+1 (as in AR models)
        shift_logits: bool | None = None,
        T: int = 1000,
        diffusion_type: Literal["absorbing", "uniform"] = "absorbing",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keep_clean_bos = keep_clean_bos
        self.shift_logits = shift_logits
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

    @staticmethod
    def _sample_generation_timesteps(
        generation_config: DiffusionGenerationConfig,
        max_length: int | None = None,
        device: str | None = None,
    ) -> Tensor:
        """Sample timesteps for the diffusion process."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if max_length is None:
            max_length = generation_config.max_new_tokens

        if generation_config.first_hitting:
            timesteps = Tensor([1.0])
            for i in range(max_length, 0, -1):
                u = torch.rand(1)
                next_t = timesteps[-1] * u ** (1 / i)
                timesteps = torch.cat((timesteps, next_t), dim=0)
            return timesteps[1:].to(device)
        timesteps = torch.linspace(
            1.0,
            generation_config.min_t,
            generation_config.num_steps + 1,
            device=device,
        )
        return timesteps

    def _generate_unconditional(  # TODO add CBG and CFG generation
        self,
        generation_config: DiffusionGenerationConfig,
        alpha_t: Tensor,
        alpha_s: Tensor,
        denoiser_inputs: DenoiserInput | None = None,
        cache: Dict[str, Tensor] | None = None,
        past_key_values: DynamicCache | None = None,
        running_generation: Tensor | None = None,
        logits_processor: LogitsProcessorList | None = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if cache is None:  # execute function evaluation
            backbone_output = self._backbone_forward(
                denoiser_inputs, past_key_values=past_key_values, **kwargs
            )
            if isinstance(backbone_output, ModelOutput) and hasattr(
                backbone_output, "logits"
            ):
                backbone_output = backbone_output.logits
            log_x_theta = self._forward(backbone_output, denoiser_inputs, **kwargs)
            if logits_processor is not None and running_generation is not None:
                for token_idx in range(log_x_theta.shape[1]):
                    # TODO: Looping over token positions like this does not allow for
                    #   some processors, e.g. length penalty which could be applied all
                    #   at once to the entire block, to be applied in parallel.
                    log_x_theta[:, token_idx] = logits_processor(
                        input_ids=running_generation,
                        scores=log_x_theta[:, token_idx],
                    )
                log_x_theta = torch.log_softmax(log_x_theta, dim=-1)  # re-normalize
            x_theta = log_x_theta.exp()
        else:
            x_theta = cache["x_theta"]
        cache = {"x_theta": x_theta}
        if generation_config.sampling_strategy == "posterior":
            q_xs = self._compute_posterior(
                x_theta, denoiser_inputs.xt, alpha_t, alpha_s
            )
            assert abs((q_xs.sum() / denoiser_inputs.xt.numel()).item() - 1.0) < 1e-6, (
                "Posterior probabilities not summing to 1."
            )
            assert bool(q_xs.isnan().sum() > 0), "NaN found in the posterior."
            xs = self._sample_categorical(q_xs, generation_config.do_sample)
        elif generation_config.sampling_strategy == "predict_and_noise":
            assert (
                abs((x_theta.sum() / denoiser_inputs.xt.numel()).item() - 1.0) < 1e-6
            ), "Denoising output probabilities not summing to 1."
            assert bool(x_theta.isnan().sum() > 0), "NaN found in the denoising output."
            assert self.config.diffusion_type == "absorbing", (
                "predict_and_noise sampling strategy only implemented for absorbing"
                " state diffusion."
            )
            # Predict
            xs = self._sample_categorical(x_theta, generation_config.do_sample)
            # Noise
            num_noise_indices = (1 - alpha_s) * xs.shape[-1]
            if generation_config.confidence_based_noising:
                conf = -x_theta.gather(-1, xs[..., None]).squeeze(-1)
                conf = torch.where(  # already decoded tokens have 'inf' confidence
                    (denoiser_inputs.xt == self.mask_token_id).bool(),  # type: ignore
                    conf,
                    torch.inf,
                )
                noise_indices = -conf.argmax(dim=-1).sort(dim=-1)[0][
                    ..., :num_noise_indices
                ]
            else:
                # TODO: implement this
                raise NotImplementedError
            # TODO: this is MDLM-specific
            xs[..., noise_indices] = self.mask_token_id
        else:
            raise NotImplementedError(
                f"Sampling strategy {generation_config.sampling_strategy} not"
                " implemented."
            )
        return xs, cache

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[Tensor] = None,
        generation_config: Optional[DiffusionGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        disable_pbar: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        # Setup sampling variables
        if generation_config is None:
            assert getattr(self, "generation_config", None) is not None, (
                "Generation config must be provided if not present in the model."
            )
            generation_config = self.generation_config
        if inputs is None:
            inputs = torch.ones((batch_size, 1), device=device) * self.bos_token_id
        if max_length is None:
            if hasattr(generation_config, "max_length"):
                max_length = generation_config.max_length
            else:
                max_length = self.max_length
        if max_new_tokens is None:
            if hasattr(generation_config, "max_new_tokens"):
                max_new_tokens = generation_config.max_new_tokens
            else:
                max_new_tokens = max_length - inputs.shape[-1]
        batch_size = batch_size if batch_size is not None else inputs.shape[0]
        assert batch_size == 1, "Batched sampling not supported yet"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        block_size = generation_config.block_size
        max_blocks = max_new_tokens // block_size

        # Sample max generation length tensor from prior
        accumulated_samples = self._sample_prior(
            device=device,
            batch_size=batch_size,
            length=max_blocks * block_size,
        )
        accumulated_samples = torch.cat([inputs, accumulated_samples], dim=-1)
        if generation_config.use_cache:
            past_key_values = self.update_past_key_values(
                inputs=inputs,
                past_key_values=DynamicCache(),
            )
            inputs_offset = inputs.shape[-1]
        else:
            past_key_values = None
            inputs_offset = 0

        logit_offset = 1 if self.config.shift_logits else 0
        total_NFEs = 0
        block_pbar = tqdm(
            range(max_blocks),
            desc="Sampling blocks",
            leave=False,
            disable=disable_pbar,
        )
        for block_id in block_pbar:
            block_NFEs = 0
            xt = accumulated_samples[
                :,
                inputs_offset + (block_id * block_size) : inputs_offset
                + ((block_id + 1) * block_size)
                + logit_offset,
            ]
            timesteps = self._sample_generation_timesteps(
                generation_config, max_length=block_size, device=device
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
            context = (
                accumulated_samples[:, : (block_id * block_size)]
                if block_id > 0 and not generation_config.use_cache
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
                    context=context,
                    past_key_values=past_key_values,
                )

                running_generation = (  # Used for logit processing
                    accumulated_samples[
                        :, inputs_offset : (block_id * block_size) + logit_offset
                    ],
                )
                xs, cache = self._generate_unconditional(
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                    denoiser_inputs=denoiser_inputs,
                    cache=cache,
                    xt=xt,
                    past_key_values=past_key_values,
                    running_generation=running_generation,
                    logits_processor=logits_processor,
                    **kwargs,
                )

                if self.config.shift_logits:
                    xs = torch.cat((xt[:, :1], xs), dim=-1)

                block_pbar.set_postfix(
                    NFEs=total_NFEs,
                    block_NFEs=block_NFEs,
                )

                if (
                    not torch.allclose(xs, xt)
                    or not generation_config.use_model_output_cache
                ):
                    cache = None
                xt = xs
            accumulated_samples[
                :,
                inputs_offset + (block_id * block_size) + logit_offset : inputs_offset
                + ((block_id + 1) * block_size)
                + logit_offset,
            ] = xt[:, 1:]
            if tokenizer is not None:
                print(tokenizer.batch_decode(accumulated_samples))
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=accumulated_samples[:, inputs_offset:],
                    scores=None,  # type: ignore
                )
                if torch.any(is_done):
                    accumulated_samples = accumulated_samples[
                        :, : ((block_id + 1) * block_size) + logit_offset
                    ]
                    break
            if generation_config.use_cache:
                if self.config.shift_logits:
                    xt = xt[:, :-1]
                past_key_values = self.update_past_key_values(
                    inputs=xt,
                    past_key_values=past_key_values,
                )
        return accumulated_samples


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
            assert self.config.block_size == self.config.max_length, (
                "Only MDLM supported as decoder-only"
            )
            static_mask = torch.full(
                (self.config.max_length, self.config.max_length), fill_value=True
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
                    Q_LEN=self.config.max_length,
                    KV_LEN=self.config.max_length,
                )
                decoder_static_mask = create_block_mask(
                    partial(
                        self._decoder_block_mask,
                        block_size=self.config.block_size,
                        seq_length=self.config.max_length,
                    ),
                    B=None,
                    H=None,
                    Q_LEN=self.config.max_length,
                    KV_LEN=self.config.max_length * 2,
                )
            else:
                encoder_static_mask = self._encoder_block_mask(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.max_length)[:, None],
                    kv_idx=torch.arange(self.config.max_length)[None, :],
                    block_size=self.config.block_size,
                )
                decoder_static_mask = self._decoder_block_mask(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.max_length)[:, None],
                    kv_idx=torch.arange(self.config.max_length * 2)[None, :],
                    block_size=self.config.block_size,
                    seq_length=self.config.max_length,
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

    def _prepare_inputs_inference(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        past_key_values: Tensor | None = None,
        **backbone_kwargs: Any,
    ) -> DenoiserInput:
        device = input_ids.device
        batch_size = input_ids.shape[0]
        if self.config.backbone_is_decoder_only:
            return super()._prepare_inputs_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                past_key_values=past_key_values,
                **backbone_kwargs,
            )
        # Encoder-decoder inputs
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        position_ids, encoder_position_ids = None, None
        if past_key_values is not None:
            cache_len = self._get_past_key_values_seq_length(past_key_values)
            if input_ids is not None:  # Skip enc: nothing new for enc to cache
                full_seq_length = cache_len + input_ids.shape[1]
                encoder_attention_mask = None
                position_ids = torch.arange(cache_len, full_seq_length).to(device)[
                    None, :
                ]
            else:  # Caching new tokens in the enc
                full_seq_length = cache_len + context.shape[-1]
                encoder_attention_mask = self.encoder_static_attention_mask[
                    None, cache_len:full_seq_length, :full_seq_length
                ]
                encoder_position_ids = torch.arange(cache_len, full_seq_length).to(
                    device
                )[None, :]
        else:  # Caching context for the first time / not using kv-cache at all
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
            decoder_attention_mask = torch.ones(
                (batch_size, input_ids.shape[1], full_seq_length),
                device=device,
            )
        else:
            decoder_attention_mask = None
        return DenoiserInput(
            xt=input_ids,
            attention_mask=decoder_attention_mask,
            context_mask=context_mask,
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
