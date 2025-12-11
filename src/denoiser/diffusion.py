from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache, DynamicCache

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        and_masks,
        create_block_mask,
    )
except ImportError:
    BlockMask, and_masks, create_block_mask = None, None, None


from src.denoiser.base import (
    Denoiser,
    DenoiserConfig,
    DenoiserInput,
    LossAndNllOutput,
)


def create_attn_mask(attn_mask):
    # noinspection PyUnusedLocal
    def padding(b, h, q_idx, kv_idx):
        return attn_mask[b, q_idx] & attn_mask[b, kv_idx]

    return padding


class DiffusionGenerationConfig(GenerationConfig):
    def __init__(
        self,
        num_steps: int = 1000,
        min_t: float = 1e-5,
        block_size: Optional[int] = None,
        first_hitting: bool = False,
        sampling_strategy: Literal["posterior", "predict_then_noise"] = "posterior",
        confidence_based_noising: bool = False,
        confidence_margin_based_noising: bool = False,
        confidence_threshold: float = 1e6,
        align_inputs_to_blocks: bool = True,
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
                        Only implemented for absorbing diffusion.
                Defaults to "posterior".
            confidence_based_noising (bool): When using the "predict_then_noise"
                strategy, whether to add noise to random positions or to those that have
                the lowest probability under x_theta.
                Cannot be used in conjunction with confidence_margin_based_noising.
                Defaults to False.
            confidence_margin_based_noising (bool): When using the "predict_then_noise"
                strategy, whether to add noise to random positions or to those that have
                the lowest probability margins under x_theta, where margin is defined as
                the absolute difference between the top two probabilities at a given
                position.
                See https://arxiv.org/abs/2502.06768 for details.
                Cannot be used in conjunction with confidence_based_noising.
                Defaults to False.
            confidence_threshold (float): Confidence threshold to use for sampling.
                Any tokens that exceed threshold are decoded.
                See https://arxiv.org/abs/2505.22618 for details.
                Defaults to 1e6.
            align_inputs_to_blocks (bool): Whether to align input tokens to block size,
                e.g., for an input of length C and block size S, context will be C // S,
                and generation will begin with a block whose first C % S tokens come
                from the input.
            kwargs: Keyword arguments passed to `GenerationConfig`.
        """
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.min_t = min_t
        # TODO: assumes we are setting max_new_tokens, which may not be the case!
        self.block_size = block_size if block_size is not None else self.max_new_tokens
        self.first_hitting = first_hitting
        if self.first_hitting:
            # TODO: log.warn that this is being overridden
            self.num_steps = min(num_steps, self.block_size)
        self.sampling_strategy = sampling_strategy
        assert not confidence_based_noising or not confidence_margin_based_noising, (
            "Cannot use both `confidence_based_noising` and"
            " `confidence_margin_based_noising`."
        )
        self.confidence_based_noising = confidence_based_noising
        self.confidence_margin_based_noising = confidence_margin_based_noising
        self.confidence_threshold = confidence_threshold
        self.align_inputs_to_blocks = align_inputs_to_blocks


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

    def __init__(self, config: MDLMConfig, tokenizer: Optional[PreTrainedTokenizer] = None, **kwargs):
        super().__init__(config, tokenizer, **kwargs)
        self._create_static_mask()
        self.neg_infinity = -1e12

    def _create_static_mask(self) -> None:
        static_mask = torch.ones(
            self.config.length, self.config.length, dtype=torch.bool
        )
        self.register_buffer(
            "static_attention_mask",
            static_mask,
        )
        self.skip_params_for_push.append("static_attention_mask")

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
        xt = torch.where(
            (move_indices * (1 - mask)).bool(), self.mask_token_id, x0
        )
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
        noise_mask = context_mask & attention_mask
        xt = self._sample_q_xt(
            x0=input_ids,
            alpha_t=alpha_t,
            mask=noise_mask,
        )
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
            cache_length = self._get_past_key_values_seq_length(past_key_values)
            full_seq_length = cache_length + input_ids.shape[-1]
            attention_mask = torch.ones(
                (input_ids.shape[0], 1, input_ids.shape[1], full_seq_length),
                device=input_ids.device,
            )  # Make attention mask 4D
            attention_mask = self._preprocess_attention_mask(
                attention_mask, dtype=torch.float
            )
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            context_mask=context_mask,
            backbone_kwargs=backbone_kwargs | {"use_cache": False},
        ), cache

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
        block_size = getattr(self.config, "block_size", denoiser_inputs.x0.shape[-1])
        if block_size > 1:
            nlls = (
                log_p_theta
                * denoiser_inputs.alpha_t_prime
                / (1 - denoiser_inputs.alpha_t)
                * denoiser_inputs.tokens_mask
            )
        else:
            nlls = - log_p_theta * denoiser_inputs.tokens_mask
        if self.training or block_size == 1:
            batch_nll = -(log_p_theta * denoiser_inputs.tokens_mask).sum(dim=-1)
        else:
            batch_nll = nlls.sum(dim=-1)
        # Average over masked tokens during training
        if (self.training and not getattr(self.config, "train_on_nelbo", False)) or block_size == 1:
            batch_nll = -(log_p_theta * denoiser_inputs.tokens_mask).sum(dim=-1)
            mask_token_indicator = (denoiser_inputs.xt == self.mask_token_id).float()
            count = mask_token_indicator.sum(dim=-1)
            token_nll = torch.where(count > 0, batch_nll / count, torch.zeros_like(batch_nll)).mean()
        else:
            count = denoiser_inputs.tokens_mask.sum(dim=-1)
            token_nll = (batch_nll / count).mean()
        return LossAndNllOutput(
            loss=token_nll,  # type: ignore
            nlls=nlls,
            other_loss_terms={
                "masked_tokens": (denoiser_inputs.xt == self.mask_token_id).int(),
                "log_p_theta": - log_p_theta * denoiser_inputs.tokens_mask,
            },
        )

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
        q_xs = x * (alpha_s - alpha_t)
        q_xs[..., self.mask_token_id] = 1 - alpha_s[..., 0]
        q_xs /= 1 - alpha_t
        return q_xs  # type: ignore

    @staticmethod
    def _sample_generation_timesteps(
        generation_config: DiffusionGenerationConfig,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.FloatTensor:
        """Sample timesteps for diffusion generation process."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if max_length is None:
            max_length = generation_config.max_new_tokens

        if (
            generation_config.first_hitting
            # TODO: first-hitting does not work with posterior
            and generation_config.sampling_strategy == "posterior"
        ):
            timesteps = torch.FloatTensor([1.0])
            for i in range(max_length, 0, -1):
                u = torch.rand(1)
                next_t = timesteps[-1] * u ** (1 / i)
                timesteps = torch.cat((timesteps, next_t), dim=0)
            return timesteps[1:].to(device)  # type: ignore
        return torch.linspace(  # type: ignore
            1.0,
            generation_config.min_t,
            generation_config.num_steps + 1,
            device=device,
        )[:-1]

    def _generate_unconditional(
        self,
        generation_config: DiffusionGenerationConfig,
        alpha_t: torch.FloatTensor,
        alpha_s: torch.FloatTensor,
        denoiser_inputs: Optional[DenoiserInput] = None,
        model_output_cache: Optional[Dict[str, torch.FloatTensor]] = None,
        cache: Optional[Dict[str, Any]] = None,
        running_generation: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        **kwargs: Any,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.FloatTensor], Dict[str, Any]]:
        cache = cache if cache is not None else {}
        if model_output_cache is None:  # execute function evaluation
            with torch.amp.autocast(
                denoiser_inputs.xt.device.type, dtype=torch.float32
            ):
                backbone_output = self._backbone_forward(
                    denoiser_inputs,
                    fix_cache_length=True,  # Do not let kv cache grow on each forward call
                    **cache,
                    **kwargs,
                )
            backbone_output = {k: v for k, v in backbone_output.items()}
            logits = backbone_output.pop("logits")
            cache = cache | backbone_output
            log_x_theta = self._forward(logits, denoiser_inputs, **kwargs)
            if logits_processor is not None:
                for token_idx in range(log_x_theta.shape[1]):
                    # TODO: Looping over token positions like this does not allow for
                    #   some processors, e.g. length penalty which could be applied all
                    #   at once to the entire block, to be applied in parallel.
                    log_x_theta[:, token_idx] = logits_processor(
                        input_ids=running_generation,
                        scores=log_x_theta[:, token_idx],  # type: ignore
                    )
                log_x_theta = torch.log_softmax(log_x_theta, dim=-1)  # re-normalize
            x_theta = log_x_theta.exp()
        else:
            x_theta = model_output_cache["x_theta"]
        model_output_cache = {"x_theta": x_theta}
        prob_check_denom = denoiser_inputs.xt.numel()
        if generation_config.sampling_strategy == "posterior":
            q_xs = self._compute_posterior(
                x_theta, denoiser_inputs.xt, alpha_t, alpha_s
            )

            assert abs((q_xs.sum() / prob_check_denom).item() - 1.0) < 1e-6, (
                "Posterior probabilities not summing to 1."
            )
            assert q_xs.isnan().sum().item() == 0, "NaN found in the posterior."
            xs = self._sample_categorical(q_xs, generation_config.do_sample)
            output = torch.where(
                (denoiser_inputs.xt != self.mask_token_id).bool(),  # type: ignore
                denoiser_inputs.xt,
                xs,
            )
        elif generation_config.sampling_strategy == "predict_and_noise":
            # assert (
            #     abs((x_theta.sum() / prob_check_denom).item() - 1.0) < 1e-6
            # ), "Denoising output probabilities not summing to 1."
            # assert x_theta.isnan().sum().item() == 0, (
            #     "NaN found in the denoising output."
            # )

            # Predict
            xs = self._sample_categorical(x_theta, generation_config.do_sample)
            xs_probs = x_theta.gather(-1, xs[..., None]).squeeze(dim=-1)
            output = xs.clone()

            # Noise
            num_noise_indices = torch.minimum(
                ((1 - alpha_s) * generation_config.block_size).to(torch.int),
                (denoiser_inputs.xt == self.mask_token_id).sum() - 1,  # type: ignore
            )
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
                noise_indices = conf.argsort(dim=-1)[..., :num_noise_indices]
            else:
                # TODO: implement random noise indices selection
                raise NotImplementedError
            output[..., noise_indices] = self.mask_token_id
            output = torch.where(
                xs_probs >= generation_config.confidence_threshold, xs, output
            )
        else:
            raise NotImplementedError(
                f"Sampling strategy {generation_config.sampling_strategy} not"
                " implemented."
            )
        return output, model_output_cache, cache  # type: ignore

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[DiffusionGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        disable_pbar: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor:
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
        accumulated_samples = self.mask_token_id * torch.ones(
            (batch_size, max_blocks * block_size), dtype=torch.int64, device=device
        )
        accumulated_samples = torch.cat([inputs, accumulated_samples], dim=-1)
        if generation_config.use_cache and inputs.numel() > 0:
            if generation_config.align_inputs_to_blocks and inputs.shape[-1] < block_size:
                cache = None
            else:
                cache = self.update_cache(
                    inputs=inputs[:, : block_size * (inputs.shape[-1] // block_size)]
                    if generation_config.align_inputs_to_blocks
                    else inputs,
                    cache={},
                )
        else:
            cache = None

        if generation_config.align_inputs_to_blocks:
            inputs_offset = (
                block_size * (inputs.shape[-1] // block_size)
                if inputs.numel() > 0
                else 0
            )
        else:
            inputs_offset = inputs.shape[-1] if inputs.numel() > 0 else 0

        total_NFEs = 0
        timesteps = self._sample_generation_timesteps(  # Re-use in every block
            generation_config, max_length=block_size, device=device
        )
        dt = (1 - generation_config.min_t) / len(timesteps)
        block_pbar = tqdm(
            range(max_blocks),
            desc="Blocks",
            leave=True,
            disable=disable_pbar,
        )
        for block_id in block_pbar:
            block_NFEs = 0
            xt = accumulated_samples[
                :,
                inputs_offset + (block_id * block_size) : inputs_offset
                + ((block_id + 1) * block_size),
            ]
            if self.mask_token_id not in xt:
                continue
            step_pbar = tqdm(
                timesteps,
                desc="T",
                total=timesteps.shape[0],
                leave=False,
                disable=disable_pbar,
            )
            model_output_cache = None
            context = (
                accumulated_samples[:, : (block_id * block_size) + inputs_offset]
                if not generation_config.use_cache
                else None
            )
            # Used for logit processing
            running_generation = accumulated_samples[
                :,
                inputs_offset : inputs_offset + (block_id * block_size),
            ]
            for t in step_pbar:
                if model_output_cache is None:
                    block_NFEs += 1
                    total_NFEs += 1
                # t is 0-dim tensor, reshape to (1, 1, 1) for broadcasting
                alpha_t, _ = self.noise_schedule(t)
                alpha_s, _ = self.noise_schedule(t - dt)
                alpha_t = alpha_t[None, None, None]
                alpha_s = alpha_s[None, None, None]
                denoiser_inputs, cache = self._prepare_inputs_inference(
                    input_ids=xt,
                    context=context,
                    cache=cache if generation_config.use_cache else None,
                )
                xs, model_output_cache, cache = self._generate_unconditional(
                    generation_config=generation_config,
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                    denoiser_inputs=denoiser_inputs,
                    model_output_cache=model_output_cache,
                    cache=cache,
                    running_generation=running_generation,  # type: ignore
                    logits_processor=logits_processor,
                    tokenizer=tokenizer,
                    **kwargs,
                )
                block_pbar.set_postfix(
                    NFEs=total_NFEs,
                    block_NFEs=block_NFEs,
                )

                if not torch.allclose(xs, denoiser_inputs.xt):
                    model_output_cache = None
                if not generation_config.use_cache:
                    xt[..., -block_size:] = xs[..., -block_size:]
                else:
                    xt = xs
                if (xt == self.mask_token_id).sum().item() == 0:
                    break
            accumulated_samples[
                :,
                inputs_offset + (block_id * block_size) : inputs_offset
                + ((block_id + 1) * block_size),
            ] = xt
            if tokenizer is not None:  # Useful for debugging
                print(tokenizer.batch_decode(accumulated_samples))
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=accumulated_samples[  # type: ignore
                        :,
                        inputs_offset : inputs_offset + ((block_id + 1) * block_size),
                    ],
                    scores=None,  # type: ignore
                )
                if torch.any(is_done):
                    accumulated_samples = accumulated_samples[
                        :,
                        : inputs_offset + ((block_id + 1) * block_size),
                    ]
                    break
            if generation_config.use_cache:
                cache = self.update_cache(
                    inputs=xt,
                    cache=cache,
                )
        return accumulated_samples  # type: ignore


class BD3LMConfig(MDLMConfig):
    """Configuration class for BD3LM models."""

    model_type = "bd3lm"
    auto_map = {
        "AutoConfig": "diffusion.BD3LMConfig",
        "AutoModel": "diffusion.BD3LM",
        "AutoModelForMaskedLM": "diffusion.BD3LM",
    }

    def __init__(
        self,
        block_size: Optional[int] = None,
        eval_block_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.eval_block_size = (
            eval_block_size if eval_block_size is not None else block_size
        )


class BD3LM(MDLM):
    """Denoiser class for BD3LM models."""

    config_class = BD3LMConfig

    def __init__(self, config: BD3LMConfig, tokenizer: Optional[PreTrainedTokenizer] = None, **kwargs):
        super().__init__(config, tokenizer, **kwargs)

    # noinspection PyUnusedLocal
    @staticmethod
    def _block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        block_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        del b, h

        # Indicate whether token belongs to xt or x0:
        xt_flag_q = (q_idx >= seq_length).bool()
        xt_flag_kv = (kv_idx >= seq_length).bool()

        # Compute block indices
        block_q = torch.where(
            xt_flag_q, (q_idx - seq_length) // block_size, q_idx // block_size
        )
        block_kv = torch.where(
            xt_flag_kv, (kv_idx - seq_length) // block_size, kv_idx // block_size
        )
        # **1. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q > block_kv) & ~xt_flag_kv & xt_flag_q

        # **2. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q == block_kv) & (xt_flag_q == xt_flag_kv)

        # **3. Block-Causal Mask (M_BC) **
        block_causal = (block_q >= block_kv) & ~xt_flag_kv & ~xt_flag_q

        # **3. Combine Masks **
        return block_diagonal | offset_block_causal | block_causal

    def _create_static_mask(self) -> None:
        if self.config.attn_backend == "sdpa":
            static_mask = self._block_mask(
                b=None,
                h=None,
                q_idx=torch.arange(self.config.length * 2)[:, None],
                kv_idx=torch.arange(self.config.length * 2)[None, :],
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
                seq_length=self.config.length,
            )
            self.register_buffer(
                "static_attention_mask",
                static_mask,
            )
            self.skip_params_for_push.append("static_attention_mask")
        elif self.config.attn_backend == "flex_attention":
            mask = partial(
                self._block_mask,
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
                seq_length=self.config.length,
            )
            self.static_attention_mask = create_block_mask(
                mask,
                B=None,
                H=None,
                Q_LEN=self.config.length * 2,
                KV_LEN=self.config.length * 2,
            )

    def _ensure_no_unmasked_blocks(
        self,
        input_ids: torch.LongTensor,
        xt: torch.LongTensor,
        context_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        n_blocks = xt.shape[1] // self.config.block_size
        # If context overlaps w/block, ignore it
        blocks_without_masks = ((xt == self.mask_token_id) + context_mask).reshape(
            -1, n_blocks, self.config.block_size
        ).sum(dim=-1) == 0
        if blocks_without_masks.sum() > 0:
            num_remasks_per_block = torch.randint(
                0,
                self.config.block_size,
                blocks_without_masks.shape,
                device=xt.device,
            )
            rand = torch.rand(xt.shape[0], xt.shape[1], device=xt.device)
            perm_indices = torch.argsort(
                rand.view(xt.shape[0], n_blocks, self.config.block_size),
                stable=True,
                dim=-1,
            )
            remask_indices = perm_indices <= num_remasks_per_block[..., None]
            xt = torch.where(
                remask_indices.view(xt.shape[0], xt.shape[1])
                * blocks_without_masks.repeat_interleave(self.config.block_size, dim=1),
                self.mask_token_id,
                xt,
            )
            if self.config.keep_clean_bos:
                xt[..., 0] = input_ids[..., 0]
        return xt

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
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
                input_ids.shape[1] // self.config.block_size
                if self.training
                else self.config.eval_block_size,
                device=input_ids.device,
            ).repeat_interleave(
                self.config.block_size
                if self.training
                else self.config.eval_block_size,
                dim=-1,
            )
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]
        noise_mask = context_mask | ~(attention_mask.bool())
        xt = self._sample_q_xt(x0=input_ids, alpha_t=alpha_t, mask=noise_mask)
        # Ensure each block has at least 1 masked token
        if self.training or self.config.block_size == 1:
            xt = self._ensure_no_unmasked_blocks(
                input_ids,
                xt,
                noise_mask,
            )
        if self.config.attn_backend == "sdpa":
            decoder_attention_mask = (
                self.static_attention_mask[None, ...]
                & attention_mask.repeat(1, 2)[:, None, :]
                & attention_mask.repeat(1, 2)[..., None]
            )[:, None, ...]  # Make attention mask 4D
            if getattr(self.config, "train_complement", False):
                block_causal = torch.arange(self.config.length)[..., None] // self.config.block_size >= torch.arange(self.config.length)[None, :] // self.config.block_size
                block_causal = block_causal[None, None, ...].to(xt.device)
                offset_block_causal = torch.arange(self.config.length)[..., None] // self.config.block_size > torch.arange(self.config.length)[None, :] // self.config.block_size
                offset_block_causal = offset_block_causal[None, None, ...].to(xt.device)
                masked_tokens = xt == self.mask_token_id
                attn_allowed = masked_tokens[..., None] == masked_tokens[..., None, :]
                attn_allowed = attn_allowed[:, None, ...]
                decoder_attention_mask[:, :, :self.config.length, :self.config.length] = (attn_allowed | offset_block_causal) & block_causal
                decoder_attention_mask[:, :, self.config.length:, :self.config.length] = (~attn_allowed | offset_block_causal) & block_causal
                decoder_attention_mask[:, :, self.config.length:, self.config.length:] = (attn_allowed & block_causal) & ~offset_block_causal
                if self.training:
                    xt = torch.where(noise_mask == 1, xt, self.mask_token_id)
            if getattr(self.config, "rm_attn_to_masked_tokens", False):
                masked_tokens = xt == self.mask_token_id
                padded_masked_tokens = F.pad(masked_tokens, (masked_tokens.shape[1], 0), value=False)
                # Zero out columns
                decoder_attention_mask = decoder_attention_mask & (~padded_masked_tokens[:, None, None, :]).to(decoder_attention_mask.dtype)
                # Keep self-attention
                decoder_attention_mask[:, :, self.config.length:, self.config.length:] |= torch.eye(self.config.length, device=xt.device)[None, None, :, :].to(xt.device).to(decoder_attention_mask.dtype)
            if getattr(self.config, "attend_to_dummy_tokens", False):
                num_dummy_tokens = getattr(self.config, "num_dummy_tokens", xt.shape[-1])
                dummy_token = self.tokenizer.unk_token_id
                xt = torch.cat((xt, torch.full((xt.shape[0], num_dummy_tokens), dummy_token, device=xt.device)), dim=-1)

                # pad attention mask on bottom and right
                decoder_attention_mask = F.pad(decoder_attention_mask, (0, num_dummy_tokens, 0, num_dummy_tokens), value=False)
                                
                # zero out rows and columns according to attention mask
                padding_attn_mask = torch.ones((num_dummy_tokens, num_dummy_tokens), device=xt.device).int()
                padding_attn_mask = padding_attn_mask & attention_mask[:, None, :num_dummy_tokens] & attention_mask[:, :num_dummy_tokens, None]
                padding_attn_mask = padding_attn_mask[:, None, ...]
                
                # allow masked tokens to attend to dummy tokens
                decoder_attention_mask[:, :, self.config.length:self.config.length*2, -num_dummy_tokens:] = attention_mask[:, None, None, :num_dummy_tokens]
                # allow clean tokens to attend to dummy tokens
                decoder_attention_mask[:, :, :self.config.length, -num_dummy_tokens:] = attention_mask[:, None, None, :num_dummy_tokens]

                # # allow dummy tokens to attend to each other
                # decoder_attention_mask[:, :, -num_dummy_tokens:, -num_dummy_tokens:] = padding_attn_mask
                decoder_attention_mask[:, :, -num_dummy_tokens:, -num_dummy_tokens:] = torch.eye(num_dummy_tokens, device=xt.device).to(decoder_attention_mask.dtype)[None, None, ...]
                
            decoder_attention_mask = self._preprocess_attention_mask(
                decoder_attention_mask, dtype=torch.float
            )
        elif self.config.attn_backend == "flex_attention":
            if context_mask.any():
                raise NotImplementedError(
                    "flex_attention with context_mask not implemented yet."
                )
            elif attention_mask is not None and (attention_mask != 1).any():
                padding_mask = create_attn_mask(
                    attention_mask.bool().repeat(2, 2).bool()
                )
                dec_masks = [
                    partial(
                        self._block_mask,
                        block_size=self.config.block_size
                        if self.training
                        else self.config.eval_block_size,
                        seq_length=self.config.length,
                    ),
                    padding_mask,
                ]
                decoder_attention_mask = create_block_mask(
                    and_masks(*dec_masks),
                    B=input_ids.shape[0],
                    H=None,
                    Q_LEN=input_ids.shape[1] * 2,
                    KV_LEN=input_ids.shape[1] * 2,
                )
            else:
                decoder_attention_mask = self.static_attention_mask
        else:
            raise ValueError("Unknown backbone backend")
        backbone_input_ids = torch.cat((input_ids, xt), dim=-1)
        position_ids = (
            torch.arange(input_ids.shape[1]).repeat(2).to(input_ids.device)[None, :]
        )
        if getattr(self.config, "attend_to_dummy_tokens", False):
            position_ids = torch.cat((position_ids, torch.arange(num_dummy_tokens).to(input_ids.device)[None, :]), dim=-1)
        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask)
        return DenoiserInput(
            xt=backbone_input_ids,  # type: ignore
            x0=input_ids,
            attention_mask=decoder_attention_mask,  # type: ignore
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
            backbone_kwargs={
                "cache_position": position_ids[0],
                "position_ids": position_ids,
            },
        )

    def _prepare_inputs_inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context: Optional[torch.LongTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        cache: Optional[Dict[str, Any]] = None,
        return_updated_cache: bool = False,
        **backbone_kwargs: Dict[str, Any],
    ) -> Tuple[DenoiserInput, Union[Dict[str, Any], None]]:
        device = input_ids.device if input_ids is not None else context.device
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        if context is not None:
            if input_ids is not None:
                input_ids = torch.cat([context, input_ids], dim=-1)
            else:
                input_ids = context
        cache_length = self._get_past_key_values_seq_length(past_key_values)
        full_seq_length = cache_length + input_ids.shape[-1]
        decoder_attention_mask = self.static_attention_mask[
            None,
            None,
            cache_length:full_seq_length,
            :full_seq_length,
        ]  # Make attention mask 4D
        decoder_attention_mask = self._preprocess_attention_mask(
            decoder_attention_mask, dtype=torch.float
        )
        position_ids = torch.arange(cache_length, full_seq_length).to(device)[None, :]
        return DenoiserInput(
            xt=input_ids,
            attention_mask=decoder_attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            backbone_kwargs={
                "position_ids": position_ids,
            }
            | backbone_kwargs,
        ), cache

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        input_length = denoiser_inputs.x0.shape[1]
        model_output = model_output[:, input_length:input_length*2, ...]
        return super()._compute_loss(
            model_output=model_output,  # type: ignore
            denoiser_inputs=denoiser_inputs,
            **kwargs,
        )


class E2D2Config(BD3LMConfig):
    """Configuration class for E2D2 models."""

    model_type = "e2d2"
    auto_map = {
        "AutoConfig": "diffusion.E2D2Config",
        "AutoModel": "diffusion.E2D2",
        "AutoModelForMaskedLM": "diffusion.E2D2",
    }

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class E2D2(BD3LM):
    """Denoiser class for E2D2 models."""

    config_class = E2D2Config

    def __init__(self, config: E2D2Config, tokenizer: Optional[PreTrainedTokenizer] = None, **kwargs):
        super().__init__(config, tokenizer, **kwargs)

    # noinspection PyUnusedLocal
    @staticmethod
    def _encoder_block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
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

        # Compute block indices
        block_q = q_idx // block_size
        block_kv = kv_idx // block_size

        # ** Block-Causal Mask **
        return block_q >= block_kv

    # noinspection PyUnusedLocal
    @staticmethod
    def _decoder_block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        block_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        # Indicate whether token belongs to xt or x0:
        xt_flag_kv = (kv_idx >= seq_length).bool()

        # Compute block indices
        block_q = q_idx // block_size
        block_kv = torch.where(
            xt_flag_kv, (kv_idx - seq_length) // block_size, kv_idx // block_size
        )
        # **1. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q > block_kv) & ~xt_flag_kv

        # **2. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q == block_kv) & xt_flag_kv

        # **3. Combine Masks **
        return block_diagonal | offset_block_causal

    def _create_static_mask(self) -> None:
        if self.config.attn_backend == "flex_attention":
            enc_mask = partial(
                self._encoder_block_mask,
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
            )
            encoder_attention_mask = create_block_mask(
                enc_mask,
                B=None,
                H=None,
                Q_LEN=self.config.length,
                KV_LEN=self.config.length,
            )
            dec_mask = partial(
                self._decoder_block_mask,
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
                seq_length=self.config.length,
            )
            decoder_attention_mask = create_block_mask(
                dec_mask,
                B=None,
                H=None,
                Q_LEN=self.config.length,
                KV_LEN=self.config.length * 2,
            )
            self.encoder_static_attention_mask = encoder_attention_mask
            self.static_attention_mask = decoder_attention_mask
        else:
            encoder_static_mask = self._encoder_block_mask(
                b=None,  # type: ignore
                h=None,  # type: ignore
                q_idx=torch.arange(self.config.length)[:, None],
                kv_idx=torch.arange(self.config.length)[None, :],
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
            )
            decoder_static_mask = self._decoder_block_mask(
                b=None,
                h=None,
                q_idx=torch.arange(self.config.length)[:, None],
                kv_idx=torch.arange(self.config.length * 2)[None, :],
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
                seq_length=self.config.length,
            )
            self.register_buffer(
                "encoder_static_attention_mask",
                encoder_static_mask,
            )
            self.register_buffer(
                "static_attention_mask",
                decoder_static_mask,
            )
            self.skip_params_for_push.append("encoder_static_attention_mask")
            self.skip_params_for_push.append("static_attention_mask")

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
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
                input_ids.shape[1] // self.config.block_size
                if self.training
                else self.config.eval_block_size,
                device=input_ids.device,
            ).repeat_interleave(
                self.config.block_size
                if self.training
                else self.config.eval_block_size,
                dim=-1,
            )
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]
        noise_mask = context_mask | ~(attention_mask.bool())
        xt = self._sample_q_xt(x0=input_ids, alpha_t=alpha_t, mask=noise_mask)
        # Ensure each block has at least 1 masked token
        if self.training or self.config.block_size == 1:
            xt = self._ensure_no_unmasked_blocks(
                input_ids,
                xt,
                noise_mask,
            )
        if self.config.attn_backend == "sdpa":
            decoder_attention_mask = (
                self.static_attention_mask[None, ...]
                & attention_mask.repeat(1, 2)[:, None, :]
                & attention_mask[..., None]
            )[:, None, ...]  # Make attention mask 4D
            encoder_attention_mask = (
                (
                    self.encoder_static_attention_mask[None, ...]
                    | context_mask[:, None, :]
                )
                & attention_mask[:, None, :]
                & attention_mask[..., None]
            )[:, None, ...]  # Make attention mask 4D
            encoder_attention_mask = self._preprocess_attention_mask(
                encoder_attention_mask, dtype=torch.float
            )
            decoder_attention_mask = self._preprocess_attention_mask(
                decoder_attention_mask, dtype=torch.float
            )
        elif self.config.attn_backend == "flex_attention":
            # TODO enable bidirectional attention on context for seq2seq tasks
            if context_mask.any():
                raise NotImplementedError(
                    "flex_attention with context_mask not implemented yet."
                )
            elif attention_mask is not None and (attention_mask != 1).any():
                padding_mask = create_attn_mask(attention_mask.bool())
                dec_padding_mask = create_attn_mask(attention_mask.repeat(1, 2).bool())
                enc_masks = [
                    partial(
                        self._encoder_block_mask,
                        block_size=self.config.block_size
                        if self.training
                        else self.config.eval_block_size,
                    ),
                    padding_mask,
                ]
                encoder_attention_mask = create_block_mask(
                    and_masks(*enc_masks),
                    B=input_ids.shape[0],
                    H=None,
                    Q_LEN=input_ids.shape[1],
                    KV_LEN=input_ids.shape[1],
                )
                dec_masks = [
                    partial(
                        self._decoder_block_mask,
                        block_size=self.config.block_size
                        if self.training
                        else self.config.eval_block_size,
                        seq_length=input_ids.shape[1],
                    ),
                    dec_padding_mask,
                ]
                decoder_attention_mask = create_block_mask(
                    and_masks(*dec_masks),
                    B=input_ids.shape[0],
                    H=None,
                    Q_LEN=input_ids.shape[1],
                    KV_LEN=input_ids.shape[1] * 2,
                )
            else:
                encoder_attention_mask = self.encoder_static_attention_mask
                decoder_attention_mask = self.static_attention_mask
        else:
            raise ValueError("Unknown backbone backend")
        position_ids = torch.arange(input_ids.shape[1]).to(input_ids.device)[None, :]
        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask)
        return DenoiserInput(
            xt=xt,
            x0=input_ids,
            attention_mask=decoder_attention_mask,
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
            backbone_kwargs={
                "encoder_input_ids": input_ids,
                "encoder_attention_mask": encoder_attention_mask,
                "encoder_position_ids": position_ids,
                "encoder_cache_position": position_ids[0],
            },
        )

    def _prepare_inputs_inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context: Optional[torch.LongTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        cache: Optional[Dict[str, Any]] = None,
        return_updated_cache: bool = False,
        **backbone_kwargs: Dict[str, Any],
    ) -> Tuple[DenoiserInput, Union[Dict[str, Any], None]]:
        device = input_ids.device if input_ids is not None else context.device
        batch_size = input_ids.shape[0] if input_ids is not None else context.shape[0]
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        if return_updated_cache:  # Indicates this is a cache update step
            context = input_ids
            input_ids = None
        position_ids, encoder_position_ids = None, None
        if cache is not None:
            past_key_values = cache.pop("past_key_values", DynamicCache())
            encoder_past_key_values = cache.pop(
                "encoder_past_key_values", DynamicCache()
            )
            encoder_last_hidden_state = cache.pop("encoder_last_hidden_state", None)
            if input_ids is not None:  # Skip enc: nothing new to cache
                cache_length = self._get_past_key_values_seq_length(past_key_values)
                if encoder_last_hidden_state is not None:
                    full_seq_length = (
                        cache_length
                        + encoder_last_hidden_state.shape[1]  # type: ignore
                        + input_ids.shape[-1]
                    )
                else:
                    full_seq_length = cache_length + input_ids.shape[-1]
                encoder_attention_mask = None
                position_ids = torch.arange(
                    cache_length, full_seq_length, device=device
                )[None, :]
            else:  # Caching new tokens in the enc
                encoder_cache_length = self._get_past_key_values_seq_length(
                    encoder_past_key_values
                    if len(encoder_past_key_values) > 0
                    else past_key_values
                )
                encoder_full_seq_length = encoder_cache_length + context.shape[-1]
                encoder_attention_mask = torch.ones(
                    (
                        1,
                        1,
                        encoder_full_seq_length - encoder_cache_length,
                        encoder_full_seq_length,
                    ),
                    device=context.device,
                )
                encoder_position_ids = torch.arange(
                    encoder_cache_length, encoder_full_seq_length
                ).to(device)[None, :]
                encoder_attention_mask = self._preprocess_attention_mask(
                    encoder_attention_mask, dtype=torch.float
                )
                full_seq_length = -1  # Not used
        else:  # Not using kv-cache
            past_key_values = None
            encoder_past_key_values, encoder_last_hidden_state = None, None
            if context is not None:
                context_len = context.shape[1]
                encoder_attention_mask = torch.ones(
                    (1, 1, context_len, context_len), device=context.device
                )
                encoder_attention_mask = self._preprocess_attention_mask(
                    encoder_attention_mask, dtype=torch.float
                )
                encoder_position_ids = torch.arange(context_len).to(device)[None, :]
            else:
                context_len = 0
                encoder_attention_mask = None
            if input_ids is not None:
                full_seq_length = context_len + input_ids.shape[1]
            else:
                full_seq_length = context_len
            position_ids = torch.arange(context_len, full_seq_length).to(device)[
                None, :
            ]
        if input_ids is not None:
            decoder_attention_mask = torch.ones(
                (batch_size, 1, input_ids.shape[1], full_seq_length),
                device=device,
            )  # Make attention mask 4D
            decoder_attention_mask = self._preprocess_attention_mask(
                decoder_attention_mask, dtype=torch.float
            )
        else:
            decoder_attention_mask = None
        return DenoiserInput(
            xt=input_ids,
            attention_mask=decoder_attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            backbone_kwargs={
                "position_ids": position_ids,
                "encoder_input_ids": context,
                "encoder_position_ids": encoder_position_ids,
                "encoder_attention_mask": encoder_attention_mask,
                "encoder_past_key_values": encoder_past_key_values,
                "encoder_last_hidden_state": encoder_last_hidden_state,
            }
            | backbone_kwargs,
        ), cache  # TODO: potentially returning cache None, violates return type

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        # Use MDLM `_compute_loss`, since BD3LM method splits model_output
        return super(BD3LM, self)._compute_loss(
            model_output=model_output,
            denoiser_inputs=denoiser_inputs,
            **kwargs,
        )


class AnyOrderBD3LM(BD3LM):
    """Denoiser class for AnyOrderBD3LM models."""

    def __init__(self, config: BD3LMConfig, tokenizer: Optional[PreTrainedTokenizer] = None, **kwargs):
        super().__init__(config, tokenizer, **kwargs)
        self.block_size = config.block_size
        if config.attn_backend == "flex_attention":
            self.static_attention_mask = None
            self.encoder_static_attention_mask = None
        self._create_static_mask()

    def _create_static_mask(self) -> None:
        # self.mask_to_mask_interaction = False
        # self.clean_block_caching = False
        # self.clean_attends_to_mask = False
        # self.clean_attends_to_clean = True
        # self.attn_sink_only = False
        # block_diagonal_mask =((torch.arange(self.config.length)[..., None] // self.config.block_size) == (torch.arange(self.config.length)[None, ...] // self.config.block_size)).bool()
        # block_causal_mask = ((torch.arange(self.config.length)[..., None] // self.config.block_size) >= (torch.arange(self.config.length)[None, ...] // self.config.block_size)).bool()
        # offset_block_causal_mask = ((torch.arange(self.config.length)[..., None] // self.config.block_size) > (torch.arange(self.config.length)[None, ...] // self.config.block_size)).bool()
        # upper_tri = torch.triu(torch.ones((self.config.length, self.config.length), dtype=torch.bool), diagonal=1)
        encoder_static_mask = torch.tril(
            torch.ones(
                (self.config.length, self.config.length),
                dtype=torch.bool,
            ),
            diagonal=-1,
        )
        # if self.clean_block_caching:
        #     encoder_static_mask *= block_diagonal_mask
        decoder_static_mask = torch.cat(
            (encoder_static_mask, torch.zeros_like(encoder_static_mask)), dim=-1
        )
        encoder_static_mask = torch.cat(
            (encoder_static_mask, torch.zeros_like(encoder_static_mask)), dim=-1
        )
        self.register_buffer("static_attention_mask", 
            torch.cat(
                (encoder_static_mask, decoder_static_mask), dim=0
            )
        )

        # # -- Attention to clean tokens --
        # if self.clean_block_caching:
        #     self.static_attention_mask = F.pad(self.static_attention_mask, (0, self.config.length, 0, self.config.length), value=False)

        #     # "caching"
        #     self.static_attention_mask[-self.config.length:, -self.config.length:] = block_causal_mask

        #     # clean tokens can attend to offset blocks
        #     self.static_attention_mask[:self.config.length, -self.config.length:] = offset_block_causal_mask

        #     # masked tokens can attend to clean blocks
        #     self.static_attention_mask[self.config.length:-self.config.length, -self.config.length:] = offset_block_causal_mask

        # if not self.clean_attends_to_clean:
        #     self.static_attention_mask[:self.config.length, :self.config.length] *= (~block_diagonal_mask)
        #     self.static_attention_mask[:self.config.length, :self.config.length] = torch.eye(self.config.length, device=self.static_attention_mask.device).bool()

        # # -- Attention to masked tokens --
        # if self.mask_to_mask_interaction:
        #     if self.attn_sink_only:
        #         mask_len = self.config.length
        #     else:
        #         mask_len = self.config.length
        #     self.static_attention_mask = F.pad(self.static_attention_mask, (0, mask_len, 0, mask_len), value=False)

        #     self.static_attention_mask[-mask_len:, -mask_len:] = block_diagonal_mask#[:mask_len, :mask_len]

        #     # can attend to previous clean blocks
        #     if self.clean_block_caching:
        #         self.static_attention_mask[-mask_len:, self.config.length*2:-mask_len] = offset_block_causal_mask
        #     else:
        #         self.static_attention_mask[-mask_len:, :self.config.length] = offset_block_causal_mask#[:mask_len]

        #     # mask tokens can attend to offset blocks
        #     if self.attn_sink_only:
        #         self.static_attention_mask[self.config.length:self.config.length*2, -mask_len:] = True
        #     else:
        #         self.static_attention_mask[self.config.length:self.config.length*2, -mask_len:] = (upper_tri * block_diagonal_mask)
        
        #     if self.clean_attends_to_mask:
        #         self.static_attention_mask[:mask_len, -mask_len:] = (upper_tri * block_diagonal_mask)#[:mask_len, :mask_len]

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        input_length = denoiser_inputs.x0.shape[1]
        # if self.mask_to_mask_interaction and not self.clean_block_caching and self.training:
        #     num_repetitions = 2
        #     model_output = model_output[:, input_length:, ...]
        # else:
        # num_repetitions = 1
        model_output = model_output[:, input_length:input_length*2, ...]
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=denoiser_inputs.x0[:, :, None] #.repeat(1, num_repetitions, 1)
        ).squeeze(-1)
        loss = -log_p_theta
        if not self.training:
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask * (
                denoiser_inputs.x0 != self.pad_token_id
            )
        nlls = loss * denoiser_inputs.tokens_mask
        
        # Apply random dropout to nlls during training
        # dropout_rate = kwargs.get("nll_dropout_rate", (self.config.block_size - 1) / self.config.block_size)
        # block_wise_dropout = kwargs.get("nll_block_wise_dropout", True)
        # dropout_rate = 0.0
        # block_wise_dropout = False
        # if self.training and dropout_rate > 0.0:
        #     if block_wise_dropout:
        #         # Block-wise dropout: randomly drop dropout% tokens in each block
        #         batch_size, seq_len = denoiser_inputs.tokens_mask.shape
        #         n_blocks = seq_len // self.block_size
        #         if n_blocks > 0:
        #             # Reshape into blocks: [batch_size, n_blocks, block_size]
        #             tokens_mask_blocks = denoiser_inputs.tokens_mask[:, :n_blocks * self.block_size].view(
        #                 batch_size, n_blocks, self.block_size
        #             )
                    
        #             # Initialize dropout mask with all ones
        #             dropout_mask_blocks = torch.ones_like(tokens_mask_blocks)
                    
        #             # Generate random values for each position in each block
        #             rand_vals = torch.rand_like(tokens_mask_blocks.float())
        #             # Only consider valid tokens for dropout
        #             rand_vals = torch.where(tokens_mask_blocks.bool(), rand_vals, float('inf'))
                    
        #             # Count valid tokens per block and calculate how many to drop
        #             valid_tokens_per_block = tokens_mask_blocks.sum(dim=-1, keepdim=True)  # [batch_size, n_blocks, 1]
        #             num_tokens_to_drop = (valid_tokens_per_block * dropout_rate).long()  # [batch_size, n_blocks, 1]
                    
        #             # For each block, select the top-k (by random value) valid tokens to drop
        #             # We use k smallest random values to determine which tokens to drop
        #             for block_idx in range(n_blocks):
        #                 block_rand = rand_vals[:, block_idx, :]  # [batch_size, block_size]
        #                 block_mask = tokens_mask_blocks[:, block_idx, :]  # [batch_size, block_size]
        #                 n_to_drop = num_tokens_to_drop[:, block_idx, 0]  # [batch_size]
                        
        #                 # For each batch, select tokens to drop
        #                 for b in range(batch_size):
        #                     if n_to_drop[b].item() > 0 and block_mask[b].any():
        #                         # Get valid indices in this block
        #                         valid_indices = torch.where(block_mask[b])[0]
        #                         if len(valid_indices) > 0:
        #                             # Get random values for valid tokens
        #                             valid_rand = block_rand[b, valid_indices]
        #                             # Select tokens with smallest random values to drop
        #                             n_drop = min(n_to_drop[b].item(), len(valid_indices))
        #                             _, drop_relative_indices = torch.topk(valid_rand, n_drop, largest=False)
        #                             drop_absolute_indices = valid_indices[drop_relative_indices]
        #                             dropout_mask_blocks[b, block_idx, drop_absolute_indices] = 0.0
                    
        #             # Only apply dropout to valid tokens
        #             dropout_mask_blocks = dropout_mask_blocks * tokens_mask_blocks
                    
        #             # Reshape back to original shape
        #             dropout_mask = dropout_mask_blocks.view(batch_size, n_blocks * self.block_size)
        #             # Handle remaining tokens (if seq_len is not divisible by block_size)
        #             if seq_len > n_blocks * self.block_size:
        #                 remaining_tokens = denoiser_inputs.tokens_mask[:, n_blocks * self.block_size:]
        #                 remaining_mask = torch.bernoulli(
        #                     torch.ones_like(remaining_tokens) * (1.0 - dropout_rate)
        #                 ).to(remaining_tokens.dtype) * remaining_tokens
        #                 dropout_mask = torch.cat([dropout_mask, remaining_mask], dim=-1)
        #             else:
        #                 # Pad to original length if needed (shouldn't happen, but just in case)
        #                 if dropout_mask.shape[1] < seq_len:
        #                     padding = denoiser_inputs.tokens_mask[:, dropout_mask.shape[1]:]
        #                     dropout_mask = torch.cat([dropout_mask, padding], dim=-1)
        #         else:
        #             # Fallback to regular dropout if no blocks
        #             dropout_mask = torch.bernoulli(
        #                 torch.ones_like(denoiser_inputs.tokens_mask) * (1.0 - dropout_rate)
        #             ).to(denoiser_inputs.tokens_mask.dtype)
        #             dropout_mask = dropout_mask * denoiser_inputs.tokens_mask
        #     else:
        #         # Regular dropout: drop tokens independently across entire sequence
        #         dropout_mask = torch.bernoulli(
        #             torch.ones_like(denoiser_inputs.tokens_mask) * (1.0 - dropout_rate)
        #         ).to(denoiser_inputs.tokens_mask.dtype)
        #         # Only apply dropout to valid tokens (where tokens_mask is True)
        #         dropout_mask = dropout_mask * denoiser_inputs.tokens_mask
            
        #     # Apply dropout to nlls
        #     nlls = nlls * dropout_mask
        #     # Update count to only include non-dropped tokens
        #     count = dropout_mask.sum()
        # else:
        # Compute per-batch counts and losses to avoid division by zero
        count = denoiser_inputs.tokens_mask.sum(dim=-1)  # Per-batch counts
        batch_nll = nlls.sum(dim=-1)  # Per-batch losses
        
        # Avoid division by zero: if count is 0, set token_nll to 0
        token_nll = torch.where(
            count > 0,
            batch_nll / count,
            torch.zeros_like(batch_nll)
        ).mean()
        
        # Extract permutation order from backbone_kwargs
        permutation_order = denoiser_inputs.backbone_kwargs.get("permutation_order")
        other_loss_terms = {}
        if permutation_order is not None:
            other_loss_terms["permutation_order"] = permutation_order
            other_loss_terms["attention_mask"] = denoiser_inputs.attention_mask
            other_loss_terms["log_p_theta"] = - log_p_theta * denoiser_inputs.tokens_mask
        return LossAndNllOutput(
            loss=token_nll,
            nlls=nlls,
            other_loss_terms=other_loss_terms,
        )  # type: ignore

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
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

        # permute input ids (where context_mask > 0 and attention_mask > 0)
        noise_mask = context_mask | ~(attention_mask.bool())
        permute_flag = noise_mask != 1
        perm_indices = None
        xt = input_ids.clone()

        batch_size, context_len = input_ids.shape
        if permute_flag.any():
            perm_indices = self.noise_schedule.sample_permutation_order(t, permute_flag, block_size=self.config.block_size)
        # if self.attn_sink_only:
        #     num_repetitions = 2
        #     attention_mask_padded = F.pad(attention_mask.repeat(1, 2), (0, 1, 0, 1), value=True)
        #     decoder_attention_mask = (
        #         self.static_attention_mask[None, ...]
        #         & attention_mask_padded[:, None, :]
        #         & attention_mask_padded[..., None]
        #     )
        # else:
        num_repetitions = self.static_attention_mask.shape[1] // input_ids.shape[1]

        decoder_attention_mask = (
            self.static_attention_mask[None, ...]
            & attention_mask.repeat(1, num_repetitions)[:, None, :]
            & attention_mask.repeat(1, num_repetitions)[..., None]
        )
        if permute_flag.any():
            seq_len = input_ids.shape[1]
            perm_indices_cols = perm_indices.repeat(1, num_repetitions)
            # perm_indices_cols = torch.cat((perm_indices_cols, torch.arange(seq_len, seq_len*2).repeat(batch_size, 1).to(input_ids.device)), dim=-1)

            perm_indices_cols[:, seq_len:seq_len*2] += seq_len
            # if self.attn_sink_only:
            #     perm_indices_cols = F.pad(perm_indices_cols, (0, 1), value=seq_len*2)
            if num_repetitions > 2:
                perm_indices_cols[:, seq_len*2:] += seq_len * 2
            if num_repetitions > 3:
                perm_indices_cols[:, seq_len*3:] += seq_len * 3
            # permute rows
            decoder_attention_mask_perm = decoder_attention_mask[
                torch.arange(batch_size).unsqueeze(1), perm_indices_cols.argsort(dim=-1)]
            # permute columns
            decoder_attention_mask_perm = torch.gather(
                decoder_attention_mask_perm,
                dim=-1,
                index=perm_indices_cols.argsort(dim=-1)[:, None, :].expand(
                    batch_size, decoder_attention_mask_perm.shape[-1], decoder_attention_mask.shape[-1]
                )
            )

            # self-attention
            decoder_attention_mask_perm[:, torch.arange(seq_len), torch.arange(seq_len)] = 1
            decoder_attention_mask_perm[:, torch.arange(seq_len, seq_len*2), torch.arange(seq_len, seq_len*2)] = 1
            # if num_repetitions > 2:
            #     decoder_attention_mask_perm[:, torch.arange(seq_len*2, seq_len*3), torch.arange(seq_len*2, seq_len*3)] = 1
            # if num_repetitions > 3:
            #     decoder_attention_mask_perm[:, torch.arange(seq_len*3, seq_len*4), torch.arange(seq_len*3, seq_len*4)] = 1
            decoder_attention_mask = decoder_attention_mask_perm

        decoder_attention_mask = self._preprocess_attention_mask(
            decoder_attention_mask[:, None], dtype=torch.float
        )
        xt = torch.where(
            (attention_mask == 1) & (context_mask == 0), self.mask_token_id, xt
        )
        xt = torch.cat((input_ids, xt), dim=-1)
        # if self.clean_block_caching:
        #     xt = torch.cat((xt, input_ids), dim=-1)
        # if self.mask_to_mask_interaction:
        #     if self.attn_sink_only:
        #         xt = torch.cat((xt, torch.full_like(input_ids[:, :1], self.mask_token_id)), dim=-1)
        #     else:
        #         xt = torch.cat((xt, torch.full_like(input_ids, self.mask_token_id)), dim=-1)

        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(batch_size, num_repetitions).to(input_ids.device)
        # if self.attn_sink_only:
        #     position_ids = F.pad(position_ids, (0, 1), value=input_ids.shape[1]*2)

        tokens_mask = attention_mask * (1 - context_mask)
        # if self.mask_to_mask_interaction and not self.clean_block_caching and self.training:
        #     tokens_mask = tokens_mask.repeat(1, 2)
        return DenoiserInput(
            xt=xt,  # type: ignore
            x0=input_ids,
            attention_mask=decoder_attention_mask,  # type: ignore
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
            backbone_kwargs={
                "position_ids": position_ids,
                "permutation_order": perm_indices.argsort(dim=-1),
            },
        )

    def _prepare_inputs_inference(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        context: torch.LongTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        cache: Optional[Dict[str, Any]] = None,
        return_updated_cache: bool = False,
        position_ids: torch.LongTensor | None = None,
        **backbone_kwargs: Any,
    ) -> DenoiserInput:
        assert (
            input_ids is not None or context is not None or new_context_ids is not None
        ), "Must provide either input_ids or context."
        batch_size = input_ids.shape[0]
        decoder_attention_mask_context = None
        device = input_ids.device
        seq_len = input_ids.shape[1]
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        cache_len = self._get_past_key_values_seq_length(past_key_values)
        full_seq_length = cache_len + seq_len
        if self.mask_token_id not in input_ids: # Caching
            attention_mask = torch.tril(
                torch.ones(
                    (full_seq_length, full_seq_length),
                    dtype=torch.bool,
                ),
            )[-seq_len:].to(device)
            # randomly permute rows and cols
            # perm_indices_rows = torch.randperm(seq_len, device=device)
            # if full_seq_length > seq_len:
            #     perm_indices_cols = torch.cat((torch.arange(full_seq_length - seq_len).to(device), perm_indices_rows + (full_seq_length - seq_len)))
            # else:
            #     perm_indices_cols = perm_indices_rows
            # attention_mask = attention_mask[perm_indices_rows]
            # attention_mask = torch.gather(
            #     attention_mask, dim=-1, index=perm_indices_cols.unsqueeze(0).expand(seq_len, full_seq_length))
            attention_mask = self._preprocess_attention_mask(
                attention_mask[None, None, ...], dtype=torch.float
            )
        else:
            attention_mask = torch.tril(
                torch.ones(
                    (full_seq_length, full_seq_length),
                    dtype=torch.bool,
                ),
            )[-seq_len:].to(device)
            attention_mask[:, -seq_len:] = 0
            attention_mask[-torch.arange(1, seq_len+1), -torch.arange(1, seq_len+1)] = 1
            attention_mask = self._preprocess_attention_mask(attention_mask[None, None, ...], dtype=torch.float)
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            backbone_kwargs={
                "position_ids": position_ids,
            }
            | backbone_kwargs,
        ), cache

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.LongTensor | None = None,
        generation_config: DiffusionGenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        disable_pbar: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor:
        assert generation_config.use_cache, (
            "Generation with AO-ARM requires use_cache=True."
        )
        assert ~generation_config.align_inputs_to_blocks, (
            "Alignment of inputs to blocks not supported."
        )
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
        all_position_ids = torch.arange(
            inputs.shape[-1] + max_blocks * block_size, device=device
        )[None, :]

        # Sample max generation length tensor from prior
        accumulated_samples = self.mask_token_id * torch.ones(
            (batch_size, max_blocks * block_size), dtype=torch.int64, device=device
        )
        accumulated_samples = torch.cat([inputs, accumulated_samples], dim=-1)
        if generation_config.use_cache and inputs.numel() > 0:
            cache = self.update_cache(
                inputs=inputs,
                position_ids=all_position_ids[:, :inputs.shape[-1]],
                cache={},
            )
            inputs_offset = inputs.shape[-1]
        else:
            cache = None
            inputs_offset = 0

        total_NFEs = 0
        block_pbar = tqdm(
            range(max_blocks),
            desc="Blocks",
            leave=True,
            disable=disable_pbar,
        )
        for block_id in block_pbar:
            block_NFEs = 0
            xt = accumulated_samples[
                :,
                inputs_offset + (block_id * block_size) : inputs_offset
                + ((block_id + 1) * block_size)
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
            dt = (1 - generation_config.min_t) / len(timesteps)
            xt_position_ids = all_position_ids[
                :,
                inputs_offset + (block_id * block_size) : inputs_offset
                + ((block_id + 1) * block_size)
            ]
            for t in step_pbar:
                block_NFEs += 1
                total_NFEs += 1
                # t is 0-dim tensor, reshape to (1, 1, 1) for broadcasting
                alpha_t, _ = self.noise_schedule(t)
                alpha_s, _ = self.noise_schedule(t - dt)
                alpha_t = alpha_t[None, None, None]
                alpha_s = alpha_s[None, None, None]

                # Only decode masked tokens
                denoiser_inputs, cache = self._prepare_inputs_inference(
                    input_ids=xt[xt == self.mask_token_id].unsqueeze(0),
                    cache=cache,
                    position_ids=xt_position_ids[xt == self.mask_token_id].unsqueeze(0),
                )

                # Used for logit processing
                running_generation = accumulated_samples[
                    :,
                    inputs.shape[-1] : inputs.shape[-1]
                    + (block_id * block_size)
                ]

                xs, model_output_cache, cache = self._generate_unconditional(
                    generation_config=generation_config,
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                    denoiser_inputs=denoiser_inputs,
                    cache=cache,
                    xt=xt,
                    running_generation=running_generation,
                    logits_processor=logits_processor,
                    **kwargs,
                )
                block_pbar.set_postfix(
                    NFEs=total_NFEs,
                    block_NFEs=block_NFEs,
                )

                if (
                    not torch.allclose(xs, xt[xt == self.mask_token_id].unsqueeze(0))
                ):
                    model_output_cache = None
                accumulated_samples[:, xt_position_ids[xt == self.mask_token_id]] = xs
                if generation_config.use_cache:
                    # Enode unmasked tokens only
                    cache = self.update_cache(
                        inputs=xs[xs != self.mask_token_id].unsqueeze(0),
                        position_ids=xt_position_ids[xs != self.mask_token_id].unsqueeze(0),
                        cache=cache,
                    )
                # Update position IDs for next tokens to decode
                xt = xs[xs == self.mask_token_id].unsqueeze(0)
                xt_position_ids = xt_position_ids[xs == self.mask_token_id].unsqueeze(0)

                if (xt == self.mask_token_id).sum().item() == 0:
                    break
            if tokenizer is not None:
                print(tokenizer.batch_decode(accumulated_samples))
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=accumulated_samples[  # type: ignore
                        :,
                        inputs_offset : inputs_offset
                        + ((block_id + 1) * block_size)
                    ],
                    scores=None,  # type: ignore
                )
                if torch.any(is_done):
                    accumulated_samples = accumulated_samples[
                        :,
                        : inputs_offset + ((block_id + 1) * block_size),
                    ]
                    break
        return accumulated_samples  # type: ignore

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
        return log_probs  # type: ignore