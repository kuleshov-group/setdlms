from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
from transformers.generation import ExponentialDecayLengthPenalty
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache, DynamicCache
import math
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
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
from composer import State, Logger


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
        sampling_strategy: Literal["posterior", "predict_and_noise"] = "posterior",
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
                    - "predict_and_noise" - Sample from the denoising model x_theta,
                        then add back noise to produce x_s.
                        Only implemented for absorbing diffusion.
                Defaults to "posterior".
            confidence_based_noising (bool): When using the "predict_and_noise"
                strategy, whether to add noise to random positions or to those that have
                the lowest probability under x_theta.
                Cannot be used in conjunction with confidence_margin_based_noising.
                Defaults to False.
            confidence_margin_based_noising (bool): When using the "predict_and_noise"
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


@dataclass
class DiffusionGenerationOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either
            equal to `max_length` or shorter if all batches finished early due to the
            `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when
            `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each
            vocabulary token before SoftMax) at each generation step.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one
            element for each generated token), with each tensor of shape
            `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when
            `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each
            vocabulary token before SoftMax) at each generation step.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one
            element for each generated token), with each tensor of shape
            `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when
            `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each
            layer of the decoder) of `torch.FloatTensor` of shape
            `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when
            `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each
            layer of the decoder) of `torch.FloatTensor` of shape
            `(batch_size, generated_length, hidden_size)`.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a
            different cache format, check the model's documentation.
            Usually, a [`~cache_utils.Cache`] instance.
        parallelism_factor (float): The parallelism factor of the generation.
            Defaults to -1.0.
    """

    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None
    parallelism_factor: Optional[float] = None
    inf_budget: Optional[float] = None
    inf_budgets: Optional[list[float]] = None
    likelihoods: Optional[torch.FloatTensor] = None

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
        noise_mask = context_mask | ~(attention_mask.bool())
        xt = self._sample_q_xt(
            x0=input_ids,
            alpha_t=alpha_t,
            mask=noise_mask,
        )
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
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
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
        if self.config.keep_clean_bos and not self.training:
            log_p_theta = log_p_theta[:, 1:]
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask[:, 1:]
            denoiser_inputs.alpha_t_prime = denoiser_inputs.alpha_t_prime[:, 1:]
            denoiser_inputs.alpha_t = denoiser_inputs.alpha_t[:, 1:]
            denoiser_inputs.xt = denoiser_inputs.xt[:, 1:]
        block_size = getattr(self.config, "block_size", denoiser_inputs.x0.shape[-1])
        if block_size > 1 or getattr(self.config, "train_on_nelbo", False):
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

        if (self.training and not getattr(self.config, "train_on_nelbo", False)) or block_size == 1:
            # Average over masked tokens during training
            batch_nll = -(log_p_theta * denoiser_inputs.tokens_mask).sum(dim=-1)
            mask_token_indicator = (denoiser_inputs.xt == self.mask_token_id).float()
            count = mask_token_indicator.sum(dim=-1) # override count to be masked tokens
            token_nll = torch.where(count > 0, batch_nll / count, torch.zeros_like(batch_nll)).mean()
        else:
            # NELBO; average over response tokens
            count = denoiser_inputs.tokens_mask.sum(dim=-1)
            token_nll = torch.where(count > 0, batch_nll / count, torch.zeros_like(batch_nll)).mean()
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
        q_xs = x * (alpha_s[..., None] - alpha_t[..., None])
        q_xs[..., self.mask_token_id] = 1 - alpha_s
        # removed in mdlm: 
        # q_xs = torch.where(alpha_t[..., None] != 1, q_xs / (1 - alpha_t[..., None]), x)
        return q_xs  # type: ignore

    def _sample_generation_timesteps(
        self,
        generation_config: DiffusionGenerationConfig,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float64,
    ) -> torch.FloatTensor:
        """Sample timesteps for diffusion generation process."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if max_length is None:
            max_length = generation_config.max_new_tokens
        sampling_strategy = getattr(generation_config, "sampling_strategy", "posterior")
        if (
            getattr(generation_config, "first_hitting", False)
            and sampling_strategy == "posterior"
        ):
            fhs_times = self.noise_schedule.compute_first_hitting_times(
                batch_size=1, length=max_length, device=device, dtype=dtype)[0].sort(descending=True).values
            return fhs_times
        return torch.linspace(  # type: ignore
            1.0,
            0.0,
            generation_config.num_steps + 1,
            device=device,
            dtype=dtype,
        )[:-1]

    def _generate_unconditional(
        self,
        generation_config: DiffusionGenerationConfig,
        t: torch.FloatTensor,
        next_t: torch.FloatTensor,
        alpha_t: torch.FloatTensor,
        alpha_s: torch.FloatTensor,
        denoiser_inputs: Optional[DenoiserInput] = None,
        model_output_cache: Optional[Dict[str, torch.FloatTensor]] = None,
        cache: Optional[Dict[str, Any]] = None,
        running_generation: Optional[torch.LongTensor] = None,
        inputs_offset: Optional[int] = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
        sample_indices: Optional[Tuple[int, int]] = None,
        input_indices: Optional[Tuple[int, int]] = None,
        return_updated_cache: bool = False,
        cache_len: Optional[int] = None,
        eval_ground_truth: torch.LongTensor | None = None,
        window_size: int = 0,
        block_size: int = 0,
        **kwargs: Any,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.FloatTensor], Dict[str, Any]]:
        cache = cache if cache is not None else {}
        if model_output_cache is None:  # execute function evaluation
            backbone_output = self._backbone_forward(
                denoiser_inputs,
                fix_cache_length=True if not return_updated_cache else False,  # Do not let kv cache grow on each forward call
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
                if eval_ground_truth is not None:
                    eval_ground_truth = eval_ground_truth[:, cache_len:]
            if sample_indices is not None and input_indices is not None:
                logits = logits[:, sample_indices - input_indices[0], :]
                denoiser_inputs.xt = denoiser_inputs.xt[..., sample_indices - input_indices[0]]
            log_x_theta = self._forward(logits, denoiser_inputs, **kwargs)
            if logits_processor is not None:
                for lp in logits_processor:
                    # TODO: DEBUG
                    for i in range(log_x_theta.shape[1]):
                        if isinstance(lp, ExponentialDecayLengthPenalty):
                            log_x_theta[:, i] = lp(
                                input_ids=running_generation[..., inputs_offset:sample_indices[0] + i + 1],
                                scores=log_x_theta[:, i],  # type: ignore
                            )
                        else:
                            log_x_theta[:, i] = lp(
                                input_ids=running_generation[..., inputs_offset:sample_indices[-1]+1],
                                scores=log_x_theta[:, i],  # type: ignore
                            )
                log_x_theta = torch.log_softmax(log_x_theta, dim=-1)  # re-normalize
            x_theta = log_x_theta.exp()
        else:
            x_theta = model_output_cache["x_theta"]

        model_output_cache = {"x_theta": x_theta}
        sampling_strategy = getattr(generation_config, "sampling_strategy", "posterior")
        if sampling_strategy == "posterior":
            q_xs = self._compute_posterior(
                x_theta, denoiser_inputs.xt, alpha_t, alpha_s
            )
            # removed in mdlm (from removing denominator)
            # assert abs((q_xs.sum() / prob_check_denom).item() - 1.0) < 1e-6, (
            #     "Posterior probabilities not summing to 1."
            # )
            assert q_xs.isnan().sum().item() == 0, "NaN found in the posterior."
            xs = self._sample_categorical(q_xs, generation_config.do_sample)
            output = torch.where(
                (denoiser_inputs.xt != self.mask_token_id).bool(),  # type: ignore
                denoiser_inputs.xt,
                xs,
            )
        elif sampling_strategy == "predict_and_noise":
            # assert (
            #     abs((x_theta.sum() / prob_check_denom).item() - 1.0) < 1e-6
            # ), "Denoising output probabilities not summing to 1."
            # assert x_theta.isnan().sum().item() == 0, (
            #     "NaN found in the denoising output."
            # )

            # Predict
            xs = self._sample_categorical(x_theta, generation_config.do_sample)
            # valid_to_unmask = (alpha_s - alpha_t) > 0.0
            # xs[~valid_to_unmask] = self.mask_token_id
            
            xs_probs = x_theta.gather(-1, xs[..., None]).squeeze(dim=-1)
            output = xs.clone()

            # Noise
            est_noise_indices_next = (next_t * block_size).to(torch.int)
            est_noise_indices_curr = (t * block_size).to(torch.int)
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
                noise_indices = conf.argsort(dim=-1)[..., :num_noise_indices]
            else:
                # Always decode the most confident token
                conf = x_theta.gather(-1, xs[..., None]).squeeze(-1)
                conf = torch.where(  # already decoded tokens have 'inf' confidence
                    (denoiser_inputs.xt == self.mask_token_id).bool(),  # type: ignore
                    conf,
                    torch.inf,
                )
                num_clean_indices = (denoiser_inputs.xt != self.mask_token_id).sum(-1) + num_to_decode
                noise_indices = conf.argsort(dim=-1)[..., :-num_clean_indices[0]]
            output[..., noise_indices] = self.mask_token_id
            output = torch.where(
                xs_probs >= generation_config.confidence_threshold, xs, output
            )
            output = torch.where(denoiser_inputs.xt == self.mask_token_id, output, denoiser_inputs.xt)
        else:
            raise NotImplementedError(
                f"Sampling strategy {sampling_strategy} not"
                " implemented."
            )
        if getattr(generation_config, "save_likelihoods", False):
            xt = denoiser_inputs.xt
            if "permutation_order" in denoiser_inputs.backbone_kwargs and denoiser_inputs.backbone_kwargs["permutation_order"] is not None:
                position_ids = denoiser_inputs.backbone_kwargs["position_ids"][:, cache_len:]
                inv_perm = position_ids.argsort(dim=-1)          # (B, L)  unordered -> ordered
                inv_inv_perm = inv_perm.argsort(dim=-1)   
                log_x_theta = log_x_theta.gather(1, inv_perm.unsqueeze(-1).repeat(1, 1, log_x_theta.shape[-1]))
                xt = torch.gather(xt, 1, inv_perm)
                output = torch.gather(output, 1, inv_perm)
                eval_ground_truth = torch.gather(eval_ground_truth, 1, inv_perm)
                position_ids = torch.gather(position_ids, 1, inv_perm)
            log_x_theta = log_x_theta.gather(-1, eval_ground_truth[..., None]).squeeze(-1)
            # select indices
            log_x_theta_clone = torch.where(xt == self.mask_token_id, log_x_theta, -torch.inf)

            # set tokens outside of window size to -inf
            if "permutation_order" in denoiser_inputs.backbone_kwargs and denoiser_inputs.backbone_kwargs["permutation_order"] is not None:
                window_start = (xt == self.mask_token_id).float().argmax(dim=-1)[:, None]
                log_x_theta_clone = torch.where((position_ids - position_ids.min(dim=-1).values[:, None]) < (window_start + window_size), log_x_theta_clone, -torch.inf)
            # for i in range(xt.shape[0]):
            #     masked_indices = (xt[i] == self.mask_token_id).nonzero(as_tuple=True)[-1]
            #     selected_index = torch.randint(0, masked_indices.shape[-1], (1,))
            #     selected_index = masked_indices[selected_index]
            #     log_x_theta_clone[i, selected_index] = torch.inf
            selected_indices = log_x_theta_clone.argsort(dim=-1, descending=True, stable=True)

            tokens_kept = torch.gather(eval_ground_truth, dim=-1, index=selected_indices[..., :1])
            output = torch.where(xt == self.mask_token_id, self.mask_token_id, output) # bypass original sampling process
            output.scatter_(dim=-1, index=selected_indices[..., 0].unsqueeze(-1), src=tokens_kept)
            log_x_theta.scatter_(dim=-1, index=selected_indices[..., 1:], src=torch.zeros_like(log_x_theta[..., 1:]))
            if "permutation_order" in denoiser_inputs.backbone_kwargs and denoiser_inputs.backbone_kwargs["permutation_order"] is not None:
                # permute back
                output = torch.gather(output, 1, inv_inv_perm)
            return output, model_output_cache, cache, log_x_theta
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
        return_dict_in_generate: Optional[bool] = False,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        eval_ground_truth: torch.LongTensor | None = None,
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
            if max_length is not None:
                max_new_tokens = max_length - inputs.shape[-1]
            else:
                if hasattr(generation_config, "max_new_tokens"):
                    max_new_tokens = generation_config.max_new_tokens
                else:
                    max_new_tokens = max_length - inputs.shape[-1]
        batch_size = inputs.shape[0]
        block_size = generation_config.block_size
        is_infill_task = self.mask_token_id in inputs
        pad_length = None
        if is_infill_task:
            num_mask_tokens = (inputs == self.mask_token_id).sum()
            if generation_config.align_inputs_to_blocks:
                pad_length = block_size - inputs.shape[-1] % block_size
                inputs = F.pad(inputs, (0, pad_length), value=self.mask_token_id)
                mask_tokens = (inputs == self.mask_token_id)
                mask_tokens[:, -pad_length:] = False
                mask_tokens = mask_tokens.view(-1, block_size)
                max_blocks = (mask_tokens.max(dim=-1).values == 1).sum()
            else:
                max_blocks = math.ceil(num_mask_tokens / block_size)
        else:
            num_mask_tokens = max_new_tokens
            max_blocks = math.ceil(num_mask_tokens / block_size)
            
        # Sample max generation length tensor from prior
        if is_infill_task:
            accumulated_samples = inputs
        else:
            accumulated_samples = self.mask_token_id * torch.ones(
                (batch_size, num_mask_tokens), dtype=torch.int64, device=device
            )
            accumulated_samples = torch.cat([inputs, accumulated_samples], dim=-1)
            if getattr(self.config, "block_size", self.config.length) == self.config.length and accumulated_samples.shape[-1] < self.config.length:
                accumulated_samples = F.pad(
                    accumulated_samples,
                    (0, self.config.length - accumulated_samples.shape[-1]),
                    value=self.tokenizer.pad_token_id)
        likelihoods = None
        if getattr(generation_config, "save_likelihoods", False):
            likelihoods = torch.zeros((batch_size, num_mask_tokens + inputs.shape[-1]), dtype=torch.float, device=device)
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

        if is_infill_task:
            inputs_offset = (accumulated_samples == self.mask_token_id)[0].nonzero().min()
        else:
            inputs_offset = inputs.shape[-1] if inputs.numel() > 0 else 0

        if generation_config.align_inputs_to_blocks:
            inputs_offset = (
                block_size * (inputs_offset // block_size)
                if inputs_offset > 0
                else 0
            )
        

        total_NFEs = 0
        timesteps = self._sample_generation_timesteps(  # Re-use in every block
            generation_config, max_length=block_size, device=device
        )
        dt = 1 / len(timesteps)
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
        if getattr(self.config, "block_size", self.config.length) == self.config.length:
            if inputs_offset < self.config.length:
                start_input_idx = 0
                end_input_idx = self.config.length
                start_sample_idx = inputs_offset
                end_sample_idx = min(start_sample_idx + block_size, inputs_offset + max_new_tokens)
            else:
                start_input_idx = inputs_offset - self.config.length + 32
                end_input_idx = start_input_idx + self.config.length
                start_sample_idx = inputs_offset
                end_sample_idx = min(start_sample_idx + block_size, end_input_idx)
        for block_id in block_pbar:
            block_NFEs = 0
            if getattr(self.config, "block_size", self.config.length) == self.config.length:
                if block_id > 0:
                    start_sample_idx += block_size
                    end_sample_idx += block_size
                if start_sample_idx >= self.config.length:
                    end_input_idx = end_sample_idx
                    start_input_idx = end_input_idx - self.config.length
                xt = accumulated_samples[:, start_input_idx:end_input_idx]
                end_sample_idx = min(end_sample_idx, self.config.length)
                end_input_idx = min(end_input_idx, self.config.length)
                sample_indices = torch.arange(start_sample_idx, end_sample_idx)
                input_indices = (start_input_idx, end_input_idx)
            elif generation_config.use_cache:
                xt = accumulated_samples[
                    :,
                    inputs_offset + (block_id * block_size) : inputs_offset
                    + ((block_id + 1) * block_size),
                ]
                sample_indices = torch.arange(inputs_offset + (block_id * block_size), inputs_offset + ((block_id + 1) * block_size))
                input_indices = None
            else:
                xt = accumulated_samples[
                    :, : inputs_offset + ((block_id + 1) * block_size)]
                sample_indices = torch.arange(inputs_offset + (block_id * block_size), min(inputs_offset + ((block_id + 1) * block_size), accumulated_samples.shape[-1]))
                input_indices = (0, inputs_offset + ((block_id + 1) * block_size))
            if pad_length is not None:
                if self.mask_token_id not in xt[:, :-pad_length]:
                    continue
            if self.mask_token_id not in xt:
                continue
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
            disable_pbar = rank != 0 or kwargs.get("disable_pbar", False)

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
            if getattr(self.config, "block_size", self.config.length) == self.config.length:
                running_generation = accumulated_samples
            else:
                running_generation = accumulated_samples[
                    :,
                    : inputs_offset + (block_id * block_size),
                ]
            if pad_length is not None:
                sample_indices = sample_indices[:-pad_length]
            for i,t in enumerate(step_pbar):
                if model_output_cache is None:
                    block_NFEs += 1
                    total_NFEs += 1
                # t is 0-dim tensor, reshape to (1, 1, 1) for broadcasting
                if getattr(generation_config, "linear_unmasking", False):
                    alpha_t = torch.ones_like(t) - t
                    alpha_s = torch.ones_like(t) - (t - dt)
                else:
                    alpha_t, _ = self.noise_schedule(t)
                    alpha_s, _ = self.noise_schedule(t - dt)
                alpha_t = alpha_t[None, None]
                alpha_s = alpha_s[None, None]
                denoiser_inputs, cache = self._prepare_inputs_inference(
                    input_ids=xt,
                    context=context,
                    cache=cache if generation_config.use_cache else None,
                )
                next_t = timesteps[i+1] if i < timesteps.shape[-1] - 1 else timesteps[-1] * 0
                generation_output = self._generate_unconditional(
                    generation_config=generation_config,
                    block_size=block_size,
                    t=t,
                    next_t=next_t,
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                    denoiser_inputs=denoiser_inputs,
                    model_output_cache=model_output_cache,
                    cache=cache,
                    running_generation=running_generation,  # type: ignore
                    inputs_offset=inputs_offset,
                    logits_processor=logits_processor,
                    tokenizer=tokenizer,
                    sample_indices=sample_indices,
                    input_indices=input_indices,
                    eval_ground_truth=None if eval_ground_truth is None else eval_ground_truth[..., sample_indices],
                    **kwargs,
                )
                if getattr(generation_config, "save_likelihoods", False):
                    xs, model_output_cache, cache, conf = generation_output
                    # likelihoods[:, sample_indices][(xs != denoiser_inputs.xt).bool()] = conf[(xs != denoiser_inputs.xt).bool()]
                    likelihoods[:, sample_indices] += torch.where(conf >= -1e6, conf, 0.0)
                else:
                    xs, model_output_cache, cache = generation_output
                block_pbar.set_postfix(
                    NFEs=total_NFEs,
                    block_NFEs=block_NFEs,
                )
                num_tokens_generated_per_step.append(
                    (xs != denoiser_inputs.xt).sum().item()
                )
                if getattr(generation_config, "compute_inf_budget", False):
                    alpha_t_prime = alpha_s - alpha_t
                    inf_budget = ((denoiser_inputs.xt == self.mask_token_id) & (alpha_t_prime != 0.0)).sum().item()
                    inf_budget_per_step.append(inf_budget)
                if not torch.allclose(xs, denoiser_inputs.xt):
                    model_output_cache = None
                if input_indices is not None:
                    xt[..., sample_indices - input_indices[0]] = xs
                    if (xt[..., sample_indices - input_indices[0]] == self.mask_token_id).sum().item() == 0:
                        break
                else:
                    xt = xs
                if (xt == self.mask_token_id).sum().item() == 0:
                    if getattr(generation_config, "compute_inf_budget", False):
                        remaining_steps = timesteps.shape[0] - len(inf_budget_per_step)
                        inf_budget_per_step.extend([0] * remaining_steps)
                    break
            if input_indices is not None:
                accumulated_samples[..., sample_indices] = xt[..., sample_indices - input_indices[0]]
            else:
                accumulated_samples[:, torch.arange(inputs_offset + (block_id * block_size), inputs_offset + ((block_id + 1) * block_size))] = xt[..., -block_size:]
            if tokenizer is not None:  # Useful for debugging
                print(tokenizer.batch_decode(accumulated_samples))
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
                    accumulated_samples = accumulated_samples[accumulated_samples != self.mask_token_id].unsqueeze(0)
                    break
            if generation_config.use_cache and getattr(self.config, "block_size", self.config.length) < self.config.length:
                cache = self.update_cache(
                    inputs=xt,
                    cache=cache,
                )
        if pad_length is not None:
            accumulated_samples = accumulated_samples[:, : -pad_length]
        parallelism_factor = sum(num_tokens_generated_per_step) / len(
            num_tokens_generated_per_step
        )
        inf_budget = None
        if getattr(generation_config, "compute_inf_budget", False):
            inf_budget = sum(inf_budget_per_step) / len(inf_budget_per_step)
        if return_dict_in_generate:
            return DiffusionGenerationOutput(
                sequences=accumulated_samples,
                parallelism_factor=parallelism_factor,
                inf_budget=inf_budget,
                inf_budgets=inf_budget_per_step,
                likelihoods=likelihoods,
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
        static_mask = self.generate_static_mask()

        if self.config.attn_backend == "flex_attention":
            self.static_attention_mask = static_mask
        else:
            self.register_buffer(
                "static_attention_mask",
                static_mask,
            )
            self.skip_params_for_push.append("static_attention_mask")

    def generate_static_mask(self) -> Union[torch.Tensor, BlockMask]:
        if self.config.attn_backend == "flex_attention":
            mask = partial(
                self._block_mask,
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
                seq_length=self.config.length,
            )
            return create_block_mask(
                mask,
                B=None,
                H=None,
                Q_LEN=self.config.length * 2,
                KV_LEN=self.config.length * 2,
            )
        else:
            static_mask = self._block_mask(
                b=None,
                h=None,
                q_idx=torch.arange(self.config.length * 2)[:, None],
                kv_idx=torch.arange(self.config.length * 2)[None, :],
                block_size=self.config.block_size,
                seq_length=self.config.length,
            )
            return static_mask

    def update_static_mask(
        self,
        new_static_mask: Union[torch.Tensor, BlockMask],
    ) -> None:
        self.static_attention_mask.copy_(new_static_mask)

    def _ensure_no_unmasked_blocks(
        self,
        input_ids: torch.LongTensor,
        xt: torch.LongTensor,
        context_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        pad_length = self.config.block_size - (xt.shape[1] % self.config.block_size)
        if pad_length > 0:
            xt = F.pad(xt, (0, pad_length), value=self.pad_token_id)
            context_mask = F.pad(context_mask, (0, pad_length), value=0)
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
        if pad_length > 0:
            xt = xt[:, :-pad_length]
            context_mask = context_mask[:, :-pad_length]
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
                    partial(self._block_mask, block_size=self.config.block_size),
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

    # noinspection PyUnusedLocal
    @staticmethod
    def _block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        seq_length: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del b, h

        # Indicate whether token belongs to xt or x0:
        xt_flag_q = (q_idx >= seq_length).bool()
        xt_flag_kv = (kv_idx >= seq_length).bool()

        q_idx = q_idx % seq_length
        kv_idx = kv_idx % seq_length

        # **1. Offset Causal Mask **
        offset_causal = (q_idx > kv_idx) & ~xt_flag_kv & xt_flag_q

        # **2. Diagonal Mask **
        diagonal = (q_idx == kv_idx) & (xt_flag_q == xt_flag_kv)

        # **3. Causal Mask **
        causal = (q_idx >= kv_idx) & ~xt_flag_kv & ~xt_flag_q

        # **4. Combine Masks **
        return diagonal | offset_causal | causal

    # noinspection PyUnusedLocal
    @staticmethod
    def _block_mask_eso(
        b,
        h,
        q_idx,
        kv_idx,
        seq_length: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del b, h

        # Indicate whether token belongs to xt or x0:
        xt_flag_q = (q_idx >= seq_length).bool()
        xt_flag_kv = (kv_idx >= seq_length).bool()

        q_idx = q_idx % seq_length
        kv_idx = kv_idx % seq_length

        # **1. Offset Causal Mask **
        offset_causal = (q_idx > kv_idx) & ~xt_flag_kv & xt_flag_q

        # **2. Causal Mask **
        diagonal = (q_idx >= kv_idx) & (xt_flag_q == xt_flag_kv)

        # **3. Causal Mask **
        causal = (q_idx >= kv_idx) & ~xt_flag_kv & ~xt_flag_q

        # **4. Combine Masks **
        return diagonal | offset_causal | causal

    def _create_static_mask(self) -> None:
        if self.config.attn_backend == "flex_attention":
            seq_length = self.config.length
            def mask_mod(b, h, q_idx, kv_idx):
                return self._block_mask(b, h, q_idx, kv_idx, seq_length=seq_length)
            self.static_attention_mask = create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=self.config.length * 2,
                KV_LEN=self.config.length * 2,
            )
        elif self.config.attn_backend == "sdpa":
            if getattr(self.config, "inefficient_training", False):
                mask = self._block_mask_eso(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.length * 2)[:, None],
                    kv_idx=torch.arange(self.config.length * 2)[None, :],
                    seq_length=self.config.length,
                )
            else:
                mask = self._block_mask(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.length * 2)[:, None],
                    kv_idx=torch.arange(self.config.length * 2)[None, :],
                    seq_length=self.config.length,
                )
            self.register_buffer("static_attention_mask", mask)
        else:
            raise ValueError("Unknown attention backend")

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        input_length = denoiser_inputs.x0.shape[1]
        model_output = model_output[:, input_length:input_length*2, ...]
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=denoiser_inputs.x0[:, :, None] #.repeat(1, num_repetitions, 1)
        ).squeeze(-1)
        if self.config.keep_clean_bos and not self.training:
            log_p_theta = log_p_theta[:, 1:]
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask[:, 1:]
            denoiser_inputs.x0 = denoiser_inputs.x0[:, 1:]
            denoiser_inputs.attention_mask = denoiser_inputs.attention_mask[:, 1:]
            denoiser_inputs.alpha_t_prime = denoiser_inputs.alpha_t_prime[:, 1:]
            denoiser_inputs.alpha_t = denoiser_inputs.alpha_t[:, 1:]
        loss = -log_p_theta
        if not self.training:
            denoiser_inputs.tokens_mask = denoiser_inputs.tokens_mask * (
                denoiser_inputs.x0 != self.pad_token_id
            )
        inefficient_eval = getattr(self.config, "inefficient_training", False) or getattr(self.config, "eval_nll", False)
        if not self.training and inefficient_eval and self.config.block_size > 1:
            coeff = denoiser_inputs.alpha_t_prime / (1 - denoiser_inputs.alpha_t)
            coeff = torch.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
            seq_len = denoiser_inputs.x0.shape[1]
            masked_indices = denoiser_inputs.xt[:, -seq_len:] == self.mask_token_id
            nlls = (
                log_p_theta
                * coeff
                * masked_indices
            )
        else:
            nlls = loss * denoiser_inputs.tokens_mask
            if getattr(self.config, "inefficient_training", False):
                nlls *= (denoiser_inputs.alpha_t_prime != 0.0)

        # Compute per-batch counts and losses to avoid division by zero
        count = denoiser_inputs.tokens_mask.sum(dim=-1)  # Per-batch counts
        batch_nll = nlls.sum(dim=-1)  # Per-batch losses

        # TODO: DEBUG
        # sm_output = F.log_softmax(model_output, dim=-1).exp()
        # pred = sm_output.argmax(dim=-1)
        # conf = torch.gather(
        #     input=sm_output, dim=-1, index=pred[:, :, None]
        # ).squeeze(-1)
        
        # Avoid division by zero: if count is 0, set token_nll to 0
        token_nll = torch.where(
            count > 0,
            batch_nll / count,
            torch.zeros_like(batch_nll)
        ).mean()
        
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
        if self.config.keep_clean_bos:
            permute_flag[:, 0] = False
        perm_indices = None
        xt = input_ids.clone()
        evaluate_nll_flag = getattr(self.config, "eval_nll", False) and not self.training and self.config.block_size > 1
        evaluate_nll_flag = evaluate_nll_flag or getattr(self.config, "inefficient_training", False)
        if not evaluate_nll_flag:
            xt = torch.where(
                (attention_mask == 1) & (context_mask == 0), self.mask_token_id, xt
            )
        else:
            xt = self._sample_q_xt(
                x0=input_ids,
                alpha_t=alpha_t,
                mask=noise_mask,
            )
            if getattr(self.config, "inefficient_training", False):
                xt = torch.where(
                   alpha_t_prime != 0.0, self.mask_token_id, xt
                )

        batch_size, seq_len = input_ids.shape
        num_repetitions = 2
        if permute_flag.any():
            with torch.no_grad():
                perm_indices = self.noise_schedule.sample_permutation_order(
                    t,
                    permute_flag,
                    self.config.block_size if self.training else self.config.eval_block_size,
                    masked_tokens=(xt == self.mask_token_id) if evaluate_nll_flag else None,
                )
        else:
            perm_indices = torch.arange(seq_len, device=input_ids.device)[None, :]
        
        alpha_t = torch.gather(alpha_t, dim=-1, index=perm_indices)
        alpha_t_prime = torch.gather(alpha_t_prime, dim=-1, index=perm_indices)
        if self.config.attn_backend == "sdpa":
            decoder_attention_mask = (
                self.static_attention_mask[None, ...]
                & attention_mask.repeat(1, num_repetitions)[:, None, :]
                & attention_mask.repeat(1, num_repetitions)[..., None]
            )
        elif self.config.attn_backend == "flex_attention":
            if context_mask.any():
                raise NotImplementedError(
                    "flex_attention with context_mask not implemented yet."
                )
            elif attention_mask is not None and (attention_mask != 1).any():
                padding_mask = create_attn_mask(attention_mask.repeat(1, 2).bool())
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
                    Q_LEN=input_ids.shape[1],
                    KV_LEN=input_ids.shape[1] * 2,
                )
            else:
                decoder_attention_mask = self.static_attention_mask
        else:
            raise ValueError("Unknown attention backend")

        xt = torch.gather(xt, dim=-1, index=perm_indices)
        input_ids = torch.gather(input_ids, dim=-1, index=perm_indices)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(batch_size, 1)
        position_ids = torch.gather(position_ids, dim=-1, index=perm_indices)

        if evaluate_nll_flag:
            xt = xt.repeat(1, 2)
        else:
            xt = torch.cat((input_ids, xt), dim=-1)
        position_ids = position_ids.repeat(1, num_repetitions)

        # TODO
        if evaluate_nll_flag:
            assert self.config.attn_backend == "sdpa", "eval_nll flag only supported for sdpa backend"
            for i in range(batch_size):
                if self.mask_token_id not in xt[i]:
                    continue
                first_mask_token_idx = (xt[i] == self.mask_token_id).float().argmax().item()
                masked_indices = torch.arange(first_mask_token_idx + seq_len, seq_len*2)

                # masked tokens may only attend to clean tokens and previous masked tokens
                clean_indices = torch.arange(seq_len, first_mask_token_idx + seq_len)
                decoder_attention_mask[i][masked_indices[:, None], clean_indices[None, :]] = False
                decoder_attention_mask[i][masked_indices[:, None], masked_indices[None, :] - seq_len] = False
        if self.config.attn_backend == "sdpa":
            decoder_attention_mask = self._preprocess_attention_mask(
                decoder_attention_mask[:, None], dtype=torch.float
            )
        tokens_mask = attention_mask * (1 - context_mask)
        
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
                "permutation_order": perm_indices,
            },
        )

    def update_cache(
        self,
        inputs: torch.LongTensor,
        cache: Optional[Dict[str, Any]] = None,
        first_hitting_times: Optional[torch.LongTensor] = None,
        **backbone_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Cache the key-value pairs for the context.
        Args:
            inputs (torch.LongTensor): The context tensor.
            cache (Dict[str, Any | None): Cache objects, e.g., past_key_values.
        Returns:
            Dict: Updated cache objects, e.g., past_key_values.
        """
        context_input, cache = self._prepare_inputs_inference(
            input_ids=inputs, cache=cache, return_updated_cache=True, first_hitting_times=first_hitting_times, **backbone_kwargs
        )
        backbone_output = self._backbone_forward(
            context_input,
            **cache,
        )
        backbone_output = {k: v for k, v in backbone_output.items()}
        backbone_output.pop("logits", None)  # Do not store logits in cache
        cache = cache | backbone_output
        return cache

    def _prepare_inputs_inference(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        context: torch.LongTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        cache: Optional[Dict[str, Any]] = None,
        return_updated_cache: bool = False,
        position_ids: torch.LongTensor | None = None,
        first_hitting_times: torch.LongTensor | None = None,
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
        perm_indices = None
        # randomly permute rows and cols
        if first_hitting_times is not None:
            # mask tokens at the end
            fhs = torch.where(input_ids == self.mask_token_id, -1e6, first_hitting_times)
            perm_indices = fhs.argsort(dim=-1, stable=True, descending=True)
        else:
            perm_indices = torch.arange(seq_len, device=device)[None, :].repeat(batch_size, 1)
        if return_updated_cache:
            attention_mask = self.static_attention_mask[cache_len:cache_len+seq_len, :cache_len+seq_len].clone()
            position_ids = torch.gather(position_ids, dim=-1, index=perm_indices)
            input_ids = torch.gather(input_ids, dim=-1, index=perm_indices)
            if self.mask_token_id in input_ids:
                first_mask_token_idx = (input_ids == self.mask_token_id).float().argmax(dim=-1)[0]
                # mask tokens cannot attend to other masked tokens
                attention_mask[first_mask_token_idx:, -(seq_len-first_mask_token_idx):] = 0
            # keep self-attention
            attention_mask[-torch.arange(1, seq_len+1), -torch.arange(1, seq_len+1)] = 1
            attention_mask = self._preprocess_attention_mask(
                attention_mask[None, None, ...], dtype=torch.float
            )
        else:
            full_possible_len = self.static_attention_mask.shape[-1] // 2
            valid_attn_indices = torch.cat(
                (torch.arange(cache_len),
                torch.arange(cache_len, cache_len+seq_len) + full_possible_len,)
            )
            attention_mask = self.static_attention_mask[cache_len:cache_len+seq_len, valid_attn_indices].clone()
            attention_mask[-torch.arange(1, seq_len+1), -torch.arange(1, seq_len+1)] = 1
            attention_mask = self._preprocess_attention_mask(attention_mask[None, None, ...], dtype=torch.float)
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            backbone_kwargs={
                "position_ids": position_ids,
                "permutation_order": perm_indices,
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
        return_dict_in_generate: Optional[bool] = False,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer: PreTrainedTokenizer | None = None,
        eval_ground_truth: torch.LongTensor | None = None,
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
            if max_length is not None:
                max_new_tokens = max_length - inputs.shape[-1]
            else:
                if hasattr(generation_config, "max_new_tokens"):
                    max_new_tokens = generation_config.max_new_tokens
                else:
                    max_new_tokens = max_length - inputs.shape[-1]
        batch_size = inputs.shape[0] if inputs is not None else batch_size
        is_infill_task = self.mask_token_id in inputs
        block_size = generation_config.block_size
        window_size = self.noise_schedule.compute_window_size()
        window_size = min(generation_config.max_window_size, window_size)

        pad_length = None
        if is_infill_task:
            mask_tokens = (inputs == self.mask_token_id)
            if generation_config.align_inputs_to_blocks:
                pad_length = block_size - inputs.shape[-1] % block_size
                inputs = F.pad(inputs, (0, pad_length), value=self.mask_token_id)
                mask_tokens[:, -pad_length:] = False
                mask_tokens = mask_tokens.view(-1, block_size)
                max_blocks = (mask_tokens.max(dim=-1).values == 1).sum()
            else:
                max_blocks = math.ceil(mask_tokens.sum() / block_size)
            all_position_ids = torch.arange(inputs.shape[-1], device=device)[None, :].repeat(batch_size, 1)
        else:
            max_blocks = max(max_new_tokens // block_size, 1)
            all_position_ids = torch.arange(
                inputs.shape[-1] + max_blocks * block_size, device=device
            )[None, :].repeat(batch_size, 1)

        # Sample max generation length tensor from prior
        if is_infill_task:
            accumulated_samples = inputs
        else:
            accumulated_samples = self.mask_token_id * torch.ones(
                (batch_size, max_blocks * block_size), dtype=torch.int64, device=device
            )
            accumulated_samples = torch.cat([inputs, accumulated_samples], dim=-1)
        likelihoods = None
        if getattr(generation_config, "save_likelihoods", False):
            likelihoods = torch.zeros((batch_size, max_blocks * block_size + inputs.shape[-1]), dtype=torch.float, device=device)
        first_hitting_times = self.noise_schedule.compute_first_hitting_times(
            batch_size=batch_size,
            length=max_blocks * block_size,
            device=device,
        )
        if generation_config.use_cache and inputs.numel() > 0:
            if is_infill_task:
                inputs_indices = (inputs != self.mask_token_id)[0]
            else:
                inputs_indices = torch.arange(inputs.shape[-1], device=device)
            cache = self.update_cache(
                inputs=inputs[:, inputs_indices],
                position_ids=all_position_ids[:, inputs_indices],
                cache={},
            )
        else:
            cache = {
                "past_key_values": DynamicCache(),
            }
            inputs_indices = torch.empty(0, device=device)

        total_NFEs = 0
        is_done = False
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
        if self.mask_token_id in accumulated_samples:
            inputs_offset = (accumulated_samples == self.mask_token_id)[0].nonzero().min()
        else:
            inputs_offset = accumulated_samples.shape[-1]
        if generation_config.align_inputs_to_blocks and inputs_offset > 0:
            inputs_offset = block_size * (inputs_offset // block_size)
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
            timesteps = timesteps[None, :].repeat(batch_size, 1)
            dt = 1 / timesteps.shape[-1]
            xt_position_ids = all_position_ids[
                :,
                inputs_offset + (block_id * block_size) : inputs_offset
                + ((block_id + 1) * block_size)
            ]
            masked_positions = (xt == self.mask_token_id)
            window_start = inputs_offset + (block_id * block_size) + masked_positions.float().argmax(dim=-1)[:, None]
            masked_positions = masked_positions & (xt_position_ids < (window_start + window_size))

            clean_len = 0

            for i in range(timesteps.shape[-1]):
                block_NFEs += 1
                total_NFEs += 1
                t = timesteps[:, i].unsqueeze(1).repeat(1, block_size)
                if getattr(generation_config, "first_hitting", False):
                    num_generated = sum(num_tokens_generated_per_step)
                    next_t = timesteps[:, num_generated + 1] if num_generated < timesteps.shape[-1] else timesteps[:, -1] * 0
                else:
                    next_t = t[:, i+1] if i < timesteps.shape[-1] - 1 else t[:, -1] * 0
                next_t = next_t.unsqueeze(1).repeat(1, block_size)
                alpha_t, _ = self.noise_schedule(t)
                alpha_s, _ = self.noise_schedule(next_t)

                masked_positions_indices = masked_positions.nonzero(as_tuple=False)[:, -1].view(batch_size, -1)
                masked_xt = torch.gather(xt, dim=-1, index=masked_positions_indices)
                masked_position_ids = torch.gather(xt_position_ids, dim=-1, index=masked_positions_indices)
                masked_first_hitting_times = torch.gather(first_hitting_times, dim=-1, index=masked_positions_indices)

                # Only decode masked tokens
                cache_len = 0 if cache.get("past_key_values", None) is None else cache["past_key_values"].get_seq_length()
                denoiser_inputs, cache = self._prepare_inputs_inference(
                    input_ids=masked_xt,
                    cache=cache,
                    position_ids=masked_position_ids,
                    first_hitting_times=masked_first_hitting_times,
                    return_updated_cache=True if (xt != self.mask_token_id).any() else False,
                )
                alpha_t_block = torch.gather(alpha_t, dim=-1, index=masked_positions_indices)
                alpha_s_block = torch.gather(alpha_s, dim=-1, index=masked_positions_indices)
                return_updated_cache = True if i > 0 or block_id > 0 else False
                if eval_ground_truth is not None:
                    eval_ground_truth_t = torch.gather(eval_ground_truth[:, inputs_offset:], dim=-1, index=masked_positions_indices)
                    eval_ground_truth_t = torch.gather(eval_ground_truth_t, dim=-1, index=denoiser_inputs.backbone_kwargs["permutation_order"])
                else:
                    eval_ground_truth_t = None
                generation_output = self._generate_unconditional(
                    generation_config=generation_config,
                    alpha_t=alpha_t_block,
                    alpha_s=alpha_s_block,
                    t=t[0][0],
                    next_t=next_t[0][0],
                    denoiser_inputs=denoiser_inputs,
                    cache=cache,
                    xt=xt,
                    running_generation=accumulated_samples[:, :(masked_positions.cumsum(-1) > 0).float().argmax(dim=-1) + inputs_offset] if logits_processor is not None else None, # used for logit processing
                    inputs_offset=inputs_offset,
                    logits_processor=logits_processor,
                    return_updated_cache=return_updated_cache,
                    cache_len=clean_len,
                    eval_ground_truth=eval_ground_truth_t,
                    sample_indices=masked_positions_indices[:, clean_len:] + inputs_offset,
                    window_size=window_size,
                    block_size=block_size,
                    **kwargs,
                )
                if getattr(generation_config, "save_likelihoods", False):
                    xs, model_output_cache, cache, conf = generation_output
                else:
                    xs, model_output_cache, cache = generation_output
                # crop kv cache and sampling output
                if return_updated_cache:
                    # assert cache_len + clean_len <= cache["past_key_values"].get_seq_length(), "Crop length is greater than cache length"
                    # assert cache_len + clean_len > 0, "Crop length is less than 1"
                    # only keep cache for the clean tokens
                    cache["past_key_values"].crop(cache_len + clean_len)
                    position_ids = denoiser_inputs.backbone_kwargs["position_ids"]
                    # update accumulated_samples, xt, masked_positions
                    cached_position_ids = position_ids[:, :clean_len]
                    unmasked_position_ids = position_ids[:, clean_len:]
                    clean_len = (torch.gather(xt, dim=-1, index=(unmasked_position_ids - inputs_offset)) != xs).sum(dim=-1).min()
                    accumulated_samples.scatter_(1, unmasked_position_ids, xs)
                    masked_positions.scatter_(1, cached_position_ids - inputs_offset, False)
                    if getattr(generation_config, "save_likelihoods", False):
                        likelihoods = torch.scatter_add(likelihoods, 1, unmasked_position_ids.sort(dim=-1).values, torch.where(conf >= -1e6, conf, 0.0))
                    xt.scatter_(1, unmasked_position_ids - inputs_offset, xs)
                else:
                    xt.scatter_(1, masked_position_ids - inputs_offset, xs)
                    clean_len = (xs != self.mask_token_id).sum(dim=-1).min()
                    accumulated_samples.scatter_(1, masked_position_ids, xs)
                    if getattr(generation_config, "save_likelihoods", False):
                        likelihoods = torch.scatter_add(likelihoods, 1, masked_position_ids.sort(dim=-1).values, torch.where(conf >= -1e6, conf, 0.0))
                window_start = (accumulated_samples == self.mask_token_id).float().argmax(dim=-1)[:, None]
                masked_positions |= (xt == self.mask_token_id) & (xt_position_ids < (window_start + window_size))
                max_masks_to_add = masked_positions.sum(dim=-1).max()
                extra_masks_to_add = max_masks_to_add - masked_positions.sum(dim=-1)
                last_masked_pos = torch.gather(xt_position_ids, dim=-1, index=(masked_positions | (xt != self.mask_token_id)).cumsum(-1).argmax(-1)[:, None]) * (extra_masks_to_add > 0)[:, None]
                masked_positions |= ((xt == self.mask_token_id) & (xt_position_ids <= (last_masked_pos + extra_masks_to_add[:, None])))
                assert (masked_positions.sum(dim=-1).unique().numel() == 1), "Multiple masked positions found"
                num_tokens_generated_per_step.append(
                    (xs != self.mask_token_id).sum().item()
                )
                if getattr(generation_config, "compute_inf_budget", False):
                    alpha_t_schedule, _ = self.noise_schedule(t)
                    alpha_s_schedule, _ = self.noise_schedule(next_t)
                    alpha_t_prime = (alpha_s_schedule - alpha_t_schedule).abs()
                    inf_budget = ((xt == self.mask_token_id) & (alpha_t_prime != 0.0)).sum().item()
                    inf_budget_per_step.append(inf_budget)
                block_pbar.set_postfix(
                    NFEs=total_NFEs,
                    block_NFEs=block_NFEs,
                )
                if (xt == self.mask_token_id).sum().item() == 0:
                    if getattr(generation_config, "compute_inf_budget", False):
                        # for avearage inf budget calculation across all t
                        remaining_steps = timesteps.shape[0] - len(inf_budget_per_step)
                        inf_budget_per_step.extend([0] * remaining_steps)
                    break
                check_stopping_criteria = (i % window_size == 0) and (i > 0)
                if check_stopping_criteria and (not getattr(generation_config, "compute_inf_budget", False)) and stopping_criteria is not None:
                    is_done = stopping_criteria(
                        input_ids=accumulated_samples[  # type: ignore
                            :,
                            : window_start[0] + window_size
                        ],
                        scores=None,  # type: ignore
                    )
                    if torch.any(is_done):
                        accumulated_samples = accumulated_samples[
                            :,
                            : window_start[0] + window_size,
                        ]
                        accumulated_samples = accumulated_samples[accumulated_samples != self.mask_token_id].unsqueeze(0)
                        break
            if tokenizer is not None:
                print(tokenizer.batch_decode(accumulated_samples))
            if torch.any(is_done):
                break
        if pad_length is not None:
            accumulated_samples = accumulated_samples[:, : -pad_length]
        parallelism_factor = sum(num_tokens_generated_per_step) / len(
            num_tokens_generated_per_step
        )
        inf_budget = None
        if getattr(generation_config, "compute_inf_budget", False):
            inf_budget = sum(inf_budget_per_step) / len(inf_budget_per_step)
        if eval_ground_truth is not None:
            assert torch.equal(accumulated_samples, eval_ground_truth), "Generated tokens do not match eval ground truth"
        if return_dict_in_generate:
            return DiffusionGenerationOutput(
                sequences=accumulated_samples,
                parallelism_factor=parallelism_factor,
                inf_budget=inf_budget,
                inf_budgets=inf_budget_per_step,
                likelihoods=likelihoods,
            )
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