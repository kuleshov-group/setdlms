"""Generation configs and outputs for diffusion denoisers."""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from transformers import GenerationConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput


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
        first_hitting: bool = True,
        sampling_strategy: Literal["posterior", "predict_and_noise"] = "posterior",
        noise_removal: bool = True,
        confidence_based_noising: bool = False,
        confidence_margin_based_noising: bool = False,
        confidence_threshold: float = 1e6,
        align_inputs_to_blocks: bool = True,
        nucleus_p: float = 1.0,
        fused_block_cache: Optional[bool] = None,
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
                Defaults to True (must match constructor default).
            sampling_strategy (str): Method for transitioning between latents.
                Options:
                    - "posterior" - Compute and sample from the posterior
                        q(x_s | x_t, x_theta).
                    - "predict_and_noise" - Sample from the denoising model x_theta,
                        then add back noise to produce x_s.
                        Only implemented for absorbing diffusion.
                Defaults to "posterior".
            noise_removal (bool): Whether to run a final denoiser-only cleanup step.
                Defaults to True.
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
            nucleus_p (float): Nucleus sampling probability.
                Defaults to 1.0.
            fused_block_cache (bool): Whether cached blockwise generation should
                fuse the previous-block cache update into the first denoising step
                for the next block. Defaults to None, which enables the fused path
                for exact BD3LM models and leaves other model families unchanged.
            kwargs: Keyword arguments passed to `GenerationConfig`.
        """
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.min_t = min_t
        self.block_size = block_size
        self.first_hitting = first_hitting
        if self.first_hitting:
            self.num_steps = min(num_steps, self.block_size)
        self.sampling_strategy = sampling_strategy
        self.noise_removal = noise_removal
        assert not confidence_based_noising or not confidence_margin_based_noising, (
            "Cannot use both `confidence_based_noising` and"
            " `confidence_margin_based_noising`."
        )
        self.confidence_based_noising = confidence_based_noising
        self.confidence_margin_based_noising = confidence_margin_based_noising
        self.confidence_threshold = confidence_threshold
        self.align_inputs_to_blocks = align_inputs_to_blocks
        self.nucleus_p = nucleus_p
        self.fused_block_cache = fused_block_cache


class SetDiffusionGenerationConfig(DiffusionGenerationConfig):
    def __init__(
        self,
        max_window_size: int = 0,
        kv_cache: bool = True,
        cache_full_infill_context: Optional[bool] = None,
        ar_caching: Optional[bool] = None,
        use_first_hitting_order_in_decode: bool = False,
        profile_throughput: bool = False,
        infill_repetition_penalty_include_right_context: bool = False,
        setdlm_infill_cache_promotion_order: str = "l2r",
        **kwargs,
    ):
        """Generation config with additional parameters for set diffusion sampling.

        Args:
            max_window_size (int): Maximum window size to use for set diffusion.
                Defaults to 0.
            kv_cache (bool): Whether to use KV caching during EsoLM sampling.
                Defaults to True, which matches upstream `config.sampling.kv_cache`.
            cache_full_infill_context (bool): Whether infilling should cache both
                left and right non-mask context before generation. Defaults to the
                AnyOrderBD3LM-compatible behavior.
            ar_caching (bool): Legacy alias for `cache_full_infill_context`.
            use_first_hitting_order_in_decode (bool): Whether non-infill SetDLM
                decoding should order masked tokens by first-hitting times.
            profile_throughput (bool): Whether to skip token sampling and only
                benchmark backbone NFEs, mirroring upstream throughput profiling.
            infill_repetition_penalty_include_right_context (bool): Whether
                infilling repetition penalty should also see fixed right-context
                tokens after the masked span.
            setdlm_infill_cache_promotion_order (str): Explicit SetDLM infill
                KV-cache promotion order. `l2r` promotes generated tokens by
                left-to-right position order, and `first_hitting` promotes them
                by noise-schedule first-hitting order.
        """
        super().__init__(**kwargs)
        self.max_window_size = max_window_size
        self.kv_cache = kv_cache
        if cache_full_infill_context is None:
            cache_full_infill_context = (
                False if ar_caching is None else bool(ar_caching)
            )
        self.cache_full_infill_context = cache_full_infill_context
        self.ar_caching = cache_full_infill_context
        self.use_first_hitting_order_in_decode = use_first_hitting_order_in_decode
        self.profile_throughput = profile_throughput
        self.infill_repetition_penalty_include_right_context = (
            infill_repetition_penalty_include_right_context
        )
        valid_cache_promotion_orders = {"l2r", "first_hitting"}
        if setdlm_infill_cache_promotion_order not in valid_cache_promotion_orders:
            raise ValueError(
                "setdlm_infill_cache_promotion_order must be one of "
                f"{sorted(valid_cache_promotion_orders)}, got "
                f"{setdlm_infill_cache_promotion_order!r}"
            )
        self.setdlm_infill_cache_promotion_order = (
            setdlm_infill_cache_promotion_order
        )


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
        parallelism_factor (float): The heuristic parallelism factor of the generation.
            Defaults to -1.0.
    """

    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None
    parallelism_factor: Optional[float] = None
