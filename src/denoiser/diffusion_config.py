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
        confidence_based_noising: bool = False,
        confidence_margin_based_noising: bool = False,
        confidence_threshold: float = 1e6,
        align_inputs_to_blocks: bool = True,
        compute_inf_budget: bool = False,
        nucleus_p: float = 1.0,
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
            compute_inf_budget (bool): Whether to compute the information budget.
                Defaults to False.
            nucleus_p (float): Nucleus sampling probability.
                Defaults to 1.0.
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
        assert not confidence_based_noising or not confidence_margin_based_noising, (
            "Cannot use both `confidence_based_noising` and"
            " `confidence_margin_based_noising`."
        )
        self.confidence_based_noising = confidence_based_noising
        self.confidence_margin_based_noising = confidence_margin_based_noising
        self.confidence_threshold = confidence_threshold
        self.align_inputs_to_blocks = align_inputs_to_blocks
        self.compute_inf_budget = compute_inf_budget
        self.nucleus_p = nucleus_p


class SetDiffusionGenerationConfig(DiffusionGenerationConfig):
    def __init__(
        self,
        max_window_size: int = 0,
        subcontext_len: int = 0,
        subcontext_shuffle: bool = False,
        kv_cache: bool = True,
        profile_throughput: bool = False,
        **kwargs,
    ):
        """Generation config with additional parameters for set diffusion sampling.

        Args:
            max_window_size (int): Maximum window size to use for set diffusion.
                Defaults to 0.
            subcontext_len (int): EsoLM subcontext count for blockwise decoding.
                Defaults to 0, which matches the upstream tokenwise path.
            subcontext_shuffle (bool): Whether to shuffle EsoLM subcontexts before
                decoding. Defaults to False.
            kv_cache (bool): Whether to use KV caching during EsoLM sampling.
                Defaults to True, which matches upstream `config.sampling.kv_cache`.
            profile_throughput (bool): Whether to skip token sampling and only
                benchmark backbone NFEs, mirroring upstream throughput profiling.
        """
        super().__init__(**kwargs)
        self.max_window_size = max_window_size
        self.subcontext_len = subcontext_len
        self.subcontext_shuffle = subcontext_shuffle
        self.kv_cache = kv_cache
        self.profile_throughput = profile_throughput


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
        non_ar_tokens_per_step (float): The average number of finalized tokens per
            scheduler step that come from the fully parallel, non-AR stage of decoding.
            Defaults to None.
    """

    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None
    parallelism_factor: Optional[float] = None
    non_ar_tokens_per_step: Optional[float] = None
    inf_budget: Optional[float] = None
    inf_budgets: Optional[list[float]] = None
