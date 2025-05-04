from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import ModelOutput

from denoiser import DenoiserInput


@dataclass
class SamplerConfig(OrderedDict):
    num_samples: int = 1
    batch_size: int = 1
    max_length: int = 512
    block_size: int = 512
    top_p: float = 0.9
    pad_context: bool = False
    first_hitting: bool = False
    low_confidence_remasking: bool = False
    disable_cache: bool = False
    kv_caching: bool = False
    shift_logits: bool = False


class Sampler(ABC):
    def __init__(self, config):
        self.config = config

    def __call__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        max_seq_len: int,
        num_steps: int,
        eps: float = 1e-5,
        device: str | None = None,
        prompt_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass

    def _sample_timesteps(
        self, eps: float, num_steps: int, device: str | None = None, **kwargs: Any
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
        return torch.linspace(1, eps, num_steps + 1, device=device)

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
        return samples

    def _logit_transform(self, logits: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Transform logits using various techniques.
        Args:
            logits (torch.Tensor): Logits to transform.
        Returns:
            torch.Tensor: Transformed logits.
        """
        if self.config.top_p < 1.0:
            logits = self._nucleus_sample(logits)
        return logits

    def _nucleus_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from the logits using nucleus sampling.
        Args:
            logits (torch.Tensor): Logits to sample from.
        Returns:
            torch.Tensor: Sampled tokens.
        """
        p = self.config.top_p
        if p == 1.0:
            return logits
        sorted_probs, sorted_indices = logits.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cum_probs <= p
        nucleus_mask[..., 0] = 1
        sorted_probs = sorted_probs * nucleus_mask
        logits.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
        logits /= logits.sum(-1, keepdim=True)
        return logits

    @abstractmethod
    def _sample_prior(
        self,
        model: torch.nn.Module,
        batch_size: int,
        max_seq_len: int,
        device: str | None = None,
    ) -> torch.Tensor:
        """
        Sample a sequence of tokens from the prior distribution.
        Args:
            batch_size (int): Number of sequences to sample.
            max_seq_len (int): Maximum sequence length.
            device (str | None): Device to use for sampling.
        Returns:
            torch.Tensor: Sampled tokens.
        """
        raise NotImplementedError("Sampler is not implemented yet.")

    def _check_stop_condition(
        self,
        xs: str,
        condition_prefix: str | None = None,
        condition_suffix: str | None = None,
    ) -> bool:
        """
        Check if the stop condition is met.
        Args:
            xs (torch.Tensor): Sampled tokens.
            condition_prefix (str | None): Prefix for the stop condition.
            condition_suffix (str | None): Suffix for the stop condition.
        Returns:
            bool: True if the stop condition is met, False otherwise.
        """
        for i in range(len(xs)):
            # check if the prefix and suffix are in the sampled tokens
            # ensure that prefix precedes suffix
            if condition_prefix is not None and condition_prefix not in xs[i]:
                return False
            if condition_suffix is not None and condition_suffix not in xs[i]:
                return False
            if (
                condition_prefix is not None
                and condition_suffix is not None
                and xs[i].index(condition_prefix) >= xs[i].index(condition_suffix)
            ):
                return False
        return True


# TODO
# class ARSampler(Sampler):


class AncestralSampler(Sampler):
    def __init__(self, config):
        super().__init__(config)

    def _sample_prior(
        self,
        model: torch.nn.Module,
        batch_size: int,
        max_seq_len: int,
        device: str | None = None,
    ) -> torch.Tensor:
        """
        Sample a sequence of tokens from the prior distribution.
        Args:
            batch_size (int): Number of sequences to sample.
            max_seq_len (int): Maximum sequence length.
            device (str | None): Device to use for sampling.
        Returns:
            torch.Tensor: Sampled tokens.
        """
        x = torch.full(
            (batch_size, max_seq_len),
            model.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        return x

    def _compute_posterior(
        self,
        model: torch.nn.Module,
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
        if self.config.first_hitting:
            return x

        if model.diffusion_type == "absorbing":
            q_xs = x * (alpha_s - alpha_t)
            q_xs[..., model.mask_token_id] = 1 - alpha_s[..., 0]
            q_xs /= 1 - alpha_t
            return q_xs

        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = torch.nn.functional.one_hot(x, model.vocab_size)
        limiting_distribution = torch.ones_like(xt_one_hot) / model.vocab_size
        if model.diffusion_type == "uniform":
            return (
                alpha_t * model.vocab_size * x * xt_one_hot
                + (alpha_ts - alpha_t) * xt_one_hot
                + d_alpha * x
                + (1 - alpha_ts) * (1 - alpha_s) * limiting_distribution
            ) / (
                alpha_t * model.vocab_size * torch.gather(x, -1, xt[..., None])
                + (1 - alpha_t)
            )
        raise NotImplementedError(
            f"Diffusion type {model.diffusion_type} not implemented."
        )

    def _generate_unconditional(  # TODO add CBG and CFG generation
        self,
        model: torch.nn.Module,
        alpha_t: torch.Tensor,
        alpha_s: torch.Tensor,
        denoiser_inputs: DenoiserInput | None = None,
        cache: Dict[str, torch.Tensor] | None = None,
        past_key_values: DynamicCache | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if cache is None:
            # pad context with masks to match the training context (improves mdlm quality)
            pad_len = 0
            if (
                self.config.pad_context
                and denoiser_inputs.xt.shape[-1] < model.config.length
            ):
                # pad with masks
                pad_len = model.config.length - denoiser_inputs.xt.shape[-1]
                pad = torch.full(
                    (denoiser_inputs.xt.shape[0], pad_len),
                    model.mask_token_id,
                    dtype=denoiser_inputs.xt.dtype,
                    device=denoiser_inputs.xt.device,
                )
                denoiser_inputs.xt = torch.cat((denoiser_inputs.xt, pad), dim=-1)
                denoiser_inputs.position_ids = torch.arange(
                    denoiser_inputs.xt.shape[-1], device=denoiser_inputs.xt.device
                )[None, :]
            backbone_output = model._backbone_forward(
                denoiser_inputs, past_key_values=past_key_values
            )
            # remove padding
            if pad_len > 0:
                backbone_output = backbone_output[:, :-pad_len]
                denoiser_inputs.xt = denoiser_inputs.xt[:, :-pad_len]
                denoiser_inputs.position_ids = torch.arange(
                    denoiser_inputs.xt.shape[-1], device=denoiser_inputs.xt.device
                )[None, :]
            if isinstance(backbone_output, ModelOutput) and hasattr(
                backbone_output, "logits"
            ):
                backbone_output = backbone_output.logits
            log_x_theta = model._forward_inference(
                backbone_output,
                denoiser_inputs,
            )  # should be the log(x_\theta) with the shape of (B, Seq, Vocab)
            x_theta = log_x_theta.exp()
            x_theta = self._logit_transform(x_theta, **kwargs)
        else:
            x_theta = cache["x_theta"]
        q_xs = self._compute_posterior(
            model, x_theta, denoiser_inputs.xt, alpha_t, alpha_s
        )
        cache = {"x_theta": x_theta}
        return q_xs, cache

    def _maybe_remask(
        self,
        xs: torch.Tensor,
        q_xs: torch.Tensor,
        xt: torch.Tensor,
        mask_token_id: int,
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
        if self.config.first_hitting:
            xs = self._first_hitting_remask(xs, q_xs, xt, mask_token_id)
        return xs

    def _first_hitting_remask(
        self,
        xs: torch.Tensor,
        q_xs: torch.Tensor,
        xt: torch.Tensor,
        mask_token_id: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        First-hitting sampler that analytically computes the next unmasking timestep
        Args:
            xs (torch.Tensor): Sampled sequence
            xt (torch.Tensor): Masked sequence
            mask_token_id (int): Mask token ID
        Returns:
            torch.Tensor: Samples adjusted for remasking.
        """
        # TODO assumes batch size 1

        # uniformly select an index (among masked tokens)
        num_masked = (xt == mask_token_id).sum(-1)

        if self.config.low_confidence_remasking:
            # select the index with the highest confidence
            xs_q = q_xs.gather(-1, xs[..., None]).squeeze(-1)
            xs_q[xt != mask_token_id] = 0
            ind = xs_q.argmax(dim=-1)
        else:
            ind = torch.randint(0, num_masked.item(), (xs.shape[0],))
            ind = (xt == mask_token_id).nonzero()[ind, 1]

        unmask_flag = torch.arange(xt.shape[-1], device=xt.device) == ind[None, :]
        # if a token is already unmasked, don't apply remasking
        unmask_flag = unmask_flag | (xt != mask_token_id)

        # remask tokens not selected
        xs[~unmask_flag] = mask_token_id
        return xs

    # def _cache_kvs(self, model: torch.nn.Module, xt: torch.Tensor):

    def _sample_timesteps(
        self,
        eps: float,
        num_steps: int,
        batch_size: int,
        max_seq_len: int,
        device: str | None = None,
    ) -> torch.Tensor:
        """
        Sample timesteps for the diffusion process.
        Returns:
            torch.Tensor: Sampled timesteps.
        """
        if self.config.first_hitting:
            # TODO: assumes batch size 1
            timesteps = torch.tensor([1.0])
            for i in range(max_seq_len, 0, -1):
                u = torch.rand(batch_size)
                next_t = timesteps[-1] * u ** (1 / i)
                timesteps = torch.cat((timesteps, next_t), dim=0)
            return timesteps[1:].to(device)
        return super()._sample_timesteps(eps, num_steps, device=device)

    def _sampling_loop(
        self,
        model: torch.nn.Module,
        batch_size: int,
        max_seq_len: int,
        num_steps: int,
        eps: float = 1e-5,
        device: str | None = None,
        disable_cache: bool = False,
        context: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        xt = self._sample_prior(model, batch_size, max_seq_len, device=device)
        timesteps = self._sample_timesteps(
            eps,
            num_steps,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=device,
        )
        dt = (1 - eps) / len(timesteps)
        pbar = tqdm(range(len(timesteps)), desc="Sampling", leave=False)
        NFEs = 0
        cache, context_mask = None, None

        # for unconditional generation, always start with bos
        if context is None or context.shape[1] == 0:
            context = torch.full(
                (batch_size, 1),
                model.bos_token_id,
                dtype=torch.long,
                device=device,
            )
        full_seq_len = context.shape[1] + max_seq_len
        # indicates which logits are used for sampling
        context_mask = torch.zeros((batch_size, full_seq_len), device=device)
        context_mask[:, : context.shape[1]] = 1
        if self.config.shift_logits:
            context_mask = context_mask[:, 1:]

        for i in pbar:
            t = timesteps[i]
            if cache is None:
                NFEs += 1
            # t is 0-dim tensor, reshape to (1, 1, 1) for broadcasting
            alpha_t, _ = model.noise_schedule(t)
            alpha_s, _ = model.noise_schedule(t - dt)
            alpha_t = alpha_t[None, None, None]
            alpha_s = alpha_s[None, None, None]
            # pass in context and xt to model
            input_ids = xt
            if context is not None:
                input_ids = torch.cat((context, input_ids), dim=1)
            if self.config.shift_logits:
                # left-shift the input_ids
                input_ids = input_ids[:, :-1]
            denoiser_inputs = model._prepare_inputs_inference(
                input_ids=input_ids,
                context_mask=context_mask,
                t=t,
            )
            q_xs, cache = self._generate_unconditional(
                model=model,
                alpha_t=alpha_t,
                alpha_s=alpha_s,
                denoiser_inputs=denoiser_inputs,
                cache=cache,
                xt=xt,
                mask_token_id=model.mask_token_id,
                past_key_values=past_key_values,
            )
            xs = self._sample_categorical(q_xs)
            xs = self._maybe_remask(xs, q_xs, xt, model.mask_token_id)
            if self.config.shift_logits:
                # apply carry-over for last token
                # (last token predicts a token not in the context)
                unmasked_last = xt[:, -1] != model.mask_token_id
                xs[:, -1][unmasked_last] = xt[:, -1][unmasked_last]
            pbar.set_postfix(
                NFEs=NFEs,
                prob_check=(q_xs.sum() / xt.numel()).item(),
                nan_check=bool(q_xs.isnan().sum() > 0),
            )
            if not torch.allclose(xs, xt) or not disable_cache:
                cache = None
            xt = xs
        # TODO: for e2d2, call encoder to cache kvs
        # if self.config.kv_caching:
        #     self._cache_kvs(model, xt)
        return xt, NFEs

    def __call__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        max_seq_len: int,
        num_steps: int,
        eps: float = 1e-5,
        device: str | None = None,
        disable_cache: bool = False,
        context: torch.Tensor | None = None,
        block_size: int | None = None,
        condition_prefix: str | None = "\\boxed{",
        condition_suffix: str | None = "}",
        **kwargs: Any,
    ) -> torch.Tensor:
        device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        block_size = max_seq_len if block_size is None else block_size
        max_blocks = max_seq_len // block_size
        if context is not None:
            accumulated_samples = context.to(device)
        else:
            accumulated_samples = torch.empty(
                (batch_size, 0), dtype=torch.int64, device=device
            )
        total_NFEs = 0
        past_key_values = None
        # cache context
        if context is not None and self.config.kv_caching:
            encoder_inputs = model._prepare_inputs_inference(
                input_ids=context,
                context_mask=torch.ones_like(context),
                t=torch.tensor([0.0], device=device),
            )
            _, past_key_values = model._backbone_forward(
                encoder_inputs,
                use_cache=True,
            )

        pbar = tqdm(range(max_blocks), desc="Sampling blocks", leave=False)
        for _ in pbar:
            # pass in context somehow
            sampled_block, block_NFEs = self._sampling_loop(
                model=model,
                batch_size=batch_size,
                max_seq_len=block_size,
                eps=eps,
                device=device,
                disable_cache=disable_cache,
                num_steps=num_steps,
                context=accumulated_samples,
                past_key_values=past_key_values,
            )

            accumulated_samples = torch.cat(
                [accumulated_samples, sampled_block], dim=-1
            )
            total_NFEs += block_NFEs
            print(model.tokenizer.batch_decode(accumulated_samples))
            if self._check_stop_condition(
                model.tokenizer.batch_decode(sampled_block),
                condition_prefix=condition_prefix,
                condition_suffix=condition_suffix,
            ):
                break

        # extra forward pass on clean block to cache kvs
        if self.config.kv_caching:
            denoiser_inputs = model._prepare_inputs_inference(input_ids=sampled_block)
            _, past_key_values = model._backbone_forward(
                denoiser_inputs, use_cache=True, past_key_values=past_key_values
            )

        # TODO after finishing this, set up notebook for testing on gsm8k
        return accumulated_samples, total_NFEs
