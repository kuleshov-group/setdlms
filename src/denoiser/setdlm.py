from functools import partial
from typing import Any, Dict, Optional

import torch
from transformers import LogitsProcessorList, PreTrainedTokenizer, StoppingCriteriaList
from transformers.cache_utils import Cache, DynamicCache

try:
    from torch.nn.attention.flex_attention import and_masks, create_block_mask
except ImportError:
    and_masks, create_block_mask = None, None

from src.denoiser.base import DenoiserInput, LossAndNllOutput
from src.denoiser.bd3lm import BD3LM, BD3LMConfig
from src.denoiser.diffusion_config import (
    DiffusionGenerationConfig,
    DiffusionGenerationOutput,
    SetDiffusionGenerationConfig,
    create_attn_mask,
)


class SetDLM(BD3LM):
    """Denoiser class for SetDLM models."""

    def __init__(
        self,
        config: BD3LMConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
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
        model_output = model_output[:, input_length : input_length * 2, ...]
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=denoiser_inputs.x0[:, :, None],  # .repeat(1, num_repetitions, 1)
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
        inefficient_eval = getattr(self.config, "inefficient_training", False)
        if not self.training and inefficient_eval and self.config.block_size > 1:
            coeff = denoiser_inputs.alpha_t_prime / (1 - denoiser_inputs.alpha_t)
            coeff = torch.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
            seq_len = denoiser_inputs.x0.shape[1]
            masked_indices = denoiser_inputs.xt[:, -seq_len:] == self.mask_token_id
            nlls = log_p_theta * coeff * masked_indices
        else:
            nlls = loss * denoiser_inputs.tokens_mask
            if getattr(self.config, "inefficient_training", False):
                nlls *= denoiser_inputs.alpha_t_prime != 0.0

        # Compute per-batch counts and losses to avoid division by zero
        count = denoiser_inputs.tokens_mask.sum(dim=-1)  # Per-batch counts
        batch_nll = nlls.sum(dim=-1)  # Per-batch losses

        # Avoid division by zero: if count is 0, set token_nll to 0
        token_nll = torch.where(
            count > 0, batch_nll / count, torch.zeros_like(batch_nll)
        ).mean()

        permutation_order = denoiser_inputs.backbone_kwargs.get("permutation_order")
        other_loss_terms = {}
        if permutation_order is not None:
            other_loss_terms["permutation_order"] = permutation_order
            other_loss_terms["attention_mask"] = denoiser_inputs.attention_mask
            other_loss_terms["log_p_theta"] = -log_p_theta * denoiser_inputs.tokens_mask
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
        evaluate_nll_flag = getattr(self.config, "inefficient_training", False)
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
                xt = torch.where(alpha_t_prime != 0.0, self.mask_token_id, xt)

        batch_size, seq_len = input_ids.shape
        num_repetitions = 2
        if permute_flag.any():
            with torch.no_grad():
                perm_indices = self.noise_schedule.sample_permutation_order(
                    t,
                    permute_flag,
                    (
                        self.config.block_size
                        if self.training
                        else self.config.eval_block_size
                    ),
                    masked_tokens=(
                        (xt == self.mask_token_id) if evaluate_nll_flag else None
                    ),
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
                        block_size=(
                            self.config.block_size
                            if self.training
                            else self.config.eval_block_size
                        ),
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
            raise ValueError("Unknown attention backend")

        xt = torch.gather(xt, dim=-1, index=perm_indices)
        input_ids = torch.gather(input_ids, dim=-1, index=perm_indices)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[
            None, :
        ].repeat(batch_size, 1)
        position_ids = torch.gather(position_ids, dim=-1, index=perm_indices)

        if evaluate_nll_flag:
            xt = xt.repeat(1, 2)
        else:
            xt = torch.cat((input_ids, xt), dim=-1)
        position_ids = position_ids.repeat(1, num_repetitions)

        # TODO
        if evaluate_nll_flag:
            assert self.config.attn_backend == "sdpa", (
                "eval_nll flag only supported for sdpa backend"
            )
            for i in range(batch_size):
                if self.mask_token_id not in xt[i]:
                    continue
                first_mask_token_idx = (
                    (xt[i] == self.mask_token_id).float().argmax().item()
                )
                masked_indices = torch.arange(
                    first_mask_token_idx + seq_len, seq_len * 2
                )

                # masked tokens may only attend to clean tokens and
                # previous masked tokens
                clean_indices = torch.arange(seq_len, first_mask_token_idx + seq_len)
                decoder_attention_mask[i][
                    masked_indices[:, None], clean_indices[None, :]
                ] = False
                decoder_attention_mask[i][
                    masked_indices[:, None], masked_indices[None, :] - seq_len
                ] = False
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
            input_ids=inputs,
            cache=cache,
            return_updated_cache=True,
            first_hitting_times=first_hitting_times,
            **backbone_kwargs,
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
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        device = input_ids.device
        seq_len = input_ids.shape[1]
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        cache_len = self._get_past_key_values_seq_length(past_key_values)
        full_seq_length = cache_len + seq_len
        # crop KV cache if we would exceed model context
        if full_seq_length > self.config.length:
            overflow = full_seq_length - self.config.length
            past_key_values = self._crop_kv_cache_left(past_key_values, overflow)
            cache_len -= overflow
            full_seq_length = cache_len + input_ids.shape[-1]
            assert cache_len > 0, "cache_len must be greater than 0"
            assert cache_len + seq_len <= self.config.length, (
                "full seq length must be less than or equal to length"
            )

        perm_indices = None
        if first_hitting_times is not None:
            # mask tokens at the end
            fhs = torch.where(
                input_ids == self.mask_token_id, -1e6, first_hitting_times
            )
            perm_indices = fhs.argsort(dim=-1, stable=True, descending=True)
        else:
            cache_flag = (input_ids != self.mask_token_id).float()
            perm_indices = cache_flag.argsort(dim=-1, stable=True, descending=True)
        if return_updated_cache:
            # create a view of attention mask to prevent rematerialization
            base = self.static_attention_mask[
                cache_len : cache_len + seq_len, : cache_len + seq_len
            ]

            position_ids = torch.gather(position_ids, dim=-1, index=perm_indices)
            input_ids = torch.gather(input_ids, dim=-1, index=perm_indices)

            edit = torch.zeros_like(base, dtype=torch.bool)
            # keep self-attention
            diag_r = torch.arange(seq_len, device=base.device)
            diag_c = base.size(1) - seq_len + diag_r
            edit[diag_r, diag_c] = True
            patched = torch.zeros_like(base)
            if (input_ids == self.mask_token_id).any():
                first_mask_token_idx = (
                    (input_ids == self.mask_token_id).float().argmax(dim=-1)[0]
                )
                num_masked_tokens = seq_len - first_mask_token_idx
                # values to write where edit=True
                patched[-num_masked_tokens:, -num_masked_tokens:] = True
            attention_mask = torch.where(patched, edit, base)

            attention_mask = self._preprocess_attention_mask(
                attention_mask[None, None, ...], dtype=torch.float
            )
        else:
            full_possible_len = self.static_attention_mask.shape[-1] // 2
            valid_attn_indices = torch.cat(
                (
                    torch.arange(cache_len, device=device),
                    torch.arange(cache_len, cache_len + seq_len, device=device)
                    + full_possible_len,
                )
            )
            base = self.static_attention_mask[
                cache_len : cache_len + seq_len,
                valid_attn_indices,
            ]

            row = torch.arange(seq_len, device=device)
            col = cache_len + row
            diag_mask = torch.zeros_like(base, dtype=torch.bool)
            diag_mask[row, col] = True
            attention_mask = torch.where(diag_mask, torch.ones_like(base), base)
            attention_mask = self._preprocess_attention_mask(
                attention_mask[None, None, ...],
                dtype=torch.float,
            )
        return (
            DenoiserInput(
                xt=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
                past_key_values=past_key_values,
                backbone_kwargs={
                    "position_ids": position_ids,
                    "permutation_order": perm_indices,
                    "return_updated_cache": return_updated_cache,
                }
                | backbone_kwargs,
            ),
            cache,
        )

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
        if is_infill_task:
            num_mask_tokens = (inputs == self.mask_token_id).sum()
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
        return masked_tensor

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.LongTensor | None = None,
        generation_config: SetDiffusionGenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        return_dict_in_generate: Optional[bool] = False,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer: PreTrainedTokenizer | None = None,
        **kwargs: Any,
    ) -> torch.LongTensor:
        # Setup sampling variables
        if generation_config is None:
            assert getattr(self, "generation_config", None) is not None, (
                "Generation config must be provided if not present in the model."
            )
            generation_config = self.generation_config
        assert generation_config.use_cache, (
            "Generation with SetDLM requires use_cache=True."
        )
        input_length = inputs.shape[-1] if inputs is not None else 0
        batch_size = inputs.shape[0] if inputs is not None else batch_size
        assert batch_size == 1, "only batch size 1 supported for inference"
        max_length, max_new_tokens = self._compute_sampling_lengths(
            generation_config=generation_config,
            input_length=input_length,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )
        is_infill_task = inputs is not None and self.mask_token_id in inputs
        block_size = generation_config.block_size
        assert block_size == self.config.length, "ao-bd3lm not supported yet"

        pad_length = None
        if is_infill_task:
            all_position_ids = torch.arange(input_length, device=device)[
                None, :
            ].repeat(batch_size, 1)
        else:
            all_position_ids = torch.arange(
                input_length + max_new_tokens, device=device
            )[None, :].repeat(batch_size, 1)

        if is_infill_task:
            num_mask_tokens = (inputs == self.mask_token_id).sum()
            first_mask_token_idx = (
                (inputs == self.mask_token_id).float().argmax(dim=-1)[0]
            )
            last_mask_token_idx = (inputs != self.mask_token_id).float()[
                :, first_mask_token_idx:
            ].argmax(dim=-1)[0] + first_mask_token_idx
            inputs_offset = 0
        else:
            num_mask_tokens = max_new_tokens
            first_mask_token_idx = input_length
            last_mask_token_idx = input_length + max_new_tokens
            inputs_offset = input_length
        accumulated_samples = self._sample_prior(
            inputs=inputs,
            batch_size=batch_size,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            is_infill_task=is_infill_task,
            device=device,
        )
        if accumulated_samples.shape[-1] > self.config.length:
            first_hitting_times = None
        else:
            first_hitting_times = self.noise_schedule.compute_first_hitting_times(
                batch_size=batch_size,
                length=(
                    accumulated_samples.shape[-1]
                    if not is_infill_task
                    else accumulated_samples[:, first_mask_token_idx:].shape[-1]
                ),
                device=device,
            )
        xt = accumulated_samples[:, inputs_offset:]
        timesteps = self._sample_generation_timesteps(
            generation_config,
            max_length=num_mask_tokens,
            device=device,
            first_hitting_times=first_hitting_times,
        )
        timesteps = timesteps[None, :].repeat(batch_size, 1)
        xt_position_ids = all_position_ids[:, inputs_offset:]
        masked_positions = xt == self.mask_token_id
        window_start = inputs_offset + masked_positions.float().argmax(dim=-1)[:, None]
        window_size = min(
            self.noise_schedule.compute_window_size(), generation_config.max_window_size
        )
        masked_positions = masked_positions & (
            xt_position_ids < (window_start + window_size)
        )
        if input_length > 0:
            # cache positions that could be decoded before the response
            # tokens are generated
            cache_flag = torch.zeros_like(accumulated_samples, dtype=torch.bool)
            cache_flag[:, :first_mask_token_idx] = True
            inputs_indices = torch.arange(first_mask_token_idx)
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
        is_done = torch.Tensor([False])
        num_tokens_generated_per_step = []
        inf_budget_per_step = []
        block_NFEs = 0
        clean_len = 0
        for i in range(timesteps.shape[-1]):
            block_NFEs += 1
            total_NFEs += 1
            t = timesteps[:, i].unsqueeze(1).repeat(1, num_mask_tokens)
            if generation_config.first_hitting:
                num_generated = sum(num_tokens_generated_per_step)
                next_t = (
                    timesteps[:, num_generated + 1]
                    if num_generated < timesteps.shape[-1]
                    else timesteps[:, -1] * 0
                )
            else:
                next_t = (
                    timesteps[:, i + 1]
                    if i < timesteps.shape[-1] - 1
                    else timesteps[:, -1] * 0
                )
            next_t = next_t.unsqueeze(1).repeat(1, num_mask_tokens)

            masked_positions_indices = masked_positions.nonzero(as_tuple=False)[
                :, -1
            ].view(batch_size, -1)
            masked_xt = torch.gather(xt, dim=-1, index=masked_positions_indices)
            masked_position_ids = torch.gather(
                xt_position_ids, dim=-1, index=masked_positions_indices
            )
            # Only decode masked tokens
            cache_len = cache["past_key_values"].get_seq_length()
            return_updated_cache = i > 0
            denoiser_inputs, cache = self._prepare_inputs_inference(
                input_ids=masked_xt,
                cache=cache,
                position_ids=masked_position_ids,
                return_updated_cache=return_updated_cache,
            )
            if not is_infill_task:
                running_generation = accumulated_samples[
                    :, first_mask_token_idx:last_mask_token_idx
                ]
            else:
                running_generation = accumulated_samples[cache_flag].unsqueeze(0)
            generation_output = self._generate_unconditional(
                generation_config=generation_config,
                t=t[0][0],
                next_t=next_t[0][0],
                denoiser_inputs=denoiser_inputs,
                cache=cache,
                xt=xt,
                running_generation=running_generation,
                inputs_offset=inputs_offset,
                logits_processor=logits_processor,
                return_updated_cache=return_updated_cache,
                cache_len=clean_len,
                sample_indices=masked_positions_indices + inputs_offset,
                window_size=window_size,
                block_size=num_mask_tokens,
                **kwargs,
            )
            xs, cache = generation_output
            # crop kv cache and sampling output
            if return_updated_cache:
                # only keep cache for the clean tokens
                cache["past_key_values"].crop(cache_len + clean_len)
                position_ids = denoiser_inputs.backbone_kwargs["position_ids"]
                # update accumulated_samples, xt, masked_positions
                cached_position_ids = position_ids[:, :clean_len]
                unmasked_position_ids = position_ids[:, clean_len:]
                clean_len = (
                    (
                        torch.gather(
                            xt, dim=-1, index=(unmasked_position_ids - inputs_offset)
                        )
                        != xs
                    )
                    .sum(dim=-1)
                    .min()
                )
                accumulated_samples.scatter_(1, unmasked_position_ids, xs)
                if is_infill_task:
                    cache_flag.scatter_(1, cached_position_ids, True)
                xt.scatter_(1, unmasked_position_ids - inputs_offset, xs)
            else:
                xt.scatter_(1, masked_position_ids - inputs_offset, xs)
                clean_len = (xs != self.mask_token_id).sum(dim=-1).min()
                accumulated_samples.scatter_(1, masked_position_ids, xs)

            # for infilling, cache tokens that would be decoded before the
            # response tokens are generated
            if is_infill_task and (accumulated_samples == self.mask_token_id).any():
                just_unmasked_idx = (masked_positions) & (xt != self.mask_token_id)
                first_masked_position_id = 0
                last_potential_position_id = accumulated_samples.shape[-1]
                first_hitting_times = self.noise_schedule.compute_first_hitting_times(
                    batch_size=batch_size,
                    length=(
                        accumulated_samples.shape[-1]
                        if not is_infill_task
                        else accumulated_samples[
                            :, first_masked_position_id:last_potential_position_id
                        ].shape[-1]
                    ),
                    device=device,
                )
                target_unmasking_time = first_hitting_times[
                    just_unmasked_idx[
                        :, first_masked_position_id:last_potential_position_id
                    ]
                ].min()
                new_cache_positions = (
                    (first_hitting_times >= target_unmasking_time)
                    & (
                        accumulated_samples[
                            :, first_masked_position_id:last_potential_position_id
                        ]
                        != self.mask_token_id
                    )
                    & ~cache_flag[
                        :, first_masked_position_id:last_potential_position_id
                    ]
                    & ~masked_positions[
                        :, first_masked_position_id:last_potential_position_id
                    ]
                )
                clean_len += new_cache_positions.sum()
                masked_positions[
                    :, first_masked_position_id:last_potential_position_id
                ] |= new_cache_positions
                cache_flag[:, first_masked_position_id:last_potential_position_id] |= (
                    new_cache_positions
                )
            if return_updated_cache:
                masked_positions.scatter_(1, cached_position_ids - inputs_offset, False)
            window_start = (
                (accumulated_samples == self.mask_token_id)
                .float()
                .argmax(dim=-1)[:, None]
            )
            masked_positions |= (xt == self.mask_token_id) & (
                xt_position_ids < (window_start + window_size)
            )
            num_tokens_generated_per_step.append(
                (xs != self.mask_token_id).sum().item()
            )
            if generation_config.compute_inf_budget:
                alpha_t_schedule, _ = self.noise_schedule(t)
                alpha_s_schedule, _ = self.noise_schedule(next_t)
                alpha_t_prime = (alpha_s_schedule - alpha_t_schedule).abs()
                inf_budget = (
                    ((xt == self.mask_token_id) & (alpha_t_prime != 0.0)).sum().item()
                )
                inf_budget_per_step.append(inf_budget)
            if (xt == self.mask_token_id).sum().item() == 0:
                if generation_config.compute_inf_budget:
                    # for inf budget calculation avg over all timesteps
                    remaining_steps = timesteps.shape[-1] - len(inf_budget_per_step)
                    inf_budget_per_step.extend([0] * remaining_steps)
                break
            check_stopping_criteria = (i % window_size == 0) and (i > 0)
            if (
                check_stopping_criteria
                and (not generation_config.compute_inf_budget)
                and stopping_criteria is not None
            ):
                is_done = stopping_criteria(
                    input_ids=accumulated_samples[  # type: ignore
                        :, : window_start[0] + window_size
                    ],
                    scores=None,  # type: ignore
                )
                if torch.any(is_done):
                    if not is_infill_task:
                        accumulated_samples = accumulated_samples[
                            :,
                            : window_start[0] + window_size,
                        ]
                    break
        if pad_length is not None:
            accumulated_samples = accumulated_samples[:, :-pad_length]
        parallelism_factor = sum(num_tokens_generated_per_step) / len(
            num_tokens_generated_per_step
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
