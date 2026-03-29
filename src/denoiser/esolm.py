from __future__ import annotations

import time
from typing import Any, Optional

import torch
from transformers import (
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache

from src.denoiser.base import DenoiserInput, DenoiserOutput
from src.denoiser.bd3lm import BD3LMConfig
from src.denoiser.diffusion_config import (
    DiffusionGenerationOutput,
    SetDiffusionGenerationConfig,
)
from src.denoiser.setdlm import SetDLM
from src.noise_schedule.noise_schedules import EsoLogLinearNoise


class EsoLMConfig(BD3LMConfig):
    """Configuration class for Eso-LMs models."""

    model_type = "esolm"
    auto_map = {
        "AutoConfig": "diffusion.EsoLMConfig",
        "AutoModel": "diffusion.EsoLM",
        "AutoModelForMaskedLM": "diffusion.EsoLM",
    }

    def __init__(
        self,
        alpha_0: float = 0.0,
        batch_split: float = 0.0,
        diffusion_shuffle: bool = False,
        diffusion_attn_mode: str = "bidirectional",
        sequential_shuffle: bool = False,
        sequential_attn_mode: str = "causal",
        loss_type: str = "elbo",
        num_iw_orders: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha_0 = alpha_0
        self.batch_split = batch_split
        self.diffusion_shuffle = diffusion_shuffle
        self.diffusion_attn_mode = diffusion_attn_mode
        self.sequential_shuffle = sequential_shuffle
        self.sequential_attn_mode = sequential_attn_mode
        self.loss_type = loss_type
        self.num_iw_orders = num_iw_orders


class EsoLM(SetDLM):
    """Eso-LMs denoiser adapted to this repository's denoiser/backbone split."""

    config_class = EsoLMConfig

    def __init__(
        self,
        config: EsoLMConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, **kwargs)
        noise_eps = 1e-3
        if getattr(self.config, "noise_config", None) is not None and hasattr(
            self.config.noise_config, "get"
        ):
            noise_eps = float(self.config.noise_config.get("eps", noise_eps))
        # Direct transcription of `s-sahoo/Eso-LMs/algo.py` lines 285-289.
        # Difference from upstream: we also mirror the instantiated schedule back
        # into `config.noise_config` so saved local configs describe the runtime
        # behavior instead of the global Hydra default.
        self.alpha_0 = float(self.config.alpha_0)
        self.num_tokens = int(self.config.length)
        self.noise_schedule = EsoLogLinearNoise(
            alpha_0=self.alpha_0,
            eps=noise_eps,
        )
        self.config.noise_config = {
            "_target_": "src.noise_schedule.noise_schedules.EsoLogLinearNoise",
            "alpha_0": self.alpha_0,
            "eps": noise_eps,
        }

    @staticmethod
    def _sigma_from_alpha_t(alpha_t: torch.FloatTensor) -> torch.FloatTensor:
        # Upstream source: `s-sahoo/Eso-LMs/trainer_base.py::_sigma_from_alphat`
        # (lines 451-452).
        return -torch.log(alpha_t)

    def _resolve_num_iw_orders(self, kwargs: dict[str, Any]) -> int:
        if "num_iw_orders" in kwargs:
            return int(kwargs.pop("num_iw_orders"))
        eval_config = getattr(self.config, "eval", None)
        if eval_config is not None and hasattr(eval_config, "num_iw_orders"):
            return int(getattr(eval_config, "num_iw_orders"))
        return int(getattr(self.config, "num_iw_orders", 0))

    def _create_static_mask(self) -> None:
        # Direct transcription of `s-sahoo/Eso-LMs/models/dit.py` lines 84-138
        # with `block_size=1`. Difference from upstream: we materialize a dense
        # boolean mask because this repository routes attention through standard
        # attention-mask tensors instead of flex-attention block masks.
        seq_len = self.config.length
        q_idx = torch.arange(seq_len * 2)[:, None]
        kv_idx = torch.arange(seq_len * 2)[None, :]
        mask = self._sequential_block_mask(q_idx=q_idx, kv_idx=kv_idx, seq_len=seq_len)
        self.register_buffer("static_attention_mask", mask)
        if "static_attention_mask" not in self.skip_params_for_push:
            self.skip_params_for_push.append("static_attention_mask")

    @staticmethod
    def _causal_mask(
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        # Eso-LMs source: `model.py::_causal_mask`.
        return q_idx >= kv_idx

    @staticmethod
    def _bidirectional_mask(
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        # Eso-LMs source: `model.py::_bidirectional_mask`.
        del kv_idx
        return q_idx == q_idx

    @staticmethod
    def _mixed_mask(
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        cutoffs: torch.Tensor,
    ) -> torch.Tensor:
        # Eso-LMs source: `model.py::_mixed_mask`.
        # Change from original: the official code emits a flex-attention block
        # mask. Here we materialize the same boolean relation for this repo's
        # standard attention-mask pipeline.
        causal = q_idx >= kv_idx
        block_identity = q_idx >= cutoffs[:, None, None]
        return causal | block_identity

    @staticmethod
    def _mixed2_mask(
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        cutoffs: torch.Tensor,
    ) -> torch.Tensor:
        # Eso-LMs source: `model.py::_mixed2_mask`.
        # Change from original: same as `_mixed_mask`; we build a dense mask
        # because this repo routes attention through `attention_mask`.
        causal = q_idx >= kv_idx
        block_identity = (q_idx < cutoffs[:, None, None]) & (
            kv_idx < cutoffs[:, None, None]
        )
        return causal | block_identity

    @staticmethod
    def _sequential_block_mask(
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        # Direct transcription of `s-sahoo/Eso-LMs/models/dit.py` lines 84-130
        # with `block_size=1`, which simplifies block indices to token indices.
        x0_flag_q = (q_idx >= seq_len).bool()
        x0_flag_kv = (kv_idx >= seq_len).bool()
        q_idx = q_idx % seq_len
        kv_idx = kv_idx % seq_len
        block_diagonal = (q_idx == kv_idx) & (x0_flag_q == x0_flag_kv)
        offset_block_causal = (q_idx > kv_idx) & x0_flag_kv & ~x0_flag_q
        block_causal = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q
        return block_diagonal | offset_block_causal | block_causal

    def _sequential_prefix_mask(
        self,
        seq_len: int,
        cutoffs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # Direct transcription of `s-sahoo/Eso-LMs/models/dit.py` lines 141-147.
        # Difference from upstream: we build the dense boolean relation directly
        # instead of returning a flex-attention block mask.
        q_idx = torch.arange(seq_len * 2, device=device)[None, :, None]
        kv_idx = torch.arange(seq_len * 2, device=device)[None, None, :]
        base_mask = self._sequential_block_mask(q_idx=q_idx, kv_idx=kv_idx, seq_len=seq_len)
        block_prefix_lm = (
            (seq_len <= q_idx)
            & (q_idx < seq_len + cutoffs[:, None, None])
            & (seq_len <= kv_idx)
            & (kv_idx < seq_len + cutoffs[:, None, None])
        )
        return base_mask | block_prefix_lm

    def _sort_indices(
        self,
        indices: torch.LongTensor,
        shuffle: bool,
        keep_masks_unshuffled: bool = False,
    ) -> torch.LongTensor:
        # Eso-LMs source: `model.py::EsoLMDiT._sort_indices` and
        # `algo.py::EsoLM._sort_indices`.
        # Change from original: the official code returns both sorted tokens and
        # the permutation because sorting happens inside the model wrapper. This
        # repo sorts tokens in the denoiser layer, so returning the permutation is
        # the cleaner integration point.
        masked = indices == self.mask_token_id
        if shuffle:
            offsets = torch.rand(indices.shape, device=indices.device) * 0.9
            if keep_masks_unshuffled and masked.any():
                num_masked = int(masked.sum().item())
                offsets = offsets.clone()
                offsets[masked] = torch.linspace(
                    0, 1, num_masked, device=indices.device
                )
        else:
            offsets = torch.linspace(
                0, 0.9, indices.shape[1], device=indices.device
            )[None, :].expand_as(indices)
        return (masked.to(offsets.dtype) + offsets).argsort(dim=-1, descending=False)

    def _sort_active_indices(
        self,
        indices: torch.LongTensor,
        active_mask: torch.BoolTensor,
        shuffle: bool,
        keep_masks_unshuffled: bool = False,
    ) -> torch.LongTensor:
        # GSM8K and other conditional tasks rely on context tokens remaining a
        # fixed prefix. Only the active target span should participate in the
        # EsoLM permutation; context and padding stay in place.
        if indices.shape != active_mask.shape:
            raise ValueError(
                "EsoLM active permutation expects `indices` and `active_mask` "
                f"to have the same shape, got {indices.shape} and {active_mask.shape}."
            )

        batch_size, seq_len = indices.shape
        base_positions = torch.arange(seq_len, device=indices.device)
        perm_indices = base_positions[None, :].expand(batch_size, -1).clone()
        for batch_idx in range(batch_size):
            active_positions = base_positions[active_mask[batch_idx]]
            if active_positions.numel() <= 1:
                continue
            local_perm = self._sort_indices(
                indices=indices[batch_idx : batch_idx + 1, active_positions],
                shuffle=shuffle,
                keep_masks_unshuffled=keep_masks_unshuffled,
            )[0]
            perm_indices[batch_idx, active_positions] = active_positions[local_perm]
        return perm_indices

    def _build_diffusion_attention_mask(
        self,
        attention_mask: torch.LongTensor,
        cutoffs: torch.LongTensor,
        attn_mode: str | None = None,
    ) -> torch.FloatTensor | None:
        # Eso-LMs source: `model.py::EsoLMDiT._get_attention_mask`.
        # Change from original: the official implementation caches flex-attention
        # block masks. This repo expects a standard 4D attention mask, so we
        # materialize the same relations explicitly.
        seq_len = attention_mask.shape[1]
        mode = attn_mode or self.config.diffusion_attn_mode
        if attention_mask.bool().all() and mode in {
            "causal",
            "bidirectional",
        }:
            if mode == "causal":
                mask = self._causal_mask(
                    torch.arange(seq_len, device=attention_mask.device)[None, :, None],
                    torch.arange(seq_len, device=attention_mask.device)[
                        None, None, :
                    ],
                )
            else:
                mask = self._bidirectional_mask(
                    torch.arange(seq_len, device=attention_mask.device)[None, :, None],
                    torch.arange(seq_len, device=attention_mask.device)[
                        None, None, :
                    ],
                )
            mask = mask.expand(attention_mask.shape[0], -1, -1)
            return self._preprocess_attention_mask(mask[:, None], dtype=torch.float)

        q_idx = torch.arange(seq_len, device=attention_mask.device)[None, :, None]
        kv_idx = torch.arange(seq_len, device=attention_mask.device)[None, None, :]
        if mode == "causal":
            mask = self._causal_mask(q_idx, kv_idx)
        elif mode == "bidirectional":
            mask = self._bidirectional_mask(q_idx, kv_idx)
        elif mode == "mixed":
            mask = self._mixed_mask(q_idx, kv_idx, cutoffs)
        elif mode == "mixed2":
            mask = self._mixed2_mask(q_idx, kv_idx, cutoffs)
        else:
            raise ValueError(f"Unsupported diffusion_attn_mode: {mode}")
        mask = (
            mask
            & attention_mask[:, None, :].bool()
            & attention_mask[:, :, None].bool()
        )
        return self._preprocess_attention_mask(mask[:, None], dtype=torch.float)

    def _build_sequential_attention_mask(
        self,
        attention_mask: torch.LongTensor,
        xt: torch.LongTensor,
        mask_token_id: int,
        attn_mode: str,
    ) -> torch.FloatTensor:
        seq_len = xt.shape[1] // 2
        q_idx = torch.arange(seq_len * 2, device=xt.device)[None, :, None]
        kv_idx = torch.arange(seq_len * 2, device=xt.device)[None, None, :]
        if attn_mode == "causal":
            mask = self._sequential_block_mask(q_idx=q_idx, kv_idx=kv_idx, seq_len=seq_len)
        elif attn_mode == "mixed":
            cutoffs = torch.sum(xt[:, :seq_len] != mask_token_id, dim=1)
            mask = self._sequential_prefix_mask(
                seq_len=seq_len,
                cutoffs=cutoffs,
                device=xt.device,
            )
        else:
            raise ValueError(f"Unsupported sequential_attn_mode: {attn_mode}")
        mask = (
            mask
            & attention_mask[:, None, :].bool()
            & attention_mask[:, :, None].bool()
        )
        return self._preprocess_attention_mask(mask[:, None], dtype=torch.float)

    @staticmethod
    def _has_native_esolm_path(*modules: Any) -> bool:
        return any(
            callable(getattr(module, "_forward_esolm", None))
            for module in modules
            if module is not None
        )

    def _build_hf_esolm_backbone_kwargs(
        self,
        denoiser_inputs: DenoiserInput,
        merged_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], int | None]:
        sort_index = merged_kwargs.pop("sort_index", None)
        if sort_index is None:
            raise ValueError("EsoLM HF backbone path requires `sort_index`.")
        sequential_input = bool(merged_kwargs.pop("sequential_input", False))
        diffusion_attn_mode = str(
            merged_kwargs.pop(
                "diffusion_attn_mode",
                self.config.diffusion_attn_mode,
            )
        )
        sequential_attn_mode = str(
            merged_kwargs.pop(
                "sequential_attn_mode",
                self.config.sequential_attn_mode,
            )
        )
        mask_token_id = int(merged_kwargs.pop("mask_token_id", self.mask_token_id))
        # HF backbones do not implement the EsoLM sigma-conditioning path.
        merged_kwargs.pop("sigma", None)

        if sequential_input:
            seq_len = denoiser_inputs.xt.shape[1] // 2
            position_ids = torch.cat([sort_index, sort_index], dim=1)
            attention_mask = self._build_sequential_attention_mask(
                attention_mask=denoiser_inputs.attention_mask,
                xt=denoiser_inputs.xt,
                mask_token_id=mask_token_id,
                attn_mode=sequential_attn_mode,
            )
            output_length = seq_len
        else:
            position_ids = sort_index
            cutoffs = torch.sum(denoiser_inputs.xt != mask_token_id, dim=1)
            attention_mask = self._build_diffusion_attention_mask(
                attention_mask=denoiser_inputs.attention_mask,
                cutoffs=cutoffs,
                attn_mode=diffusion_attn_mode,
            )
            output_length = None

        # The custom HF/Qwen path expects the same dense 4D additive mask used by
        # the native EsoLM backbone. Passing the old dict wrapper here leaves the
        # mask uninterpreted by the model stack, which breaks upstream-equivalent
        # per-example ordering behavior.
        merged_kwargs["position_ids"] = position_ids
        merged_kwargs["attention_mask"] = attention_mask
        return merged_kwargs, output_length

    def _prepare_diffusion_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        context_mask: torch.LongTensor,
        t: torch.FloatTensor | None,
    ) -> DenoiserInput:
        # Upstream sources:
        # - `s-sahoo/Eso-LMs/algo.py::EsoLM.nll` (lines 441-474)
        # - `s-sahoo/Eso-LMs/trainer_base.py::_sigma_from_alphat` (lines 451-452)
        # Deviation from upstream: the local denoiser still owns padding/context
        # bookkeeping, but the sorted rotary features and attention masks now live
        # in the backbone just like the official Eso-LMs implementation.
        if t is None:
            t = torch.rand(input_ids.shape[0], device=input_ids.device)
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        alpha_t = alpha_t.reshape(input_ids.shape[0], -1)
        alpha_t_prime = alpha_t_prime.reshape(input_ids.shape[0], -1)
        if alpha_t.shape[1] != 1 or alpha_t_prime.shape[1] != 1:
            raise ValueError(
                "EsoLM diffusion expects per-example alpha_t and alpha_t_prime "
                "from the noise schedule."
            )
        sigma = self._sigma_from_alpha_t(alpha_t.squeeze(-1))
        noise_mask = context_mask | ~attention_mask.bool()
        xt = self._sample_q_xt(x0=input_ids, alpha_t=alpha_t, mask=noise_mask)
        perm_indices = self._sort_active_indices(
            indices=xt,
            active_mask=(~noise_mask.bool()) & attention_mask.bool(),
            shuffle=self.config.diffusion_shuffle,
            keep_masks_unshuffled=False,
        )
        xt = torch.gather(xt, dim=-1, index=perm_indices)
        x0 = torch.gather(input_ids, dim=-1, index=perm_indices)
        attention_mask = torch.gather(attention_mask, dim=-1, index=perm_indices)
        context_mask = torch.gather(context_mask, dim=-1, index=perm_indices)
        valid_tokens = attention_mask * (1 - context_mask)
        tokens_mask = attention_mask * (1 - context_mask) * (xt == self.mask_token_id)
        return DenoiserInput(
            xt=xt,
            x0=x0,
            attention_mask=attention_mask,
            context_mask=context_mask,
            valid_tokens=valid_tokens,
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
            backbone_kwargs={
                "sigma": sigma,
                "sort_index": perm_indices,
                "diffusion_attn_mode": self.config.diffusion_attn_mode,
                "mask_token_id": self.mask_token_id,
            },
        )

    def _prepare_sequential_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        context_mask: torch.LongTensor,
    ) -> DenoiserInput:
        # Upstream sources:
        # - `s-sahoo/Eso-LMs/algo.py::EsoLM._reconstruction_loss` (lines 419-439)
        # - `s-sahoo/Eso-LMs/models/dit.py::EsoLMDiT._sequential_features`
        #   (lines 922-947)
        # Sequential conditioning is explicitly zero here, matching upstream's
        # `dummy_t0 = torch.zeros(...)` rather than relying on backbone defaults.
        alpha_0 = torch.full_like(
            input_ids, fill_value=float(self.config.alpha_0), dtype=torch.float
        )
        noise_mask = context_mask | ~attention_mask.bool()
        z0 = self._sample_q_xt(x0=input_ids, alpha_t=alpha_0, mask=noise_mask)
        perm_indices = self._sort_active_indices(
            indices=z0,
            active_mask=(~noise_mask.bool()) & attention_mask.bool(),
            shuffle=self.config.sequential_shuffle,
            keep_masks_unshuffled=True,
        )
        z0 = torch.gather(z0, dim=-1, index=perm_indices)
        x0 = torch.gather(input_ids, dim=-1, index=perm_indices)
        attention_mask = torch.gather(attention_mask, dim=-1, index=perm_indices)
        context_mask = torch.gather(context_mask, dim=-1, index=perm_indices)
        valid_tokens = attention_mask * (1 - context_mask)
        masked_tokens = z0 == self.mask_token_id
        sequential_xt = torch.cat([z0, x0], dim=1)
        sequential_attention_mask = torch.cat([attention_mask, attention_mask], dim=1)
        return DenoiserInput(
            xt=sequential_xt,
            x0=x0,
            attention_mask=sequential_attention_mask,
            context_mask=context_mask,
            valid_tokens=valid_tokens,
            tokens_mask=valid_tokens * masked_tokens,
            t=torch.zeros(x0.shape[0], device=x0.device),
            alpha_t=alpha_0,
            alpha_t_prime=torch.zeros_like(alpha_0),
            backbone_kwargs={
                "sigma": torch.zeros(x0.shape[0], device=x0.device, dtype=torch.float32),
                "sort_index": perm_indices,
                "sequential_input": True,
                "sequential_attn_mode": self.config.sequential_attn_mode,
                "mask_token_id": self.mask_token_id,
            },
        )

    @staticmethod
    def _reduce_branch_loss(
        nlls: torch.FloatTensor,
        valid_tokens: torch.FloatTensor,
    ) -> torch.FloatTensor:
        total_nll = nlls.sum()
        total_valid = valid_tokens.sum()
        return torch.where(
            total_valid > 0,
            total_nll / total_valid,
            torch.zeros((), device=nlls.device, dtype=nlls.dtype),
        )

    def _compute_diffusion_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # Upstream source:
        # - `s-sahoo/Eso-LMs/algo.py` lines 141-150 (`MDLM.nll_per_token`)
        # - `s-sahoo/Eso-LMs/algo.py` lines 391-399 and 468-474 (`EsoLM.nll/_loss`)
        # Differences from upstream:
        # - `valid_tokens` is carried inside `DenoiserInput`, so we reduce here.
        # - We branch on `self.training` to match upstream `train_mode`.
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=denoiser_inputs.x0[:, :, None],
        ).squeeze(-1)
        if self.training and self.config.loss_type == "low_var":
            nlls = -log_p_theta * denoiser_inputs.tokens_mask
        else:
            coeff = -(denoiser_inputs.alpha_t_prime / (1 - denoiser_inputs.alpha_t))
            coeff = torch.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
            nlls = -log_p_theta * denoiser_inputs.tokens_mask * coeff
        loss = self._reduce_branch_loss(nlls, denoiser_inputs.valid_tokens)
        return loss, nlls

    def _compute_sequential_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # Upstream source:
        # - `s-sahoo/Eso-LMs/algo.py` lines 419-439
        # - `s-sahoo/Eso-LMs/models/dit.py` lines 922-967
        # Difference from upstream: the backbone truncation now happens inside
        # `DIT.forward`, so this branch only applies the masked-token loss.
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=denoiser_inputs.x0[:, :, None],
        ).squeeze(-1)
        nlls = -log_p_theta * denoiser_inputs.tokens_mask
        loss = self._reduce_branch_loss(nlls, denoiser_inputs.valid_tokens)
        return loss, nlls

    def _tokens_unmasked_per_step(self, num_steps: int) -> list[int]:
        # Direct transcription of `s-sahoo/Eso-LMs/algo.py::EsoLM._tokens_unmasked_per_step`
        # (lines 496-510).
        # Deviation from upstream: the local denoiser stores sequence length in
        # `config.length`, so we use `self.num_tokens` as the faithful equivalent.
        # We also use `torch.binomial` instead of `np.random.binomial` to avoid
        # introducing a new runtime dependency into this repo.
        return self._tokens_unmasked_per_step_for_length(
            num_steps=num_steps,
            num_tokens=self.num_tokens,
        )

    def _tokens_unmasked_per_step_for_length(
        self,
        num_steps: int,
        num_tokens: int,
    ) -> list[int]:
        remaining_tokens = num_tokens
        num_tokens_to_unmask: list[int] = []
        dt = 1 / num_steps
        for t in torch.linspace(1.0, dt, steps=num_steps).tolist():
            alpha_t, _ = self.noise_schedule(
                torch.tensor(t, dtype=torch.float32)
            )
            alpha_s, _ = self.noise_schedule(
                torch.tensor(t - dt, dtype=torch.float32)
            )
            alpha_t_float = float(alpha_t)
            alpha_s_float = float(alpha_s)
            unmask_prob = (alpha_s_float - alpha_t_float) / (1 - alpha_t_float)
            n_unmask = int(
                torch.binomial(
                    torch.tensor(float(remaining_tokens)),
                    torch.tensor(float(unmask_prob)),
                ).item()
            )
            if n_unmask != 0:
                num_tokens_to_unmask.append(n_unmask)
                remaining_tokens -= n_unmask
        if remaining_tokens != 0 and self.alpha_0 == 1:
            num_tokens_to_unmask.append(remaining_tokens)
        return num_tokens_to_unmask

    @staticmethod
    def _reverse_indices(sort_idx: torch.LongTensor) -> torch.LongTensor:
        return sort_idx.argsort(dim=-1)

    def _esolm_generation_plan(
        self,
        num_samples: int,
        generation_config: SetDiffusionGenerationConfig,
        device: torch.device,
        num_tokens: int | None = None,
    ) -> tuple[list[int], torch.LongTensor]:
        # Direct transcription of `s-sahoo/Eso-LMs/algo.py::EsoLM.generate_samples`
        # (lines 522-559).
        # Deviation from upstream: we read options from the local
        # `SetDiffusionGenerationConfig` instead of Hydra's `config.sampling`.
        total_tokens = self.num_tokens if num_tokens is None else int(num_tokens)
        subcontext_len = int(getattr(generation_config, "subcontext_len", 0) or 0)
        if subcontext_len == 0:
            unmask_k_tokens = self._tokens_unmasked_per_step_for_length(
                num_steps=generation_config.num_steps,
                num_tokens=total_tokens,
            )
            num_diffusion_tokens = sum(unmask_k_tokens)
            sort_idx = torch.rand(num_samples, total_tokens, device=device).argsort(
                dim=-1, descending=False
            )
            sort_idx[:, num_diffusion_tokens:] = (
                sort_idx[:, num_diffusion_tokens:].sort(dim=-1).values
            )
        else:
            if total_tokens % subcontext_len != 0:
                raise ValueError(
                    "EsoLM subcontext sampling requires length divisible by "
                    f"subcontext_len, got {total_tokens} and {subcontext_len}."
                )
            block_size = total_tokens // subcontext_len
            sort_idx = torch.arange(total_tokens, device=device)
            sort_idx = sort_idx.view(block_size, subcontext_len).t()
            if getattr(generation_config, "subcontext_shuffle", False):
                n_rows, n_cols = sort_idx.shape
                row_perm = torch.stack(
                    [torch.randperm(n_cols, device=device) for _ in range(n_rows)]
                )
                sort_idx = torch.gather(sort_idx, 1, row_perm)
                sort_idx = sort_idx[torch.randperm(subcontext_len, device=device)]
            sort_idx = sort_idx.flatten()
            num_diffusion_tokens = int(total_tokens * self.alpha_0)
            unmask_k_tokens = [block_size] * int(subcontext_len * self.alpha_0)
            sort_idx[num_diffusion_tokens:] = (
                sort_idx[num_diffusion_tokens:].sort().values
            )
            sort_idx = sort_idx[None].expand(num_samples, -1)

        unmask_k_tokens = unmask_k_tokens + [1] * (
            total_tokens - num_diffusion_tokens
        )
        return unmask_k_tokens, sort_idx

    def _prepare_generation_inputs(
        self,
        num_samples: int,
        generation_config: SetDiffusionGenerationConfig,
        device: torch.device,
        prompt_input_ids: torch.LongTensor | None = None,
        target_length: int | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor, list[int], int]:
        prompt_length = 0
        if prompt_input_ids is not None:
            if prompt_input_ids.dim() != 2:
                raise ValueError(
                    "EsoLM prompt-conditioned generation expects rank-2 `inputs`."
                )
            if num_samples != prompt_input_ids.shape[0]:
                raise ValueError(
                    "EsoLM `num_samples` must match the prompt batch size when "
                    "`prompt_input_ids` is provided."
                )
            prompt_input_ids = prompt_input_ids.to(device)
            if (prompt_input_ids == self.mask_token_id).any():
                raise NotImplementedError(
                    "EsoLM sampling does not support infilling prompts; only "
                    "contiguous clean prefixes are supported."
                )
            prompt_length = int(prompt_input_ids.shape[1])
            if target_length is None:
                target_length = prompt_length + self.num_tokens
            if target_length < prompt_length:
                raise ValueError(
                    "EsoLM target generation length cannot be shorter than the prompt."
                )
            sampled_tokens = target_length - prompt_length
            if sampled_tokens > self.num_tokens:
                raise ValueError(
                    "EsoLM prompt-conditioned generation samples at most "
                    "`config.length` new tokens; received "
                    f"{sampled_tokens} > {self.num_tokens}."
                )
            if sampled_tokens == 0:
                return (
                    prompt_input_ids.clone(),
                    torch.arange(prompt_length, device=device)[None, :].expand(
                        num_samples, -1
                    ),
                    [],
                    prompt_length,
                )
            unmask_k_tokens, continuation_sort_idx = self._esolm_generation_plan(
                num_samples=num_samples,
                generation_config=generation_config,
                device=device,
                num_tokens=sampled_tokens,
            )
            prompt_sort_idx = torch.arange(prompt_length, device=device)[
                None, :
            ].expand(num_samples, -1)
            sort_idx = torch.cat(
                [prompt_sort_idx, prompt_length + continuation_sort_idx],
                dim=1,
            )
            x = torch.full(
                (num_samples, target_length),
                fill_value=self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
            x[:, :prompt_length] = prompt_input_ids
            x = torch.gather(x, dim=1, index=sort_idx)
            return x, sort_idx, unmask_k_tokens, prompt_length

        target_length = self.num_tokens if target_length is None else int(target_length)
        if target_length != self.num_tokens:
            raise ValueError(
                "EsoLM unconditional sampling expects `target_length == config.length`; "
                f"received {target_length} and {self.num_tokens}."
            )
        unmask_k_tokens, sort_idx = self._esolm_generation_plan(
            num_samples=num_samples,
            generation_config=generation_config,
            device=device,
            num_tokens=target_length,
        )
        x = torch.full(
            (num_samples, target_length),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        x = torch.gather(x, dim=1, index=sort_idx)
        return x, sort_idx, unmask_k_tokens, prompt_length

    def _apply_stopping_criteria(
        self,
        samples: torch.LongTensor,
        stopping_criteria: StoppingCriteriaList | None,
        prompt_length: int = 0,
    ) -> torch.LongTensor:
        if stopping_criteria is None or len(stopping_criteria) == 0:
            return samples
        if samples.ndim != 2 or samples.shape[1] == 0:
            return samples

        stop_positions: list[int] = []
        search_start = max(prompt_length + 1, 1)
        total_length = samples.shape[1]
        for batch_idx in range(samples.shape[0]):
            stop_pos = total_length
            for end in range(search_start, total_length + 1):
                should_stop = stopping_criteria(
                    input_ids=samples[batch_idx : batch_idx + 1, :end],
                    scores=None,
                )
                if isinstance(should_stop, torch.Tensor):
                    stop_flag = bool(should_stop.reshape(-1)[0].item())
                else:
                    stop_flag = bool(should_stop)
                if stop_flag:
                    stop_pos = end
                    break
            stop_positions.append(stop_pos)

        if all(stop_pos == total_length for stop_pos in stop_positions):
            return samples
        if all(stop_pos == stop_positions[0] for stop_pos in stop_positions):
            return samples[:, : stop_positions[0]]

        pad_value = self.pad_token_id
        if pad_value is None:
            pad_value = self.eos_token_id
        if pad_value is None:
            pad_value = self.mask_token_id
        max_stop = max(stop_positions)
        truncated = torch.full(
            (samples.shape[0], max_stop),
            fill_value=pad_value,
            dtype=samples.dtype,
            device=samples.device,
        )
        for batch_idx, stop_pos in enumerate(stop_positions):
            truncated[batch_idx, :stop_pos] = samples[batch_idx, :stop_pos]
        return truncated

    def _forward_sample_step(
        self,
        zt: torch.LongTensor,
        sort_idx: torch.LongTensor,
        last_k_start: int,
        curr_k_start: int,
        curr_k_end: int,
    ) -> torch.FloatTensor:
        # Upstream source:
        # - `s-sahoo/Eso-LMs/algo.py::EsoLM.generate_samples` (lines 569-579)
        # - `s-sahoo/Eso-LMs/models/dit.py::EsoLMDiT.forward_sample`
        #   (lines 973-1009)
        # The official sampler requires an EsoLM-specific backbone decode path.
        if not hasattr(self.backbone, "forward_sample"):
            raise NotImplementedError(
                "EsoLM sampling requires a backbone with `forward_sample`; "
                f"`{type(self.backbone).__name__}` is unsupported."
            )
        backbone_output = self.backbone.forward_sample(
            zt=zt,
            sort_idx=sort_idx,
            last_k_start=last_k_start,
            curr_k_start=curr_k_start,
            curr_k_end=curr_k_end,
        )
        return getattr(backbone_output, "logits", backbone_output[0])

    def _backbone_forward(
        self,
        denoiser_inputs: DenoiserInput,
        **backbone_kwargs: Any,
    ):
        backbone_modules = (
            self.backbone,
            getattr(self.backbone, "_orig_mod", None),
            getattr(self.backbone, "model", None),
            getattr(getattr(self.backbone, "_orig_mod", None), "model", None),
        )
        backbone_supports_esolm = any(
            getattr(module, "is_esolm_backbone", False)
            for module in backbone_modules
            if module is not None
        )
        if not backbone_supports_esolm:
            raise NotImplementedError(
                "EsoLM requires a backbone marked with `is_esolm_backbone=True`; "
                f"`{type(self.backbone).__name__}` is not configured for that path."
            )
        merged_kwargs = dict(denoiser_inputs.backbone_kwargs)
        merged_kwargs.update(backbone_kwargs)
        if self._has_native_esolm_path(*backbone_modules):
            return self.backbone(
                denoiser_inputs.xt,
                attention_mask=denoiser_inputs.attention_mask,
                past_key_values=denoiser_inputs.past_key_values,
                **merged_kwargs,
            )

        hf_kwargs, sequential_output_length = self._build_hf_esolm_backbone_kwargs(
            denoiser_inputs=denoiser_inputs,
            merged_kwargs=merged_kwargs,
        )
        backbone_output = self.backbone(
            denoiser_inputs.xt,
            past_key_values=denoiser_inputs.past_key_values,
            **hf_kwargs,
        )
        if sequential_output_length is not None and getattr(
            backbone_output, "logits", None
        ) is not None:
            backbone_output.logits = backbone_output.logits[:, :sequential_output_length]
        return backbone_output

    def _any_order_ar_loss(self, x0: torch.LongTensor) -> torch.FloatTensor:
        # Direct transcription of `s-sahoo/Eso-LMs/algo.py::_any_order_ar_loss`
        # (lines 308-329). Deviation from upstream: we use `torch.binomial`
        # instead of `np.random.binomial` to stay within the local torch-only
        # runtime.
        offsets = torch.rand(1, self.num_tokens, device=x0.device)
        sort_idx = offsets.argsort(dim=-1, descending=False)
        num_diffusion = int(
            torch.binomial(
                torch.tensor(float(self.num_tokens), device=x0.device),
                torch.tensor(float(self.alpha_0), device=x0.device),
            ).item()
        )
        sort_idx[:, num_diffusion:] = sort_idx[:, num_diffusion:].sort(dim=-1).values
        sort_idx = sort_idx.expand(x0.shape[0], self.num_tokens)
        x0 = torch.gather(x0, dim=1, index=sort_idx)
        z0 = self._sample_q_xt(
            x0=x0,
            alpha_t=torch.zeros_like(x0, dtype=torch.float),
            mask=torch.zeros_like(x0, dtype=torch.float),
        )
        denoiser_inputs = DenoiserInput(
            xt=torch.cat([z0, x0], dim=1),
            x0=x0,
            attention_mask=torch.ones((x0.shape[0], x0.shape[1] * 2), device=x0.device),
            valid_tokens=torch.ones_like(x0, dtype=torch.float),
            tokens_mask=torch.ones_like(x0, dtype=torch.float),
            backbone_kwargs={
                "sigma": torch.zeros(x0.shape[0], device=x0.device, dtype=torch.float32),
                "sort_index": sort_idx,
                "sequential_input": True,
                "sequential_attn_mode": self.config.sequential_attn_mode,
                "mask_token_id": self.mask_token_id,
            },
        )
        backbone_output = self._backbone_forward(denoiser_inputs)
        logits = getattr(backbone_output, "logits", backbone_output[0])
        log_probs = self._forward(logits, denoiser_inputs)
        logp_per_token = log_probs.gather(-1, x0[:, :, None])[:, :, 0]
        return logp_per_token.sum(dim=1)

    def _importance_weighted_output(
        self,
        input_ids: torch.LongTensor,
        num_iw_orders: int,
    ) -> DenoiserOutput:
        # Upstream source: `s-sahoo/Eso-LMs/algo.py::_importance_weighted_loss`
        # (lines 331-353).
        if num_iw_orders <= 0:
            raise ValueError("Importance-weighted evaluation requires num_iw_orders > 0.")
        batch_size = input_ids.shape[0]
        logp_per_seq_per_order = torch.zeros(
            (batch_size, num_iw_orders),
            device=input_ids.device,
        )
        for i in range(num_iw_orders):
            logp_per_seq_per_order[:, i] = self._any_order_ar_loss(input_ids)
        log_num_orders = torch.log(
            torch.tensor(float(num_iw_orders), device=input_ids.device)
        )
        logp_per_seq = torch.logsumexp(logp_per_seq_per_order, dim=1) - log_num_orders
        nll_per_seq = -logp_per_seq
        loss = nll_per_seq.sum() / (batch_size * self.num_tokens)
        tokens_mask = torch.ones_like(input_ids, dtype=torch.float)
        nlls = (nll_per_seq / self.num_tokens)[:, None].expand_as(tokens_mask)
        return DenoiserOutput(
            denoiser_output=None,
            logits=None,
            tokens_mask=tokens_mask,
            loss=loss,
            nlls=nlls,
            other_loss_terms={"num_iw_orders": num_iw_orders},
        )

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int,
        num_steps: int | None = None,
        eps: float = 1e-5,
        generation_config: SetDiffusionGenerationConfig | None = None,
        prompt_input_ids: torch.LongTensor | None = None,
        target_length: int | None = None,
    ) -> tuple[torch.LongTensor, float, float]:
        # Direct transcription of `s-sahoo/Eso-LMs/algo.py::EsoLM.generate_samples`
        # (lines 513-599).
        # Deviations from upstream:
        # - the local implementation uses `mask_token_id` as the prior sample,
        #   which is equivalent to the official absorbing-diffusion prior;
        # - sampling goes through the shared `GenerationConfig` surface and local
        #   `_sample_categorical` / `_nucleus_sample` helpers instead of raw
        #   Gumbel-max code, but the induced categorical distribution is the same.
        del eps
        if generation_config is None:
            generation_config = getattr(self, "generation_config", None)
        if generation_config is None:
            generation_config = SetDiffusionGenerationConfig(
                num_steps=self.num_tokens if num_steps is None else num_steps,
                block_size=self.num_tokens,
                use_cache=True,
            )
        if num_steps is not None:
            generation_config.num_steps = num_steps
        if not getattr(generation_config, "use_cache", True):
            raise NotImplementedError(
                "EsoLM sampling requires `use_cache=True`; the upstream sampler is "
                "implemented around KV-cache reuse."
            )

        parameter = next(self.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        profile_throughput = bool(getattr(generation_config, "profile_throughput", False))
        if not hasattr(self.backbone, "forward_sample") or not hasattr(
            self.backbone, "reset_kv_cache"
        ):
            raise NotImplementedError(
                "EsoLM sampling is only supported for the dedicated EsoLM DiT "
                "backbone. Hugging Face and other generic backbones are "
                "unsupported because their decode path is not upstream-equivalent."
            )

        x, sort_idx, unmask_k_tokens, prompt_length = self._prepare_generation_inputs(
            num_samples=num_samples,
            generation_config=generation_config,
            device=device,
            prompt_input_ids=prompt_input_ids,
            target_length=target_length,
        )
        assert prompt_length + sum(unmask_k_tokens) == x.shape[1]
        unmasked_tokens = 0
        self.backbone.reset_kv_cache()
        if hasattr(self.backbone, "reset_sorted_rotary_cache"):
            self.backbone.reset_sorted_rotary_cache()

        if profile_throughput and device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        nfe = 0
        if prompt_length > 0:
            self._forward_sample_step(
                zt=x,
                sort_idx=sort_idx,
                last_k_start=0,
                curr_k_start=prompt_length,
                curr_k_end=prompt_length,
            )
            nfe += 1
        for i, k in enumerate(unmask_k_tokens):
            generation_offset = prompt_length
            last_k_start = (
                generation_offset
                if i == 0
                else generation_offset + unmasked_tokens - unmask_k_tokens[i - 1]
            )
            logits = self._forward_sample_step(
                zt=x,
                sort_idx=sort_idx,
                last_k_start=last_k_start,
                curr_k_start=generation_offset + unmasked_tokens,
                curr_k_end=generation_offset + unmasked_tokens + k,
            )
            nfe += 1
            if not profile_throughput:
                logits = logits.clone()
                logits[:, :, self.mask_token_id] = self.neg_infinity
                probs = logits.float().softmax(dim=-1)
                if generation_config.nucleus_p < 1.0:
                    probs = self._nucleus_sample(probs, generation_config.nucleus_p)
                indices = slice(
                    generation_offset + unmasked_tokens,
                    generation_offset + unmasked_tokens + k,
                )
                y = self._sample_categorical(
                    probs,
                    do_sample=getattr(generation_config, "do_sample", True),
                )
                x[:, indices] = y
            unmasked_tokens += k

        if profile_throughput and device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        self.backbone.reset_kv_cache()
        if hasattr(self.backbone, "reset_sorted_rotary_cache"):
            self.backbone.reset_sorted_rotary_cache()
        x = torch.gather(x, dim=1, index=self._reverse_indices(sort_idx))
        return x, float(nfe), duration

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
    ) -> torch.LongTensor | DiffusionGenerationOutput:
        # Upstream source: `s-sahoo/Eso-LMs/algo.py::EsoLM.generate_samples`
        # (lines 513-599).
        # Deviation from upstream: the Hugging Face-style `generate()` entrypoint is
        # retained for compatibility with this repo, but we fail fast for features
        # that the official Eso-LMs sampler does not implement, such as conditional
        # decoding and custom logits/stopping processors.
        del device, tokenizer, kwargs
        if generation_config is None:
            generation_config = getattr(self, "generation_config", None)
        if generation_config is None:
            generation_config = SetDiffusionGenerationConfig(
                num_steps=self.num_tokens,
                block_size=self.num_tokens,
                use_cache=True,
            )
        if logits_processor is not None and len(logits_processor) > 0:
            raise NotImplementedError(
                "EsoLM generation does not support custom logits processors."
            )

        input_length = inputs.shape[-1] if inputs is not None else 0
        if inputs is not None and inputs.numel() > 0:
            if (inputs == self.mask_token_id).any():
                raise NotImplementedError(
                    "EsoLM generation does not support infilling prompts; pass a "
                    "contiguous prompt prefix and `max_new_tokens` instead."
                )
            batch_size = inputs.shape[0]
            _, requested_new_tokens = self._compute_sampling_lengths(
                generation_config=generation_config,
                input_length=input_length,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )
            requested_new_tokens = int(requested_new_tokens)
            if requested_new_tokens < 0:
                raise ValueError(
                    "EsoLM prompt-conditioned generation received a negative "
                    f"`max_new_tokens` value: {requested_new_tokens}."
                )
            if requested_new_tokens == 0:
                if return_dict_in_generate:
                    return DiffusionGenerationOutput(
                        sequences=inputs,
                        parallelism_factor=0.0,
                        inf_budget=None,
                        inf_budgets=None,
                    )
                return inputs
            samples, nfe, _ = self.generate_samples(
                num_samples=batch_size,
                num_steps=getattr(generation_config, "num_steps", self.num_tokens),
                eps=getattr(generation_config, "min_t", 1e-5),
                generation_config=generation_config,
                prompt_input_ids=inputs,
                target_length=input_length + requested_new_tokens,
            )
            samples = self._apply_stopping_criteria(
                samples=samples,
                stopping_criteria=stopping_criteria,
                prompt_length=input_length,
            )
            if return_dict_in_generate:
                return DiffusionGenerationOutput(
                    sequences=samples,
                    parallelism_factor=requested_new_tokens / max(nfe, 1.0),
                    inf_budget=None,
                    inf_budgets=None,
                )
            return samples

        requested_length = (
            max_new_tokens
            if max_new_tokens is not None
            else max_length if max_length is not None else self.num_tokens
        )
        if requested_length != self.num_tokens:
            raise ValueError(
                "EsoLM unconditional generation samples exactly `config.length` "
                f"tokens; received request for {requested_length}."
            )

        samples, nfe, _ = self.generate_samples(
            num_samples=batch_size,
            num_steps=getattr(generation_config, "num_steps", self.num_tokens),
            eps=getattr(generation_config, "min_t", 1e-5),
            generation_config=generation_config,
        )
        samples = self._apply_stopping_criteria(
            samples=samples,
            stopping_criteria=stopping_criteria,
            prompt_length=0,
        )
        if return_dict_in_generate:
            return DiffusionGenerationOutput(
                sequences=samples,
                parallelism_factor=self.num_tokens / max(nfe, 1.0),
                inf_budget=None,
                inf_budgets=None,
            )
        return samples

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        compute_loss: Optional[bool] = True,
        **kwargs: Any,
    ) -> DenoiserOutput:
        # Eso-LMs source: `algo.py::EsoLM._loss`, `algo.py::EsoLM.nll`, and
        # `algo.py::EsoLM._reconstruction_loss`.
        # Change from original: the official repository computes diffusion and
        # sequential losses in a trainer class, while this repo expects all loss
        # computation to happen inside the denoiser. We therefore override
        # `forward` to execute the two branches and merge their outputs here.
        if past_key_values is not None:
            raise NotImplementedError(
                "EsoLM does not support the generic `past_key_values` forward path. "
                "Use `generate_samples()` / `generate()` with the dedicated "
                "EsoLM backbone sampler instead."
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)
        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
            context_mask = context_mask.to(torch.int)

        if not compute_loss:
            diffusion_inputs = self._prepare_diffusion_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
                t=t,
            )
            backbone_output = self._backbone_forward(diffusion_inputs, **kwargs)
            logits = getattr(backbone_output, "logits", backbone_output[0])
            denoiser_output = self._forward(logits, diffusion_inputs, **kwargs)
            return DenoiserOutput(
                denoiser_output=denoiser_output,
                logits=logits,
                tokens_mask=diffusion_inputs.tokens_mask,
                past_key_values=getattr(backbone_output, "past_key_values", None),
            )

        num_iw_orders = self._resolve_num_iw_orders(kwargs)
        if num_iw_orders > 0:
            if not attention_mask.bool().all():
                raise NotImplementedError(
                    "EsoLM importance-weighted evaluation only supports full-length "
                    "sequences, matching upstream."
                )
            if context_mask.bool().any():
                raise NotImplementedError(
                    "EsoLM importance-weighted evaluation does not support context "
                    "tokens; upstream evaluates unconditional sequences only."
                )
            return self._importance_weighted_output(
                input_ids=input_ids,
                num_iw_orders=num_iw_orders,
            )

        batch_size = input_ids.shape[0]
        split_batch = int(self.config.batch_split * batch_size)
        if split_batch < 0 or split_batch > batch_size:
            raise ValueError(
                f"EsoLM batch_split={self.config.batch_split} produced "
                f"split_batch={split_batch} for batch_size={batch_size}."
            )

        do_diffusion = self.config.alpha_0 != 0
        do_sequential = self.config.alpha_0 != 1
        if do_diffusion and split_batch == 0:
            raise ValueError(
                f"EsoLM diffusion branch is empty but alpha_0={self.config.alpha_0} "
                f"requires it. Increase batch_split or batch size "
                f"(batch_split={self.config.batch_split}, batch_size={batch_size})."
            )
        if do_sequential and split_batch == batch_size:
            raise ValueError(
                f"EsoLM sequential branch is empty but alpha_0={self.config.alpha_0} "
                f"requires it. Decrease batch_split or increase batch size "
                f"(batch_split={self.config.batch_split}, batch_size={batch_size})."
            )

        losses = []
        nll_chunks = []
        tokens_mask_chunks = []
        other_loss_terms: dict[str, Any] = {}
        denoiser_output = None
        backbone_output = None

        if do_diffusion:
            diffusion_inputs = self._prepare_diffusion_inputs(
                input_ids=input_ids[:split_batch],
                attention_mask=attention_mask[:split_batch],
                context_mask=context_mask[:split_batch],
                t=t[:split_batch] if t is not None and t.ndim > 0 else None,
            )
            diffusion_backbone = self._backbone_forward(diffusion_inputs, **kwargs)
            diffusion_logits = getattr(diffusion_backbone, "logits", diffusion_backbone[0])
            diffusion_output = self._forward(diffusion_logits, diffusion_inputs, **kwargs)
            diffusion_loss, diffusion_nlls = self._compute_diffusion_loss(
                diffusion_output,
                diffusion_inputs,
            )
            losses.append(diffusion_loss)
            nll_chunks.append(diffusion_nlls)
            tokens_mask_chunks.append(diffusion_inputs.tokens_mask)
            other_loss_terms["diffusion_tokens_mask"] = diffusion_inputs.tokens_mask
            other_loss_terms["diffusion_valid_tokens"] = diffusion_inputs.valid_tokens
            denoiser_output = diffusion_output
            backbone_output = diffusion_logits

        if do_sequential:
            sequential_inputs = self._prepare_sequential_inputs(
                input_ids=input_ids[split_batch:],
                attention_mask=attention_mask[split_batch:],
                context_mask=context_mask[split_batch:],
            )
            sequential_backbone = self._backbone_forward(sequential_inputs, **kwargs)
            sequential_logits = getattr(
                sequential_backbone, "logits", sequential_backbone[0]
            )
            sequential_output = self._forward(
                sequential_logits,
                sequential_inputs,
                **kwargs,
            )
            sequential_loss, sequential_nlls = self._compute_sequential_loss(
                sequential_output,
                sequential_inputs,
            )
            losses.append(sequential_loss)
            nll_chunks.append(sequential_nlls)
            tokens_mask_chunks.append(sequential_inputs.tokens_mask)
            other_loss_terms["sequential_tokens_mask"] = sequential_inputs.tokens_mask
            other_loss_terms["sequential_valid_tokens"] = sequential_inputs.valid_tokens
            other_loss_terms["reconstruction_loss"] = sequential_loss
            denoiser_output = sequential_output
            backbone_output = sequential_logits

        if not losses:
            raise ValueError(
                "EsoLM produced no active loss branch. Check alpha_0 and batch_split."
            )

        loss = torch.stack(losses).sum()
        nlls = torch.cat(nll_chunks, dim=0)
        tokens_mask = torch.cat(tokens_mask_chunks, dim=0)
        return DenoiserOutput(
            denoiser_output=denoiser_output,
            logits=backbone_output,
            tokens_mask=tokens_mask,
            loss=loss,
            nlls=nlls,
            other_loss_terms=other_loss_terms,
        )
