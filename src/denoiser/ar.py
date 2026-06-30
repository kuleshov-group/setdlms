import copy
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import (
    ExponentialDecayLengthPenalty,
    MinNewTokensLengthLogitsProcessor,
)
from transformers.generation.utils import GenerateOutput

from src.denoiser.base import Denoiser, DenoiserConfig, DenoiserInput, LossAndNllOutput


class ARConfig(DenoiserConfig):
    """Configuration class for autoregressive (AR) models."""

    model_type = "ar"
    auto_map = {
        "AutoConfig": "ar.ARConfig",
        "AutoModel": "ar.AR",
        "AutoModelForCausalLM": "ar.AR",
    }

    def __init__(
        self,
        length: Optional[int] = None,
        backbone_config: Optional[Dict[str, Any]] = None,
        tokenization_config: Optional[Dict[str, Any]] = None,
        noise_config: None = None,
        **kwargs,
    ):
        super().__init__(
            length=length,
            backbone_config=backbone_config,
            noise_config=noise_config,
            tokenization_config=tokenization_config,
            **kwargs,
        )


class AR(Denoiser):
    """Denoiser class for autoregressive (AR) models."""

    config_class = ARConfig

    def __init__(
        self,
        config: ARConfig,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
    ) -> DenoiserInput:
        # Prepare inputs for autoregressive model
        labels = copy.deepcopy(input_ids[..., 1:])[..., None]
        input_ids = input_ids[..., :-1]
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            attention_mask = attention_mask[..., :-1]
        if context_mask is None:
            context_mask = torch.zeros_like(input_ids)
        elif context_mask.sum() == 0 and (
            attention_mask is None or (attention_mask == 1).all()
        ):
            attention_mask = None
        else:
            context_mask = context_mask[..., :-1]
        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask)
        return DenoiserInput(
            xt=input_ids,  # type: ignore
            x0=labels,  # type: ignore
            attention_mask=attention_mask,
            context_mask=context_mask,
            tokens_mask=tokens_mask,
            past_key_values=past_key_values,
        )

    def _prepare_inputs_inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context: Optional[torch.LongTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        cache: Optional[Dict[str, Any]] = None,
        **backbone_kwargs: Any,
    ) -> DenoiserInput:
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        cache_length = self._get_past_key_values_seq_length(past_key_values)
        full_seq_length = cache_length + input_ids.shape[-1]
        # --- crop KV cache if we would exceed model context ---
        if full_seq_length > self.config.length:
            overflow = full_seq_length - self.config.length
            past_key_values = self._crop_kv_cache_left(past_key_values, overflow)
        cache["past_key_values"] = past_key_values
        return DenoiserInput(
            xt=input_ids,
            x0=context,
            attention_mask=attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            backbone_kwargs=backbone_kwargs,
        )

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        # Shift labels
        loss = -torch.gather(model_output, -1, denoiser_inputs.x0).squeeze(-1)

        nlls = loss * denoiser_inputs.tokens_mask

        # Compute per-batch counts and losses to avoid division by zero
        count = denoiser_inputs.tokens_mask.sum(dim=-1)  # Per-batch counts
        batch_nll = nlls.sum(dim=-1)  # Per-batch losses

        # Avoid division by zero: if count is 0, set token_nll to 0
        token_nll = torch.where(
            count > 0, batch_nll / count, torch.zeros_like(batch_nll)
        ).mean()

        return LossAndNllOutput(loss=token_nll, nlls=nlls)  # type: ignore

    def _nucleus_sample(self, p_x0: torch.FloatTensor, p: float):
        if p >= 1.0:
            return p_x0
        sorted_probs, sorted_indices = p_x0.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)

        # Match the diffusion samplers: remove tokens after the first token that
        # pushes cumulative mass over p, while keeping that crossing token.
        nucleus_mask = cum_probs >= p
        nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
        nucleus_mask[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(nucleus_mask, 0.0)
        filtered = torch.zeros_like(p_x0)
        filtered.scatter_(-1, sorted_indices, sorted_probs)
        filtered /= filtered.sum(-1, keepdim=True)
        return filtered

    def _generate_unconditional(
        self,
        generation_config: GenerationConfig,
        x: torch.LongTensor,
        log_x_theta: torch.FloatTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        processor_input_ids: Optional[torch.LongTensor] = None,
        length_penalty_input_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        if logits_processor is not None and len(logits_processor) > 0:
            processor_input_ids = (
                x if processor_input_ids is None else processor_input_ids
            )
            length_penalty_input_ids = (
                processor_input_ids
                if length_penalty_input_ids is None
                else length_penalty_input_ids
            )
            for lp in logits_processor:
                if isinstance(lp, MinNewTokensLengthLogitsProcessor):
                    eos_token_id = getattr(lp, "eos_token_id", None)
                    if isinstance(eos_token_id, torch.Tensor):
                        lp.eos_token_id = eos_token_id.to(
                            device=log_x_theta.device
                        )
                lp_input_ids = (
                    length_penalty_input_ids
                    if isinstance(
                        lp,
                        (
                            ExponentialDecayLengthPenalty,
                            MinNewTokensLengthLogitsProcessor,
                        ),
                    )
                    else processor_input_ids
                )
                log_x_theta = lp(input_ids=lp_input_ids, scores=log_x_theta)
            log_x_theta = log_x_theta.log_softmax(dim=-1)
        else:
            log_x_theta = self._forward(log_x_theta, denoiser_inputs=None)
        probs = log_x_theta.exp()
        if getattr(generation_config, "nucleus_p", 1.0) < 1.0:
            probs = self._nucleus_sample(
                probs, p=getattr(generation_config, "nucleus_p", 1.0)
            )
        y = self._sample_categorical(
            probs, do_sample=getattr(generation_config, "do_sample", False)
        )
        confidence = probs.gather(-1, y[..., None]).squeeze(dim=-1)
        return y, confidence

    def _crop_kv_cache_left(self, past_key_values: Any, drop: int) -> Any:
        """
        Drop `drop` tokens from the *left/oldest* side of the KV cache.
        Works with common DynamicCache-like implementations that store per-layer
        key/value tensors in `key_cache` / `value_cache` lists.
        Falls back to no-op if structure is unknown.
        """
        if drop <= 0 or past_key_values is None:
            return past_key_values

        assert hasattr(past_key_values, "key_cache") and hasattr(
            past_key_values, "value_cache"
        ), "DynamicCache-like structure not found"
        key_cache = getattr(past_key_values, "key_cache")
        value_cache = getattr(past_key_values, "value_cache")
        for i in range(len(past_key_values)):
            k = key_cache[i]
            v = value_cache[i]
            if k is None or v is None:
                continue

            key_cache[i] = k[..., drop:, :]
            value_cache[i] = v[..., drop:, :]
        return past_key_values

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        disable_pbar: Optional[bool] = None,  # not used; compat. w/other denoisers
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        assert tokenizer is not None, "Tokenizer is required"
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "generate"):
            generation_logits_processor = logits_processor
            if logits_processor is not None and inputs is not None:
                generation_logits_processor = copy.deepcopy(logits_processor)
                for lp in generation_logits_processor:
                    if isinstance(lp, ExponentialDecayLengthPenalty):
                        # Hydra constructs this processor with input_ids_seq_length=0.
                        # HF generate passes the full prompt as input_ids, so offset the
                        # start index to make regulation_start count generated tokens.
                        lp.regulation_start += inputs.shape[-1]
                    elif isinstance(lp, MinNewTokensLengthLogitsProcessor):
                        # Hydra constructs this with prompt_length_to_skip=0. HF
                        # generate passes prompt+generated tokens, so offset the
                        # prompt skip to make min_new_tokens count target tokens.
                        lp.prompt_length_to_skip += inputs.shape[-1]
                        eos_token_id = getattr(lp, "eos_token_id", None)
                        if isinstance(eos_token_id, torch.Tensor):
                            lp.eos_token_id = eos_token_id.to(device=inputs.device)
            outputs = self.backbone.model.generate(
                inputs=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=generation_config,
                logits_processor=generation_logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            return outputs
        num_pred_tokens = max_new_tokens
        batch_size = inputs.shape[0]
        input_len = inputs.shape[-1]
        device = inputs.device
        cache = {}
        all_position_ids = (
            torch.arange(input_len + num_pred_tokens).to(device).unsqueeze(0)
        )

        if input_len == 0:
            x = torch.full(
                (batch_size, num_pred_tokens + 1),
                fill_value=tokenizer.mask_token_id,
                dtype=torch.long,
                device=device,
            )
            x[:, 0] = tokenizer.bos_token_id
            input_len = 0
        else:
            x = torch.cat(
                (
                    inputs,
                    torch.full(
                        (batch_size, num_pred_tokens),
                        fill_value=tokenizer.mask_token_id,
                        dtype=torch.long,
                        device=device,
                    ),
                ),
                dim=-1,
            )
        accumulated_confidence = torch.full(
            x.shape,
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        accumulated_confidence[x != tokenizer.mask_token_id] = 1.0
        if input_len != 0:
            # cache
            cache_inputs = self._prepare_inputs_inference(
                inputs,
                None,
                None,
                None,
                cache,
                position_ids=all_position_ids[:, :input_len],
            )
            backbone_output = self._backbone_forward(
                cache_inputs,
            )
            if isinstance(backbone_output, torch.Tensor):
                logits = backbone_output
            else:
                logits = backbone_output["logits"]
            logits[:, :, tokenizer.mask_token_id] = -torch.inf
            next_token, token_confidence = self._generate_unconditional(
                generation_config,
                x,
                logits[:, -1],
                logits_processor,
                processor_input_ids=x[:, :input_len],
                length_penalty_input_ids=x[:, input_len:input_len],
            )
            x[:, input_len] = next_token
            accumulated_confidence[:, input_len] = token_confidence.to(
                accumulated_confidence
            )
            cache["past_key_values"] = backbone_output["past_key_values"]

        for i in range(input_len, x.shape[-1] - 1):
            denoiser_inputs = self._prepare_inputs_inference(
                x[:, i].unsqueeze(-1),
                None,
                None,
                None,
                cache,
                position_ids=all_position_ids[:, i : i + 1],
            )
            backbone_output = self._backbone_forward(
                denoiser_inputs,
            )
            if isinstance(backbone_output, torch.Tensor):
                logits = backbone_output
            else:
                logits = backbone_output["logits"]
            logits[:, :, tokenizer.mask_token_id] = -torch.inf
            cache["past_key_values"] = backbone_output["past_key_values"]
            next_token, token_confidence = self._generate_unconditional(
                generation_config,
                x,
                logits[:, -1],
                logits_processor,
                processor_input_ids=x[:, : i + 1],
                length_penalty_input_ids=x[:, input_len : i + 1],
            )
            x[:, i + 1] = next_token
            accumulated_confidence[:, i + 1] = token_confidence.to(
                accumulated_confidence
            )
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=x[:, : i + 2],
                    scores=None,
                    token_confidence=accumulated_confidence[:, : i + 2],
                )
                if torch.any(is_done):
                    x = x[:, : i + 2]
                    break

        # Decode output
        return x
