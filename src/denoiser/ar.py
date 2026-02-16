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
from transformers.generation.utils import GenerateOutput
from tqdm import tqdm
from src.denoiser.base import (
    Denoiser,
    DenoiserConfig,
    DenoiserInput,
    LossAndNllOutput,
)


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
        elif (
            context_mask.sum() == 0
            and (attention_mask is None or (attention_mask == 1).all())
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
        count = denoiser_inputs.tokens_mask.sum(dim=-1)

        batch_nll = nlls.sum(dim=-1)
        token_nll = (batch_nll / count).mean()

        return LossAndNllOutput(loss=token_nll, nlls=nlls)  # type: ignore

    def _nucleus_sample(self, p_x0: torch.FloatTensor, p: float):
        if p == 1.0:
            return p_x0
        sorted_probs, sorted_indices = p_x0.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cum_probs <= p
        nucleus_mask[..., 0] = 1
        sorted_probs = sorted_probs * nucleus_mask
        p_x0.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
        p_x0 /= p_x0.sum(-1, keepdim=True)
        return p_x0

    def _generate_unconditional(
        self,
        generation_config: GenerationConfig,
        x: torch.LongTensor,
        log_x_theta: torch.FloatTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> torch.LongTensor:
        # need to sample a gumbel for each token
        # to save memory in variable-length sampling
        batch_size = x.shape[0]
        device = x.device
        # assert tokenizer is not None, "Tokenizer is required"
        # if logits_processor is not None:
        #     log_x_theta = logits_processor(input_ids=x, scores=log_x_theta)
        #     log_x_theta = log_x_theta.log_softmax(dim=-1)
        log_x_theta = self._nucleus_sample(log_x_theta.exp(), p=getattr(generation_config, "nucleus_p", 1.0)).log()
        y = self._sample_categorical(log_x_theta.exp(), do_sample=getattr(generation_config, "do_sample", False))
        return y

    def _crop_kv_cache_left(
        self, past_key_values: Any, drop: int
    ) -> Any:
        """
        Drop `drop` tokens from the *left/oldest* side of the KV cache.
        Works with common DynamicCache-like implementations that store per-layer
        key/value tensors in `key_cache` / `value_cache` lists.
        Falls back to no-op if structure is unknown.
        """
        if drop <= 0 or past_key_values is None:
            return past_key_values

        assert hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"), "DynamicCache-like structure not found"
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

        # if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "generate"):
        #     outputs = self.backbone.model.generate(
        #         inputs=inputs,
        #         attention_mask=torch.ones_like(inputs),
        #         generation_config=generation_config,
        #         logits_processor=logits_processor,
        #         # TODO: debug: passing EOS stopping criteria generates EOS right away?
        #         # stopping_criteria=stopping_criteria,
        #         max_length=max_length,
        #         max_new_tokens=max_new_tokens,
        #         # TODO: Can we pass this in `generation_config`?
        #         # eos_token_id=None,  # Uncomment for t-put runs; prevents stopping at EOS
        #         **kwargs,
        #     )
        #     return outputs
        num_pred_tokens = max_new_tokens
        batch_size = inputs.shape[0]
        input_len = inputs.shape[-1]
        device = inputs.device
        vocab_size = tokenizer.vocab_size
        cache = {}
        all_position_ids = torch.arange(input_len + num_pred_tokens).to(device).unsqueeze(0)

        if input_len == 0:
            x = torch.zeros(
                (batch_size, num_pred_tokens + 1),
                dtype=torch.long,
                device=device)
            x[:, 0] = tokenizer.bos_token_id
            input_len = 0
        else:
            x = torch.cat((inputs, torch.zeros((batch_size, num_pred_tokens), dtype=torch.long, device=device)), dim=-1)
            # cache
            cache_inputs = self._prepare_inputs_inference(inputs, None, None, None, cache, position_ids=all_position_ids[:, :input_len])
            backbone_output = self._backbone_forward(
                cache_inputs,
            )
            if isinstance(backbone_output, torch.Tensor):
                logits = backbone_output
            else:
                logits = backbone_output["logits"]
            logits[:, :, tokenizer.mask_token_id] = -torch.inf
            log_x_theta = self._forward(logits, cache_inputs, **kwargs)
            x[:, input_len] = self._generate_unconditional(generation_config, x[:, :input_len], log_x_theta[:, -1], logits_processor, tokenizer)
            cache["past_key_values"] = backbone_output["past_key_values"]
        
        for i in range(input_len, x.shape[-1] - 1):
            denoiser_inputs = self._prepare_inputs_inference(x[:, i].unsqueeze(-1), None, None, None, cache, position_ids=all_position_ids[:, i:i+1])
            backbone_output = self._backbone_forward(
                denoiser_inputs,
            )
            if isinstance(backbone_output, torch.Tensor):
                logits = backbone_output
            else:
                logits = backbone_output["logits"]
            logits[:, :, tokenizer.mask_token_id] = -torch.inf
            cache["past_key_values"] = backbone_output["past_key_values"]
            log_x_theta = self._forward(logits, denoiser_inputs, **kwargs)
            x[:, i + 1] = self._generate_unconditional(generation_config, x[:, input_len:i+1], log_x_theta[:, -1], logits_processor, tokenizer)
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=x[:, :i+2],
                    scores=None,
                )
                if torch.any(is_done):
                    x = x[:, :i+2]
                    break
                    
        # Decode output
        return x
