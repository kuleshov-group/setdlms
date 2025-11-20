from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Cache,
    FlashAttentionKwargs,
    Qwen3Attention,
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    eager_attention_forward,
    rotate_half,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache

logger = logging.get_logger(__name__)


def custom_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, q_start_idx=0):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos[..., q_start_idx:, :]) + (
        rotate_half(q) * sin[..., q_start_idx:, :]
    )
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CustomQwen3Attention(Qwen3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        # Add pre-attention layernorm for learned position embeddings
        # Use the same type as q_norm (typically RMSNorm for Qwen3)
        self.pre_attn_layernorm = type(self.q_norm)(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        q_start_idx: int = 0,  # > 0: decoder pass w/encoder inputs in hidden_states
        use_learned_position_embeddings: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        sa_hidden_sates = hidden_states[:, q_start_idx:, :]
        query_input_shape = sa_hidden_sates.shape[:-1]
        query_hidden_shape = (*query_input_shape, -1, self.head_dim)

        # Apply pre-attention layernorm if using learned position embeddings
        if use_learned_position_embeddings:
            sa_hidden_sates = self.pre_attn_layernorm(sa_hidden_sates)

        query_states = self.q_norm(
            self.q_proj(sa_hidden_sates).reshape(query_hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE only if not using learned positional embeddings
        if not use_learned_position_embeddings and position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = custom_apply_rotary_pos_emb(
                query_states, key_states, cos, sin, q_start_idx=q_start_idx
            )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models
            # cache_position needed for the static cache
            if use_learned_position_embeddings or position_embeddings is None:
                cache_kwargs = {"cache_position": cache_position}
            else:
                cos, sin = position_embeddings
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # NOTE: downcast for flex-attention compatibility
        query_states, key_states = (
            query_states.to(value_states.dtype),
            key_states.to(value_states.dtype),
        )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*query_input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CustomQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.self_attn = CustomQwen3Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        q_start_idx: int = 0,
        use_learned_position_embeddings: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states[:, q_start_idx:, ...]

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            q_start_idx=q_start_idx,
            use_learned_position_embeddings=use_learned_position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # return hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class CustomQwen3Model(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                CustomQwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # Initialize weights and apply final processing
        # if getattr(config, "add_position_embeddings_additional", False):
        #     self.position_embeddings_additional = nn.Linear(config.head_dim * 2, config.head_dim)
        
        # Initialize learned positional embeddings if enabled
        self.use_learned_position_embeddings = False
        if self.use_learned_position_embeddings:
            max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
            self.learned_position_embeddings = nn.Embedding(
                max_position_embeddings, config.hidden_size
            )
            # Initialize with small random values
            nn.init.normal_(self.learned_position_embeddings.weight, std=0.02)
        
        self.post_init()
    

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # Apply learned positional embeddings if enabled
        if self.use_learned_position_embeddings:
            # Add learned positional embeddings to hidden states
            # position_ids should already be set above, but handle edge case
            if position_ids is None:
                seq_length = hidden_states.shape[1]
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            # Clamp position_ids to valid range for learned embeddings
            max_pos = self.learned_position_embeddings.num_embeddings
            position_ids_clamped = position_ids.clamp(max=max_pos - 1)
            position_embeddings_learned = self.learned_position_embeddings(position_ids_clamped)
            hidden_states = hidden_states + position_embeddings_learned
            # Set position_embeddings to None for attention (RoPE won't be applied)
            position_embeddings = None
        else:
            # create position embeddings to be shared across the decoder layers (RoPE)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # additional_position_ids = flash_attn_kwargs.get("permutation_order", None)
        # if additional_position_ids is not None:
        #     position_embeddings_additional = self.rotary_emb(hidden_states, additional_position_ids)
        #     position_embeddings = (
        #         self.position_embeddings_additional(torch.cat([position_embeddings[0], position_embeddings_additional[0]], dim=-1)),
        #         self.position_embeddings_additional(torch.cat([position_embeddings[1], position_embeddings_additional[1]], dim=-1)),
        #     )
        #     # position_embeddings = (
        #     #     position_embeddings[0] + position_embeddings_additional[0],
        #     #     position_embeddings[1] + position_embeddings_additional[1],
        #     # )
        #     # position_embeddings = (
        #     #     torch.cat([position_embeddings[0][..., :(self.config.head_dim // 2)], position_embeddings_additional[0][..., (self.config.head_dim // 2):]], dim=-1),
        #     #     torch.cat([position_embeddings[1][..., :(self.config.head_dim // 2)], position_embeddings_additional[1][..., (self.config.head_dim // 2):]], dim=-1),
        #     # )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_learned_position_embeddings=self.use_learned_position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        # Initialize a new model with custom layers
        self.model = CustomQwen3Model(config)

        # Initialize weights and apply final processing
        self.post_init()
