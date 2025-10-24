from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

from src.backbone.custom_modeling_qwen3 import CustomQwen3ForCausalLM

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None


logger = logging.get_logger(__name__)


@dataclass
class EncoderBaseModelOutputWithPast(ModelOutput):
    """Custom (encoder) model output.
    Stores previous decoder and updated encoder cache and encoder last hidden state.
    """

    past_key_values: Optional[Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]] = (
        None
    )
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_past_key_values: Optional[
        Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]
    ] = None


@dataclass
class DecoderCausalLMOutputWithPast(ModelOutput):
    """Custom (decoder) model output.
    Stores previous encoder and updated decoder cache and decoder logits.
    """

    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]] = (
        None
    )
    encoder_past_key_values: Optional[
        Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]
    ] = None


class LLMasEncoderDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_length: int,
        attn_backend: str = "sdpa",
        freeze_encoder: bool = False,
        reinit_encoder: bool = False,
        reinit_decoder: bool = False,
        tie_encoder_decoder_weights: bool = False,
        use_encoder_causal_mask: bool = False,
        num_encoder_layers: int = -1,
        num_decoder_layers: int = -1,
        keep_top_encoder_layers: bool = False,
        keep_top_decoder_layers: bool = False,
        use_gradient_checkpointing: bool = False,
        **llm_init_kwargs,
    ):
        assert not (tie_encoder_decoder_weights and reinit_decoder), (
            "Cannot tie encoder-decoder weights and reinitialize decoder."
        )
        assert not (tie_encoder_decoder_weights and freeze_encoder), (
            "Cannot freeze encoder weights when tying encoder-decoder weights."
        )
        super().__init__()
        self.use_encoder_causal_mask = use_encoder_causal_mask
        self.tie_encoder_decoder_weights = tie_encoder_decoder_weights

        if reinit_encoder:
            assert num_encoder_layers > 0
            encoder_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                num_hidden_layers=num_encoder_layers,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )
            self.encoder = CustomQwen3ForCausalLM(encoder_config)
        else:
            self.encoder = CustomQwen3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )
            assert num_encoder_layers <= len(self.encoder.model.layers), (
                f"Cannot keep {num_encoder_layers} layers. "
                f"Pre-trained model only has {len(self.encoder.model.layers)} layers."
            )
            num_encoder_layers = (
                len(self.encoder.model.layers)
                if num_encoder_layers == -1
                else num_encoder_layers
            )
            if keep_top_encoder_layers:
                self.encoder.model.layers = self.encoder.model.layers[
                    -num_encoder_layers:
                ]
            else:
                self.encoder.model.layers = self.encoder.model.layers[
                    :num_encoder_layers
                ]

        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if "embed_tokens" not in name:
                    param.requires_grad = False
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        if tie_encoder_decoder_weights:
            self.decoder = self.encoder
            num_decoder_layers = (
                len(self.decoder.model.layers)
                if num_decoder_layers == -1
                else num_decoder_layers
            )
            assert num_decoder_layers <= len(self.decoder.model.layers), (
                f"Cannot keep {num_decoder_layers} layers. "
                f"Pre-trained model only has {len(self.decoder.model.layers)} layers."
            )
            # Keep **top** layers when tying weights
            self.decoder_layer_idxs = list(range(len(self.encoder.model.layers)))[
                -num_decoder_layers:
            ]

        else:
            if reinit_decoder:
                assert num_decoder_layers > 0
                decoder_config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    num_hidden_layers=num_decoder_layers,
                    attn_implementation=attn_backend,
                    **llm_init_kwargs,
                )
                self.decoder = CustomQwen3ForCausalLM(decoder_config)
            else:
                self.decoder = CustomQwen3ForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                    **llm_init_kwargs,
                )
                assert num_decoder_layers <= len(self.decoder.model.layers), (
                    f"Cannot keep {num_decoder_layers} layers. "
                    f"Pre-trained model only has {len(self.decoder.layers)} layers."
                )
                if keep_top_decoder_layers:
                    self.decoder.model.layers = self.decoder.model.layers[
                        -num_decoder_layers:
                    ]
                else:
                    self.decoder.model.layers = self.decoder.model.layers[
                        :num_decoder_layers
                    ]
            del self.decoder.model.embed_tokens
            # if in the original LM, the lm_head is weight-tied to embedding,
            # point decoder lm_head to encoder's (instead of initializing separately)
            if (
                self.encoder.lm_head.weight.data_ptr()
                == self.encoder.model.embed_tokens.weight.data_ptr()
            ):
                self.decoder.lm_head = self.encoder.lm_head
            else:
                del self.encoder.lm_head
            if use_gradient_checkpointing:
                self.decoder.gradient_checkpointing_enable()
        self.max_length = max_length

    def freeze_encoder(self):
        for p in self.encoder.model.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.model.parameters():
            p.requires_grad = True

    # noinspection PyUnusedLocal
    def forward(
        self,
        # Decoder inputs
        input_ids: torch.LongTensor,
        attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        encoder_last_hidden_state: Optional[torch.FloatTensor] = None,
        # Encoder inputs
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_cache_position: Optional[torch.LongTensor] = None,
        encoder_past_key_values: Optional[DynamicCache] = None,
        # Additional args
        fix_cache_length: bool = True,  # Not used; compatibility with other backbones
        return_updated_cache: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[DecoderCausalLMOutputWithPast, EncoderBaseModelOutputWithPast]:
        # During training/eval encoder_last_hidden_state is None.
        # During generation encoder_last_hidden_state can be not None.
        new_seen_tokens = (
            0
            if encoder_last_hidden_state is None
            else encoder_last_hidden_state.shape[1]
        )
        # Encode clean tokens
        if encoder_input_ids is not None:
            if self.use_encoder_causal_mask:
                encoder_attention_mask = None  # None --> enforces use of causal mask
            if encoder_cache_position is None and encoder_position_ids is not None:
                encoder_cache_position = encoder_position_ids[0]
            encoder_output = self.encoder.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                use_cache=True,
                past_key_values=encoder_past_key_values,
                cache_position=encoder_cache_position,
            )
            if return_updated_cache:
                # encoder_output.past_key_values now contains latest encoder input
                return EncoderBaseModelOutputWithPast(
                    encoder_last_hidden_state=encoder_output.last_hidden_state,
                    encoder_past_key_values=encoder_output.past_key_values,
                    past_key_values=past_key_values,
                )
            encoder_last_hidden_state = encoder_output.last_hidden_state

        # Run decoder with xattn to clean token hidden states
        if encoder_last_hidden_state is None:  # No new encoder tokens
            q_start_idx = 0
            decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)
            if cache_position is None:
                if position_ids is not None:
                    cache_position = position_ids[0]
                else:
                    past_seen_tokens = (
                        past_key_values.get_seq_length()
                        if past_key_values is not None
                        else 0
                    )
                    cache_position = torch.arange(
                        past_seen_tokens,
                        past_seen_tokens + decoder_hidden_states.shape[1],
                        device=decoder_hidden_states.device,
                    )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            decoder_position_embeddings = self.decoder.model.rotary_emb(
                decoder_hidden_states, position_ids
            )
        else:
            q_start_idx = encoder_last_hidden_state.shape[1]
            decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)
            decoder_hidden_states = torch.cat(
                [
                    encoder_last_hidden_state,
                    decoder_hidden_states,
                ],
                dim=1,
            )
            if cache_position is None:
                if position_ids is not None:
                    cache_position = position_ids[0]
                else:
                    past_seen_tokens = (
                        past_key_values.get_seq_length()
                        if past_key_values is not None
                        else 0
                    )
                    cache_position = torch.cat(
                        [
                            torch.arange(  # clean token position ids
                                past_seen_tokens,
                                past_seen_tokens + encoder_last_hidden_state.shape[1],
                                device=decoder_hidden_states.device,
                            ),
                            torch.arange(  # noisy position ids
                                past_seen_tokens + new_seen_tokens,
                                past_seen_tokens + new_seen_tokens + input_ids.shape[1],
                                device=decoder_hidden_states.device,
                            ),
                        ],
                        dim=-1,
                    )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            decoder_position_embeddings = self.decoder.model.rotary_emb(
                decoder_hidden_states, position_ids
            )

        if hasattr(self.decoder.model, "_update_causal_mask"):  # bc on transformers
            # noinspection PyProtectedMember
            attention_mask = self.decoder.model._update_causal_mask(
                attention_mask=attention_mask,
                input_tensor=decoder_hidden_states,
                cache_position=cache_position,
                past_key_values=past_key_values,
                output_attentions=False,
            )
        for decoder_layer in self.decoder.model.layers:
            layer_idx = decoder_layer.self_attn.layer_idx
            if (
                self.tie_encoder_decoder_weights
                and layer_idx not in self.decoder_layer_idxs
            ):
                continue
            # past_key_values gets updated in-place.
            # Record previous length to re-truncate after each layer forward
            if past_key_values is not None and len(past_key_values) > layer_idx:
                prev_cache_len = past_key_values[layer_idx][0].shape[-2]  # type: ignore
            else:
                prev_cache_len = 0
            cache_len = prev_cache_len + new_seen_tokens

            if self.decoder.model.gradient_checkpointing and self.training:
                # noinspection PyProtectedMember
                decoder_hidden_states = self.decoder._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    decoder_hidden_states,  # hidden_states=,
                    attention_mask,  # attention_mask=,
                    position_ids,  # position_ids=,
                    past_key_values,  # past_key_value=,
                    False,  # output_attentions=,
                    True,  # use_cache=,
                    cache_position,  # cache_position=,
                    decoder_position_embeddings,  # position_embeddings=,
                    q_start_idx,  # q_start_idx=
                )[0]  # Shape: (input_ids.shape[0], input_ids.shape[1], hidden_dim)
            else:
                decoder_hidden_states = decoder_layer(
                    hidden_states=decoder_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=decoder_position_embeddings,
                    q_start_idx=q_start_idx,  # Indicates where to slice output
                    **flash_attn_kwargs,
                )[0]  # Shape: (input_ids.shape[0], input_ids.shape[1], hidden_dim)
            # Update decoder_hidden_states
            if q_start_idx > 0:
                decoder_hidden_states = torch.cat(
                    [
                        encoder_last_hidden_state,
                        decoder_hidden_states,
                    ],
                    dim=1,
                )

            if past_key_values is not None:
                # DynamicCache extends along sequence dimension by default;
                # truncate back to original cache len + encoder output length
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[
                    layer_idx
                ][..., :cache_len, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[
                    layer_idx
                ][..., :cache_len, :]
        decoder_hidden_states = self.decoder.model.norm(
            decoder_hidden_states[:, q_start_idx:, :]
        )
        logits = self.decoder.lm_head(decoder_hidden_states)
        return DecoderCausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            encoder_past_key_values=encoder_past_key_values,
            # Do not need to store encoder_last_hidden_state.
            # If it was passed in, then it has become part of the past_key_values cache.
        )


class LLMasEncoderDecoderShareKV(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_length: int,
        attn_backend: str = "sdpa",
        freeze_encoder: bool = False,
        reinit_encoder: bool = False,
        reinit_decoder: bool = False,
        tie_encoder_decoder_weights: bool = False,
        use_encoder_causal_mask: bool = False,
        num_encoder_layers: int = -1,
        num_decoder_layers: int = -1,
        keep_top_encoder_layers: bool = False,
        keep_top_decoder_layers: bool = False,
        use_gradient_checkpointing: bool = False,
        **llm_init_kwargs,
    ):
        assert not (tie_encoder_decoder_weights and reinit_decoder), (
            "Cannot tie encoder-decoder weights and reinitialize decoder."
        )
        assert not (tie_encoder_decoder_weights and freeze_encoder), (
            "Cannot freeze encoder weights when tying encoder-decoder weights."
        )
        super().__init__()
        self.use_encoder_causal_mask = use_encoder_causal_mask
        self.tie_encoder_decoder_weights = tie_encoder_decoder_weights

        if reinit_encoder:
            assert num_encoder_layers > 0
            encoder_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                num_hidden_layers=num_encoder_layers,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )
            self.encoder = AutoModelForCausalLM.from_config(encoder_config)
        else:
            self.encoder = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )
            assert num_encoder_layers <= len(self.encoder.model.layers), (
                f"Cannot keep {num_encoder_layers} layers. "
                f"Pre-trained model only has {len(self.encoder.model.layers)} layers."
            )
            num_encoder_layers = (
                len(self.encoder.model.layers)
                if num_encoder_layers == -1
                else num_encoder_layers
            )
            if keep_top_encoder_layers:
                self.encoder.model.layers = self.encoder.model.layers[
                    -num_encoder_layers:
                ]
            else:
                self.encoder.model.layers = self.encoder.model.layers[
                    :num_encoder_layers
                ]

        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if "embed_tokens" not in name:
                    param.requires_grad = False
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        if tie_encoder_decoder_weights:
            self.decoder = self.encoder
            num_decoder_layers = (
                len(self.decoder.model.layers)
                if num_decoder_layers == -1
                else num_decoder_layers
            )
            assert num_decoder_layers <= len(self.decoder.model.layers), (
                f"Cannot keep {num_decoder_layers} layers. "
                f"Pre-trained model only has {len(self.decoder.model.layers)} layers."
            )
            # Keep **top** layers when tying weights
            self.decoder_layer_idxs = list(range(len(self.encoder.model.layers)))[
                -num_decoder_layers:
            ]

        else:
            if reinit_decoder:
                assert num_decoder_layers > 0
                decoder_config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    num_hidden_layers=num_decoder_layers,
                    attn_implementation=attn_backend,
                    **llm_init_kwargs,
                )
                self.decoder = AutoModelForCausalLM(decoder_config)
            else:
                self.decoder = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                    **llm_init_kwargs,
                )
                assert num_decoder_layers <= len(self.decoder.model.layers), (
                    f"Cannot keep {num_decoder_layers} layers. "
                    f"Pre-trained model only has {len(self.decoder.layers)} layers."
                )
                if keep_top_decoder_layers:
                    self.decoder.model.layers = self.decoder.model.layers[
                        -num_decoder_layers:
                    ]
                else:
                    self.decoder.model.layers = self.decoder.model.layers[
                        :num_decoder_layers
                    ]
            del self.decoder.model.embed_tokens
            # Even for frozen encoder, ensure embedding tokens are trainable
            self.encoder.model.embed_tokens.requires_grad_(True)
            unused_self_attn_params = ["o_proj", "q_norm", "q_proj"]
            unused_layernorm_params = ["input_layernorm", "post_attention_layernorm"]
            for unused_param in unused_self_attn_params:
                if hasattr(self.encoder.model.layers[-1].self_attn, unused_param):
                    getattr(
                        self.encoder.model.layers[-1].self_attn, unused_param
                    ).requires_grad_(False)
            self.encoder.model.layers[-1].mlp.requires_grad_(False)
            self.encoder.model.norm.requires_grad_(False)
            for unused_param in unused_layernorm_params:
                if hasattr(self.encoder.model.layers[-1], unused_param):
                    getattr(self.encoder.model.layers[-1], unused_param).requires_grad_(
                        False
                    )
            # if in the original LM, the lm_head is weight-tied to embedding,
            # point decoder lm_head to encoder's (instead of initializing separately)
            if (
                self.encoder.lm_head.weight.data_ptr()
                == self.encoder.model.embed_tokens.weight.data_ptr()
            ):
                self.decoder.lm_head = self.encoder.lm_head
            else:
                del self.encoder.lm_head
            if use_gradient_checkpointing:
                self.decoder.gradient_checkpointing_enable()
        self.max_length = max_length

    def freeze_encoder(self):
        for p in self.encoder.model.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.model.parameters():
            p.requires_grad = True

    # noinspection PyUnusedLocal
    def forward(
        self,
        # Decoder inputs
        input_ids: torch.LongTensor,
        attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        encoder_last_hidden_state: Optional[torch.FloatTensor] = None,  # Not used
        # Encoder inputs
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[Union[torch.FloatTensor, BlockMask]] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_cache_position: Optional[torch.LongTensor] = None,
        encoder_past_key_values: Optional[DynamicCache] = None,  # Not used
        # Additional args
        fix_cache_length: bool = True,  # Not used; compatibility with other backbones
        return_updated_cache: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[CausalLMOutputWithPast, BaseModelOutputWithPast]:
        # Encode clean tokens
        if encoder_input_ids is not None:
            if self.use_encoder_causal_mask:
                encoder_attention_mask = None  # None --> enforces use of causal mask
            if encoder_cache_position is None and encoder_position_ids is not None:
                encoder_cache_position = encoder_position_ids[0]
            past_key_values = self.encoder.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
                cache_position=encoder_cache_position,
            ).past_key_values
            if return_updated_cache:
                # encoder_output.past_key_values now contains latest encoder input
                return BaseModelOutputWithPast(
                    past_key_values=past_key_values,
                )

        # Run decoder with xattn to clean token hidden states
        decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)
        if cache_position is None:
            if position_ids is not None:
                cache_position = position_ids[0]
            else:  # During training / validation position_ids are not provided
                cache_position = torch.arange(
                    decoder_hidden_states.shape[1],
                    device=decoder_hidden_states.device,
                )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        decoder_position_embeddings = self.decoder.model.rotary_emb(
            decoder_hidden_states, position_ids
        )

        if hasattr(self.decoder.model, "_update_causal_mask"):  # bc on transformers
            # noinspection PyProtectedMember
            attention_mask = self.decoder.model._update_causal_mask(
                attention_mask=attention_mask,
                input_tensor=decoder_hidden_states,
                cache_position=cache_position,
                past_key_values=past_key_values,
                output_attentions=False,
            )
        for decoder_layer in self.decoder.model.layers:
            layer_idx = decoder_layer.self_attn.layer_idx
            if (
                self.tie_encoder_decoder_weights
                and layer_idx not in self.decoder_layer_idxs
            ):
                continue
            # past_key_values gets updated in-place.
            # Record previous length to truncate after each layer forward
            if past_key_values is not None and len(past_key_values) > layer_idx:
                prev_cache_len = past_key_values[layer_idx][0].shape[-2]  # type: ignore
            else:
                prev_cache_len = 0

            decoder_hidden_states = decoder_layer(
                hidden_states=decoder_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=position_ids[0],
                position_embeddings=decoder_position_embeddings,
                **flash_attn_kwargs,
            )[0]  # Shape: (input_ids.shape[0], input_ids.shape[1], hidden_dim)

            if past_key_values is not None:
                # DynamicCache extends along sequence dimension by default;
                # truncate back to original cache len + encoder output length
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
        decoder_hidden_states = self.decoder.model.norm(decoder_hidden_states)
        logits = self.decoder.lm_head(decoder_hidden_states)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )
