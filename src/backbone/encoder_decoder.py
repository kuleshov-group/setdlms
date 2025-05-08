from typing import Union

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging

try:
    from torch.nn.attention.flex_attention import BlockMask
except ModuleNotFoundError:
    BlockMask = None


logger = logging.get_logger(__name__)


class LLMasEncoderDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_length: int,
        keep_every_n_encoder_layers: int = 1,
        keep_every_n_decoder_layers: int = 1,
        attn_backend: str = "sdpa",
        freeze_encoder: bool = False,
        reinit_decoder: bool = False,
        tie_encoder_decoder_weights: bool = False,
        use_encoder_causal_mask: bool = False,
    ):
        assert keep_every_n_encoder_layers <= keep_every_n_decoder_layers, (
            "Cannot remove more encoder than decoder layers."
        )
        assert keep_every_n_decoder_layers % keep_every_n_encoder_layers == 0, (
            "Encoder-Decoder layers are mismatched; cross attention will not work."
        )
        super().__init__()
        self.encoder = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            attn_implementation=attn_backend,
        )

        # freeze encoder layers
        if freeze_encoder:
            assert use_encoder_causal_mask
            for param in self.encoder.parameters():
                param.requires_grad = False

        # tie encoder and decoder weights
        if tie_encoder_decoder_weights:
            assert not freeze_encoder
            self.decoder = self.encoder
            self.keep_every_n_decoder_layers = (
                keep_every_n_decoder_layers // keep_every_n_encoder_layers
            )
        else:
            self.keep_every_n_decoder_layers = keep_every_n_decoder_layers
            # reinitialize decoder layers
            if reinit_decoder:
                decoder_config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                )
                self.decoder = AutoModelForCausalLM(decoder_config)
            else:
                self.decoder = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                )

        # delete layers from encoder / decoder
        if keep_every_n_encoder_layers > 1:
            encoder_layers_post_surgery = []
            for i, encoder_layer in enumerate(self.encoder.model.layers):
                if (i + 1) % keep_every_n_encoder_layers == 0:
                    encoder_layers_post_surgery.append(encoder_layer)
            self.encoder.model.layers = nn.ModuleList(encoder_layers_post_surgery)
        if keep_every_n_decoder_layers > 1 and not tie_encoder_decoder_weights:
            decoder_layers_post_surgery = []
            for i, decoder_layer in enumerate(self.decoder.model.layers):
                if (i + 1) % keep_every_n_decoder_layers == 0:
                    decoder_layers_post_surgery.append(decoder_layer)
            self.decoder.model.layers = nn.ModuleList(decoder_layers_post_surgery)
        self.keep_every_n_encoder_layers = keep_every_n_encoder_layers
        self.use_encoder_causal_mask = use_encoder_causal_mask
        self.tie_encoder_decoder_weights = tie_encoder_decoder_weights
        if not tie_encoder_decoder_weights:
            del self.decoder.model.embed_tokens
            del self.decoder.model.norm
            del self.decoder.model.rotary_emb
            # if lm head is weight-tied to embedding, point decoder lm head to encoder
            # (instead of initializing a separate lm head)
            if "lm_head.weight" not in dict(self.encoder.named_parameters()):
                self.decoder.lm_head = self.encoder.lm_head
            else:
                del self.encoder.lm_head
        self.max_length = max_length

    def forward(
        self,
        input_ids: Tensor,  # for Decoder
        attention_mask: Union[Tensor, BlockMask],  # for Decoder
        encoder_input_ids: Tensor | None = None,  # for Encoder
        encoder_attention_mask: Union[Tensor, BlockMask] | None = None,
        past_key_values: DynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: Tensor | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor:
        if past_key_values is None:
            past_key_values = DynamicCache()
            cache_position = position_ids

        # Encode clean tokens
        if encoder_input_ids is not None:
            encoder_position_ids = torch.arange(
                encoder_input_ids.shape[-1], device=encoder_input_ids.device
            ).unsqueeze(0)
            if self.use_encoder_causal_mask:
                encoder_attention_mask = None  # must use causal mask
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask[:, None, ...].to(
                    self.encoder.dtype
                )
                min_dtype = torch.finfo(self.encoder.dtype).min
                encoder_attention_mask = torch.where(
                    encoder_attention_mask == 0.0, min_dtype, 0.0
                )
            past_key_values = self.encoder.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
                cache_position=cache_position,
            ).past_key_values
            if input_ids is None:
                return past_key_values

        # Run decoder with xattn to clean tokens
        decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0)
        decoder_position_embeddings = self.encoder.model.rotary_emb(
            decoder_hidden_states, position_ids
        )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + decoder_hidden_states.shape[1],
                device=decoder_hidden_states.device,
            )

        attention_mask = attention_mask[:, None, ...].to(self.decoder.dtype)
        min_dtype = torch.finfo(self.encoder.dtype).min
        attention_mask = torch.where(attention_mask == 0.0, min_dtype, 0.0)
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
                and (layer_idx + 1) % self.keep_every_n_decoder_layers != 0
            ):
                continue
            if past_key_values is not None:
                prev_cache_len = past_key_values[layer_idx][0].shape[-2]
            else:
                prev_cache_len = 0

            # TODO maybe adopt gradient checkpointing from transformers
            # cross-attend to encoder kvs
            decoder_hidden_states = decoder_layer(
                hidden_states=decoder_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=decoder_position_embeddings,
                **flash_attn_kwargs,
            )[0]
            if past_key_values is not None:
                # DynamicCache extends along sequence dimension by default
                # truncating back to original, encoder output length
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[
                    layer_idx
                ][..., :prev_cache_len, :]
        decoder_hidden_states = self.encoder.model.norm(decoder_hidden_states)
        decoded_tokens = self.decoder.lm_head(decoder_hidden_states)
        return decoded_tokens
