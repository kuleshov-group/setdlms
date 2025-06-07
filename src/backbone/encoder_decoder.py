import torch
from torch import nn
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging

from src.backbone.custom_modeling_qwen3 import CustomQwen3ForCausalLM

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None


logger = logging.get_logger(__name__)


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
        keep_bottom_n_encoder_layers: int = -1,
        keep_top_n_decoder_layers: int = -1,
    ):
        assert (
            keep_top_n_decoder_layers <= keep_bottom_n_encoder_layers
            or keep_bottom_n_encoder_layers == -1
        ), (
            "Cannot keep more decoder layers than encoder layers: "
            f"{keep_top_n_decoder_layers=} > {keep_bottom_n_encoder_layers=}."
        )
        assert not (tie_encoder_decoder_weights and reinit_decoder), (
            "Cannot tie encoder-decoder weights and reinitialize decoder."
        )
        super().__init__()
        self.use_encoder_causal_mask = use_encoder_causal_mask
        self.tie_encoder_decoder_weights = tie_encoder_decoder_weights

        if reinit_encoder:
            encoder_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_backend,
            )
            self.encoder = CustomQwen3ForCausalLM.from_config(encoder_config)
        else:
            self.encoder = CustomQwen3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_backend,
            )
        keep_top_n_decoder_layers = (
            len(self.encoder.model.layers)
            if keep_top_n_decoder_layers == -1
            else keep_top_n_decoder_layers
        )
        keep_bottom_n_encoder_layers = (
            len(self.encoder.model.layers)
            if keep_bottom_n_encoder_layers == -1
            else keep_bottom_n_encoder_layers
        )
        self.encoder.model.layers = self.encoder.model.layers[
            :keep_bottom_n_encoder_layers
        ]
        self.decoder_layers_to_keep = list(range(keep_bottom_n_encoder_layers))[
            -keep_top_n_decoder_layers:
        ]

        # freeze encoder layers
        if freeze_encoder:
            assert use_encoder_causal_mask
            for name, param in self.encoder.named_parameters():
                if "embed_tokens" not in name:
                    param.requires_grad = False

        # tie encoder and decoder weights
        if tie_encoder_decoder_weights:
            assert not freeze_encoder
            self.decoder = self.encoder

        else:
            # reinitialize decoder layers
            if reinit_decoder:
                decoder_config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                )
                self.decoder = CustomQwen3ForCausalLM(decoder_config)
            else:
                self.decoder = CustomQwen3ForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                )
            del self.decoder.model.embed_tokens
            decoder_layers_post_surgery = []
            for decoder_layer_idx in self.decoder_layers_to_keep:
                decoder_layers_post_surgery.append(
                    self.decoder.model.layers[decoder_layer_idx]
                )
            self.decoder.model.layers = nn.ModuleList(decoder_layers_post_surgery)
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
            # if lm head is weight-tied to embedding, point decoder lm head to encoder
            # (instead of initializing a separate lm head)
            if (
                self.encoder.lm_head.weight.data_ptr()
                == self.encoder.model.embed_tokens.weight.data_ptr()
            ):  # noqa: E501
                self.decoder.lm_head = self.encoder.lm_head
            else:
                del self.encoder.lm_head
        self.max_length = max_length

    def forward(
        self,
        input_ids: torch.LongTensor,  # for Decoder
        attention_mask: torch.FloatTensor | BlockMask | None = None,  # for Decoder
        encoder_input_ids: torch.LongTensor | None = None,  # for Encoder
        encoder_attention_mask: torch.FloatTensor | BlockMask | None = None,
        past_key_values: DynamicCache | None = None,
        position_ids: torch.LongTensor | None = None,
        encoder_position_ids: torch.LongTensor | None = None,
        return_past_key_values: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor | DynamicCache:
        if past_key_values is None:
            past_key_values = DynamicCache()

        # Encode clean tokens
        if encoder_input_ids is not None:
            if encoder_position_ids is None:
                encoder_position_ids = torch.arange(
                    encoder_input_ids.shape[-1], device=encoder_input_ids.device
                ).unsqueeze(0)
            if self.use_encoder_causal_mask:
                encoder_attention_mask = None  # must use causal mask
            past_key_values = self.encoder.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
                cache_position=encoder_position_ids[0],
            ).past_key_values
            if return_past_key_values:
                return past_key_values

        # Run decoder with xattn to clean tokens
        decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0)
        decoder_position_embeddings = self.decoder.model.rotary_emb(
            decoder_hidden_states, position_ids
        )
        # noinspection PyProtectedMember
        attention_mask = self.decoder.model._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=decoder_hidden_states,
            cache_position=position_ids[0],
            past_key_values=past_key_values,
            output_attentions=False,
        )

        for decoder_layer in self.decoder.model.layers:
            layer_idx = decoder_layer.self_attn.layer_idx
            if layer_idx not in self.decoder_layers_to_keep:
                continue
            if past_key_values is not None and len(past_key_values) == len(
                self.encoder.model.layers
            ):
                prev_cache_len = past_key_values[layer_idx][0].shape[-2]  # type: ignore
            else:
                prev_cache_len = 0

            # TODO maybe adopt gradient checkpointing from transformers
            # Cross-attend to encoder kvs
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
        decoder_hidden_states = self.decoder.model.norm(decoder_hidden_states)
        decoded_tokens = self.decoder.lm_head(decoder_hidden_states)
        return decoded_tokens
