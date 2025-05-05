from typing import Union

import torch
from torch import Tensor, nn
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging

try:
    from torch.nn.attention.flex_attention import BlockMask
except ModuleNotFoundError:
    BlockMask = None


from src.backbone.custom_modeling_llama import LlamaForCausalLM
from src.backbone.custom_modeling_qwen3 import Qwen3ForCausalLM

logger = logging.get_logger(__name__)


class LlamaAsEncoderDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        block_size: int,
        max_length: int,
        keep_every_n_encoder_layers: int = 1,
        keep_every_n_decoder_layers: int = 1,
        attn_backend: str = "sdpa",
        freeze_encoder: bool = False,
        reinit_decoder: bool = False,
        recompute_kvs: bool = True,
    ):
        assert keep_every_n_encoder_layers <= keep_every_n_decoder_layers, (
            "Cannot remove more encoder than decoder layers."
        )
        super().__init__()
        if "Qwen" in pretrained_model_name_or_path:
            self.encoder = Qwen3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_backend,
            )

            # freeze encoder layers
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

            # reinitialize decoder layers
            if reinit_decoder:
                decoder_config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                )
                self.decoder = Qwen3ForCausalLM(decoder_config)
            else:
                self.decoder = Qwen3ForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                )
        elif "Llama" in pretrained_model_name_or_path:
            self.encoder = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_backend,
            )
            # freeze encoder layers
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

            # reinitialize decoder layers
            if reinit_decoder:
                decoder_config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                )
                self.decoder = LlamaForCausalLM(decoder_config)
            else:
                self.decoder = LlamaForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    attn_implementation=attn_backend,
                )
        else:
            raise ValueError(
                f"Unsupported model type: {pretrained_model_name_or_path}. "
                "Please use either Qwen or Llama."
            )

        # delete layers from encoder / decoder
        if keep_every_n_encoder_layers < len(self.encoder.model.layers):
            encoder_layers_post_surgery = []
            for i, encoder_layer in enumerate(self.encoder.model.layers):
                if i % keep_every_n_encoder_layers == 0:
                    encoder_layers_post_surgery.append(encoder_layer)
            self.encoder.model.layers = nn.ModuleList(encoder_layers_post_surgery)
        del self.encoder.lm_head

        if keep_every_n_decoder_layers < len(self.decoder.model.layers):
            decoder_layers_post_surgery = []
            for i, decoder_layer in enumerate(self.decoder.model.layers):
                if i % keep_every_n_decoder_layers == 0:
                    decoder_layers_post_surgery.append(decoder_layer)
            self.decoder.model.layers = nn.ModuleList(decoder_layers_post_surgery)
        del self.decoder.model.embed_tokens
        self.block_size = block_size
        self.max_length = max_length
        self.freeze_encoder = freeze_encoder
        self.recompute_kvs = recompute_kvs

    def forward(
        self,
        input_ids: Tensor,  # for Decoder
        attention_mask: Union[Tensor, BlockMask],  # for Decoder
        encoder_input_ids: Tensor | None = None,  # for Encoder
        encoder_attention_mask: Union[Tensor, BlockMask] | None = None,
        past_key_values: DynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        position_ids: Tensor | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor:
        decoder_use_cache_flag = use_cache or not self.recompute_kvs
        if decoder_use_cache_flag and past_key_values is None:
            past_key_values = DynamicCache()
            cache_position = position_ids

        # Encode clean tokens
        encoder_hidden_state = None
        if encoder_input_ids is not None and encoder_input_ids.shape[1] > 0:
            encoder_position_ids = torch.arange(
                encoder_input_ids.shape[-1], device=encoder_input_ids.device
            ).unsqueeze(0)
            if self.freeze_encoder:
                encoder_attention_mask = None  # must use causal mask
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask[:, None, ...].to(
                    self.encoder.dtype
                )
            # If we don't recompute KVs, we need to cache KVs from encoder
            encoder_hidden_state = self.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                output_hidden_states=True,
                use_cache=not self.recompute_kvs,
                past_key_values=(past_key_values if not self.recompute_kvs else None),
                cache_position=(cache_position if not self.recompute_kvs else None),
            )[-1]
        # Run decoder with xattn to clean tokens
        decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0)
            position_ids = torch.cat(
                (
                    position_ids,
                    torch.arange(
                        encoder_hidden_state.shape[1], device=input_ids.device
                    ).unsqueeze(0),
                ),
                dim=-1,
            )
        decoder_position_embeddings = self.decoder.model.rotary_emb(
            decoder_hidden_states, position_ids
        )
        attention_mask = attention_mask[:, None, ...].to(self.decoder.dtype)
        for i, decoder_layer in enumerate(self.decoder.model.layers):
            decoder_hidden_states = decoder_layer(
                hidden_states=decoder_hidden_states,
                encoder_hidden_states=encoder_hidden_state,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=decoder_use_cache_flag,
                cache_position=cache_position,
                position_embeddings=decoder_position_embeddings,
                **flash_attn_kwargs,
            )[0]
        if use_cache:
            return past_key_values
        # Only keep logits for masked tokens
        decoder_hidden_states = self.decoder.model.norm(decoder_hidden_states)
        decoded_tokens = self.decoder.lm_head(decoder_hidden_states)
        return decoded_tokens
