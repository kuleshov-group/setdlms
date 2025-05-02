from typing import Union

import torch
from torch import Tensor, nn
from transformers import AutoConfig
from transformers.cache_utils import Cache
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
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        keep_every_n_encoder_layers: int = 1,
        keep_every_n_decoder_layers: int = 1,
        attn_backend: str = "sdpa",
        freeze_encoder: bool = False,
        reinit_decoder: bool = False,
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
        max_encoder_layers = n_encoder_layers * keep_every_n_encoder_layers
        if keep_every_n_encoder_layers < len(self.encoder.model.layers):
            encoder_layers_post_surgery = []
            for i, encoder_layer in enumerate(
                self.encoder.model.layers[:max_encoder_layers]
            ):
                if i % keep_every_n_encoder_layers == 0:
                    encoder_layers_post_surgery.append(encoder_layer)
            self.encoder.model.layers = nn.ModuleList(encoder_layers_post_surgery)
        del self.encoder.lm_head

        max_decoder_layers = n_decoder_layers * keep_every_n_decoder_layers
        if keep_every_n_decoder_layers < len(self.decoder.model.layers):
            decoder_layers_post_surgery = []
            for i, decoder_layer in enumerate(
                self.decoder.model.layers[
                    max_encoder_layers : max_encoder_layers + max_decoder_layers
                ]
            ):
                if i % keep_every_n_decoder_layers == 0:
                    decoder_layers_post_surgery.append(decoder_layer)
            self.decoder.model.layers = nn.ModuleList(decoder_layers_post_surgery)
        del self.decoder.model.embed_tokens
        self.block_size = block_size
        self.max_length = max_length

    def forward(
        self,
        input_ids: Tensor,  # for Decoder
        attention_mask: Union[Tensor, BlockMask],  # for Decoder
        encoder_input_ids: Tensor | None = None,  # for Encoder
        encoder_attention_mask: Union[Tensor, BlockMask] | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor:
        # TODO: not sure how to use this yet...
        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache()
        # if cache_position is None:
        #     past_seen_tokens = (
        #         past_key_values.get_seq_length() if past_key_values is not None else 0
        #     )
        #     cache_position = torch.arange(
        #         past_seen_tokens,
        #         past_seen_tokens + inputs_embeds.shape[1],
        #         device=inputs_embeds.device,
        #     )
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        #
        # causal_mask = self.llama._update_causal_mask(
        #     attention_mask,
        #     inputs_embeds,
        #     cache_position,
        #     past_key_values,
        #     output_attentions,
        # )

        # Encode clean tokens
        # encoder_hidden_states = self.llama.model.embed_tokens(encoder_input_ids)
        encoder_hidden_state = None
        if encoder_input_ids is not None:
            encoder_position_ids = torch.arange(
                encoder_input_ids.shape[-1], device=encoder_input_ids.device
            ).unsqueeze(0)
            # encoder_position_embeddings = self.llama.model.rotary_emb(
            #     encoder_hidden_states, encoder_position_ids
            # )
            if encoder_attention_mask is not None:
                encoder_attention_mask[:, None, ...].to(self.encoder.dtype)
            encoder_hidden_state = self.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                output_hidden_states=True,
            )[-1]

        # Run decoder with xattn to clean tokens
        decoder_hidden_states = self.encoder.model.embed_tokens(input_ids)
        decoder_position_ids = torch.arange(
            decoder_hidden_states.shape[1], device=input_ids.device
        ).unsqueeze(0)
        if encoder_hidden_state is not None:
            decoder_position_ids = torch.cat(
                (
                    decoder_position_ids,
                    torch.arange(
                        encoder_hidden_state.shape[1], device=input_ids.device
                    ).unsqueeze(0),
                ),
                dim=-1,
            ).unsqueeze(0)
        decoder_position_embeddings = self.decoder.model.rotary_emb(
            decoder_hidden_states, decoder_position_ids
        )
        attention_mask = attention_mask[:, None, ...].to(self.decoder.dtype)
        for i, decoder_layer in enumerate(self.decoder.model.layers):
            decoder_hidden_states = decoder_layer(
                hidden_states=decoder_hidden_states,
                encoder_hidden_states=encoder_hidden_state,
                attention_mask=attention_mask,
                position_ids=decoder_position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=decoder_position_embeddings,
                **flash_attn_kwargs,
            )[0]  # [:, : input_ids.shape[1], :]
        # Only keep logits for masked tokens
        decoder_hidden_states = self.decoder.model.norm(decoder_hidden_states)
        return self.decoder.lm_head(decoder_hidden_states)
