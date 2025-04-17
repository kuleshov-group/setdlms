from typing import Union

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging

try:
    from torch.nn.attention.flex_attention import BlockMask
except ModuleNotFoundError:
    BlockMask = None


logger = logging.get_logger(__name__)


class LlamaAsEncoderDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_layers_for_encoder: int,
        block_size: int,
        max_length: int,
    ):
        super().__init__()
        # TODO: consider init of decoder layers from scratch
        self.llama = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )  # TODO: try something like this:, attn_implementation="flex")
        self.num_layers_for_encoder = num_layers_for_encoder
        self.block_size = block_size
        self.max_length = max_length

    def forward(
        self,
        encoder_input_ids: Tensor,
        decoder_input_ids: Tensor,
        encoder_attention_mask: Union[Tensor, BlockMask],
        decoder_attention_mask: Union[Tensor, BlockMask],
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
        encoder_hidden_states = self.llama.model.embed_tokens(encoder_input_ids)
        encoder_position_ids = torch.arange(
            encoder_input_ids.shape[-1], device=encoder_input_ids.device
        ).unsqueeze(0)
        encoder_position_embeddings = self.llama.model.rotary_emb(
            encoder_hidden_states, encoder_position_ids
        )
        for encoder_layer in self.llama.model.layers[: self.num_layers_for_encoder]:
            encoder_hidden_states = encoder_layer(
                encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                encoder_position_ids=encoder_position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=encoder_position_embeddings,
                **flash_attn_kwargs,
            )[0]

        # Run decoder with xattn to clean tokens
        decoder_inputs_embeds = self.llama.model.embed_tokens(decoder_input_ids)
        decoder_hidden_states = torch.cat(
            (decoder_inputs_embeds, encoder_hidden_states), dim=-1
        )
        decoder_position_ids = torch.cat(
            (
                torch.arange(
                    decoder_inputs_embeds.shape[1], device=decoder_input_ids.device
                ),
                torch.arange(
                    encoder_hidden_states.shape[1], device=decoder_input_ids.device
                ),
            ),
            dim=-1,
        ).unsqueeze(0)
        decoder_position_embeddings = self.llama.model.rotary_emb(
            decoder_hidden_states, decoder_position_ids
        )
        for decoder_layer in self.llama.model.layers[self.num_layers_for_encoder :]:
            decoder_hidden_states = decoder_layer(
                decoder_hidden_states,
                attention_mask=decoder_attention_mask,
                encoder_position_ids=decoder_position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=decoder_position_embeddings,
                **flash_attn_kwargs,
            )[0]

        # Only keep logits for masked tokens
        decoder_hidden_states = [..., decoder_inputs_embeds.shape[1]]
        decoder_hidden_states = self.llama.model.norm(decoder_hidden_states)
        return self.llama.lm_head(decoder_hidden_states)
