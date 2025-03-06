from typing import Any

import torch
from torch import nn

from src.backbone.modules.dit_modules import (
    DDiTBlock,
    DDitFinalLayer,
    RotaryEmbedding,
    TimestepEmbedding,
)


class DIT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        causal: bool,
        use_adaln: bool,
        hidden_size: int = 768,
        cond_dim: int = 128,
        n_blocks: int = 12,
        n_heads: int = 12,
        scale_by_sigma: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.causal = causal
        self.use_adaln = use_adaln
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.timestep_embed = TimestepEmbedding(cond_dim)
        self.rotary_embed = RotaryEmbedding(hidden_size // n_heads)
        self.activation = nn.SiLU()

        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                DDiTBlock(
                    causal=causal,
                    use_adaln=use_adaln,
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            use_adaln=use_adaln,
            hidden_size=hidden_size,
            out_channels=vocab_size,
            cond_dim=cond_dim,
        )
        self.scale_by_sigma = scale_by_sigma

    def forward(
        self, input_ids: torch.Tensor, noise: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        """Forward pass for DIT model.

        Args:
            input_ids: Input ids of shape (batch_size, sequence_length)
            noise: Noise float tensor of shape (batch_size,)
        """
        x = self.vocab_embed(input_ids)
        c = None if self.causal else self.activation(self.timestep_embed(noise))
        rotary_cos_sin = self.rotary_embed(x)

        for block in self.blocks:
            x = block(x, rotary_cos_sin, c)
        x = self.output_layer(x, c)

        return x
