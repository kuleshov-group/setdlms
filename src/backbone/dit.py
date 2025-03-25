import math
from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------


def multi_head_attention(
    q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False
) -> Tensor:
    # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
    # where the 3 represents Q, K, V packed in that order
    attention_output = F.scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k.transpose(1, 2),
        value=v.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    # [batch_size, seq_len, num_heads, head_dim]
    attention_output = attention_output.transpose(1, 2)
    return einops.rearrange(attention_output, "b s h d -> b s (h d)")


def modulate(x: Tensor, *, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


# --------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: Tensor) -> Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            t = t.to(dtype=torch.float32)
            t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def rotary_embedding(
    seq_len: int,
    base: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    # dims are: batch, seq_len, qkv, head, dim
    cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
    sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
    # This makes the transformation on v an identity.
    cos[:, :, 2, :, :].fill_(1.0)
    sin[:, :, 2, :, :].fill_(0.0)
    return cos, sin


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.dim = dim

    def forward(self, x: Tensor, seq_dim: int = 1) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type="cuda", enabled=False):
            return rotary_embedding(
                x.shape[seq_dim],
                base=self.base,
                dim=self.dim,
                device=x.device,
                dtype=torch.float32,
            )


# noinspection LongLine
def rotate_half(x, interleaved=False):
    # Copied from: https://github.com/Dao-AILab/flash-attention/blob/a09abcd32d3cae4d83b313446e887f38d02b799f/flash_attn/layers/rotary.py#L11  # noqa: E501
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return einops.rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


# noinspection LongLine
def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    # Copied from: https://github.com/Dao-AILab/flash-attention/blob/a09abcd32d3cae4d83b313446e887f38d02b799f/flash_attn/layers/rotary.py#L20  # noqa: E501
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = einops.repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = einops.repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def split_and_apply_rotary_pos_emb(
    qkv: Tensor, rotary_cos_sin: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor, Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
        cos, sin = rotary_cos_sin
        qkv = qkv.to(dtype=torch.float32)
        cos = cos.type_as(qkv)
        sin = sin.type_as(qkv)
        cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
        sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
        q, k, v = qkv.chunk(3, dim=2)
        q = apply_rotary_emb_torch(q.squeeze(dim=2), cos, sin)
        k = apply_rotary_emb_torch(k.squeeze(dim=2), cos, sin)
        v = v.squeeze(dim=2)
    return q, k, v


# --------------------------------------------------------------------------------
# Core Model
# --------------------------------------------------------------------------------


class DDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        cond_dim: int,
        causal: bool,
        use_adaln: bool,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.causal = causal
        self.use_adaln = use_adaln
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * hidden_size, hidden_size, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)

        if use_adaln:
            self.adaln = nn.Linear(cond_dim, 6 * hidden_size)
            self.adaln.weight.data.zero_()
            self.adaln.bias.data.zero_()

    def forward(
        self, x: Tensor, rotary_cos_sin: tuple[Tensor, Tensor], c: Tensor | None = None
    ) -> Tensor:
        """Forward pass for a single DDiT block.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size)
            rotary_cos_sin: Tuple with rotary cosine and sine tensors, each of shape
                (batch_size, sequence_length, 3 (qkv), n_heads, hidden_size // n_heads)
            c: Timestamp embedding
        """
        x_skip = x
        x = self.norm1(x)

        if self.use_adaln:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(
                c
            ).chunk(6, dim=-1)
            x = modulate(x, shift=shift_msa, scale=scale_msa)
        else:
            shift_msa, scale_msa, shift_mlp, scale_mlp = None, None, None, None
            gate_msa, gate_mlp = 1.0, 1.0

        qkv = einops.rearrange(
            self.attn_qkv(x),
            "b s (three h d) -> b s three h d",
            three=3,
            h=self.n_heads,
        )
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)

        x = multi_head_attention(q, k, v, is_causal=self.causal)

        x = self.attn_out(x)
        x = gate_msa * self.dropout1(x) + x_skip
        if self.use_adaln:
            y = self.mlp(modulate(self.norm2(x), shift=shift_mlp, scale=scale_mlp))
        else:
            y = self.mlp(self.norm2(x))
        x = gate_mlp * self.dropout2(y) + x

        return x


class DDitFinalLayer(nn.Module):
    def __init__(
        self, hidden_size: int, out_channels: int, cond_dim: int, use_adaln: bool
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaln.weight.data.zero_()
            self.adaln.bias.data.zero_()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        if self.use_adaln:
            shift, scale = self.adaln(c).chunk(2, dim=-1)
            x = modulate(self.norm_final(x), shift=shift, scale=scale)
            return self.linear(x)
        return self.linear(self.norm_final(x))


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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.causal = causal
        self.use_adaln = use_adaln
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.timestep_embed = (
            TimestepEmbedding(cond_dim) if use_adaln and not causal else None
        )
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

    def forward(
        self, input_ids: Tensor, noise: Tensor | None = None, **_: Any
    ) -> Tensor:
        """Forward pass for DIT model.

        Args:
            input_ids: Input ids of shape (batch_size, sequence_length)
            noise: (Optional) Noise float tensor of shape (batch_size,)
        """
        x = self.vocab_embed(input_ids)
        c = (
            None
            if self.causal or not self.use_adaln or noise is None
            else self.activation(self.timestep_embed(noise))
        )
        rotary_cos_sin = self.rotary_embed(x)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)

        x = self.output_layer(x, c)

        return x
