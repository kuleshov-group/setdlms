import math
import typing

import einops
from einops import rearrange
from functools import partial
try:
  import flash_attn
  import flash_attn.layers.rotary
except:
  pass
import huggingface_hub
import os
import omegaconf
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
try:
    from transformers.utils import use_kernel_forward_from_hub
except ImportError:
    # Fallback if decorator is not available
    def use_kernel_forward_from_hub(kernel_name):
        def decorator(cls):
            return cls
        return decorator

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, DynamicCache
  FLEX_ATTN_AVAILABLE = True
except:
  FLEX_ATTN_AVAILABLE = False
  BlockMask = None
  DynamicCache = None

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
  """
  Constructs the specialized block diffusion attention mask for training
  composed of three masks:
  - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
  - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
  - **Block Causal Mask (M_BC)**: Attention to update x0

  Args:
      b, h: Batch and head indices (ignored for mask logic).
      q_idx, kv_idx: Query and Key indices.
      seq_len: Total sequence length.
      block_size: Defines the block structure.

  Returns:
      A boolean attention mask.
  """

  # Indicate whether token belongs to xt or x0
  x0_flag_q = (q_idx >= n)
  x0_flag_kv = (kv_idx >= n)

  # Compute block indices
  block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
  block_kv = torch.where(x0_flag_kv == 1,
                        (kv_idx - n) // block_size,
                        kv_idx // block_size)

  # **1. Block Diagonal Mask (M_BD) **
  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

  # **2. Offset Block-Causal Mask (M_OBC) **
  offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
  )

  # **3. Block-Causal Mask (M_BC) **
  block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

  # **4. Combine Masks **
  return block_diagonal | offset_block_causal | block_causal

def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift

def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)

def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)

def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, position_ids):
    seq_len = position_ids.shape[-1]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = position_ids.type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(position_ids.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v

def apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)

def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
  # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
  # where the 3 represents Q, K, V packed in that order
  attention_output = F.scaled_dot_product_attention(
    query=q.transpose(1, 2),
    key=k.transpose(1, 2),
    value=v.transpose(1, 2),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False)
  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]

def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
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
      - math.log(max_period)
      * torch.arange(start=0, end=half).to(t.dtype).to(t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockCausal(nn.Module):
  def __init__(self, n, dim, n_heads, mlp_ratio=4, dropout=0.1, max_seqlen=1024, adaLN=False, cond_dim=None, attn_backend='flash_attn', norm_type='layernorm'):
    super().__init__()
    self.n_heads = n_heads
    self.max_seqlen = max_seqlen
    self.n = n

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend
    self.norm_type = norm_type
  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin):
    qkv = self.attn_qkv(x)
      
    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn':
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
          
    return qkv

  def cross_attn(self, qkv, mask=None, causal=False):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      is_causal=causal,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def forward(self,
              x,
              rotary_cos_sin,
              c=None,
              causal=True,
              mask=None,
              **kwargs):
    del kwargs
    batch_size, seq_len = x.shape[0], x.shape[1]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = rearrange(
        self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
        ).chunk(6, dim=-1)

    # attention operation
    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)
    
    qkv = self.get_qkv(x, rotary_cos_sin)
    if self.attn_backend == 'flash_attn':
      qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len,
        step=seq_len, dtype=torch.int32, device=qkv.device)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0.0, causal=True)
      x = einops.rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    else:
      x = self.cross_attn(qkv, causal=causal, mask=mask)
      
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x),
        None,
        gate_msa,
        x_skip,
        self.dropout)
      # mlp operation
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class DDiTBlock(nn.Module):
  def __init__(self, n, dim, n_heads, adaLN,
               latent_dim=None, cond_dim=None,
               latent_conditioning=-1, mlp_ratio=4,
               dropout=0.1, block_size=1, max_seqlen=1024, attn_backend='flash_attn', norm_type='layernorm'):
    super().__init__()
    self.max_seqlen = max_seqlen
    self.n = n
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.latent_conditioning = latent_conditioning
    self.block_size = block_size

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)

    if norm_type == "qknorm":
      self.q_norm = LayerNorm(dim // n_heads)
      self.k_norm = LayerNorm(dim // n_heads)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.cache_idx = 0

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend
    self.norm_type = norm_type
  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin):
    qkv = self.attn_qkv(x)

    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)

    if self.norm_type == "qknorm":
      q_states = self.q_norm(qkv[:, :, 0])
      k_states = self.k_norm(qkv[:, :, 1])
      qkv = torch.stack([q_states, k_states, qkv[:, :, 2]], dim=2)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn':
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv
  
  def attn_mlp(self, x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x),
        None,
        gate_msa,
        x_skip,
        self.dropout)
      # mlp operation
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x

  def cross_attn(self, qkv, mask=None, causal=False):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=causal,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def cross_attn_flex(self, qkv, mask=None):
    qkv = rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
    x = fused_flex_attention(
      qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
    x = rearrange(x, 'b h s d -> b s (h d)')
    return x

  def forward(self,
              x,
              rotary_cos_sin,
              c,
              causal=False,
              mask=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = rearrange(
        self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
        ).chunk(6, dim=-1)

    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)

    qkv = self.get_qkv(x, rotary_cos_sin)
    # attention
    if self.attn_backend == 'flash_attn' and mask is None:
      qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=causal)
      x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)     
    elif self.attn_backend == 'flex_attention' and FLEX_ATTN_AVAILABLE:
      x = self.cross_attn_flex(qkv, mask=mask)
    elif self.attn_backend == 'sdpa':
      x = self.cross_attn(qkv, mask=mask, causal=causal)
    else:
      raise ValueError('Unknown attention backend')
    x = self.attn_mlp(x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip)
    return x
   
class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim, 
               adaLN, tie_word_embeddings=False, norm_type='layernorm'):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.tie_word_embeddings = tie_word_embeddings
    self.norm_type = norm_type

  def forward(self, x, c):
    x = self.norm_final(x)
    if c is not None:
      if c.shape[0] == x.shape[0]:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      else:
        shift, scale = rearrange(
          self.adaLN_modulation(c), '(b h) d -> b h d', b=x.shape[0]).chunk(2, dim=-1)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self,
        length: int,
        causal_attention: bool,
        adaln: bool,
        hidden_size: int,
        cond_dim: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
        tie_word_embeddings: bool,
        vocab_size: int,
        block_size: int,
        attn_backend: str,
        norm_type: str,
        pretrained_model_name_or_path: str):
    super().__init__()
    self.causal = causal_attention
    self.n = length
    self.adaLN = adaln
    self.vocab_size = vocab_size
    self.block_size = block_size
    dim = hidden_size
    cond_dim = cond_dim
    self.n_heads = n_heads
    self.norm_type = norm_type
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    if self.adaLN == True:
      self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // n_heads)
    self.attn_backend = attn_backend
    self.max_seqlen = 1024

    blocks = []
    for _ in range(num_layers):
      if self.causal:
        block = DDiTBlockCausal(
          n=length,
          dim=dim,
          n_heads=n_heads,
          dropout=dropout,
          adaLN=self.adaLN,
          cond_dim=cond_dim,
          attn_backend=self.attn_backend,
          norm_type=self.norm_type)
      else:
        block = DDiTBlock(
          n=length,
          dim=dim,
          n_heads=n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=dropout,
          block_size=self.block_size,
          attn_backend=self.attn_backend,
          max_seqlen=self.max_seqlen,
          norm_type=self.norm_type)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)
    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN,
      tie_word_embeddings=tie_word_embeddings,
      norm_type=self.norm_type)
    if pretrained_model_name_or_path is not None:
      state_dict = torch.load(pretrained_model_name_or_path, weights_only=False)
      state_dict = state_dict["state_dict"]
      # replace all keys
      new_state_dict = {}
      for key in state_dict.keys():
        new_key = key
        if "backbone." in key:
          new_key = key.replace("backbone.", "")
        if "_orig_mod." in new_key:
          new_key = new_key.replace("_orig_mod.", "")
        new_state_dict[new_key] = state_dict[key]
      del state_dict
      self.load_state_dict(new_state_dict, strict=False)
    print(self)
  

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference
    
  def forward(self,
    input_ids: torch.LongTensor,
    attention_mask: torch.FloatTensor | Any | None = None,
    position_ids: torch.LongTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    past_key_values: Any | None = None,
    fix_cache_length: bool = False,  # False for AR, True for diffusion models
    return_updated_cache=False,
    sigma=None,
    **kwargs,
  ) -> CausalLMOutputWithPast | BaseModelOutputWithPast:
    x = self.vocab_embed(input_ids)
    if sigma is None:
      if self.adaLN:
        sigma = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        t_cond = F.silu(self.sigma_map(sigma))
      else:
        t_cond = None

    if sigma is not None:
      t_cond = F.silu(self.sigma_map(sigma))

    if position_ids is None:
      position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).to(input_ids.dtype)
    else:
      position_ids = position_ids[0]
    rotary_cos_sin = self.rotary_emb(position_ids)

    if self.causal:
      attention_mask = None
    
    for i in range(len(self.blocks)):
      x = self.blocks[i](
        x,
        rotary_cos_sin,
        c=t_cond,
        causal=self.causal,
        mask=attention_mask,)
    x = self.output_layer(x, t_cond)
    return CausalLMOutputWithPast(
      logits=x,
      past_key_values=None)