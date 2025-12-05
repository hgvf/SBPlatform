"""
Attention Blocks
"""

import torch
import torch.nn as nn
import math
from ..registry import register_attention, register_block

@register_attention("self_attention_1d")
class SelfAttn1D(nn.Module):
  """
  標準 1D self-attention
  """

  def __init__(
    self,
    channels: int,
    num_heads: int = 4
  ):
    super(SelfAtten1D, self).__init__()

    self.num_heads = num_heads
    self.channels = channels
    assert channels & num_heads == 0

    self.norm = GroupNorm(8, channels)
    self.qkv = nn.Linear(channels, channels * 3)
    self.proj = nn.Linear(channels, channels)

  def forward(
    self,
    x: torch.FloatTensor
  ) -> torch.FloatTensor:
    
    B, L, C = x.shape
    residual = x

    # GroupNorm needs (B, C, L)
    x = self.norm(x.permute(0, 2, 1))

    # Linear needs (B, L, C)
    qkv = self.qkv(x.permute(0, 2, 1))

    q, k, v = qkv.chunk(3, dim=-1)

    # Reshape for multi-head
    q = q.view(B, self.num_heads, L, C // self.num_heads)
    k = k.view(B, self.num_heads, L, C // self.num_heads)
    v = v.view(B, self.num_heads, L, C // self.num_heads)

    # Attention
    scale = (C // self.num_heads) ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)

    out = torch.matmul(attn, v)
    out = self.proj(out)

    return out + residual

@register_attention("cross_attention")
class CrossAttn1D(nn.Module):
  """
  Cross attention for conditional generation
  """

  def __init__(
    self, 
    d_q: int,
    d_k: int,
    num_heads: int = 4
  ):
    """
    Args:
      d_q: Dimension of query.
      d_k: Dimension of context.
      num_heads: Number of heads.
    """

    super(CrossAttn1D, self).__init__()

    self.num_heads = num_heads
    self.d_q = d_q

    self.norm_query = GroupNorm(8, d_q)
    self.norm_context = nn.LayerNorm(d_k)

    self.q = nn.Linear(d_q, d_q)
    self.k = nn.Linear(d_k, d_k)
    self.v = nn.Linear(d_k, d_k)
    self.proj = nn.Linear(d_q, d_q)

  def forward(
    self,
    x: torch.FloatTensor,
    context: torch.FloatTensor
  ) -> torch.FloatTensor:

    # x, context: (B, L, C)
    B, L, C = x.shape
    residual = x

    # GroupNorm needs (B, C, L)
    x = self.norm_query(x.permute(0, 2, 1))

    # LayerNorm needs (B, L, C)
    context = self.norm_context(context)

    q = self.q(x.permute(0, 2, 1)).view(B, self.num_heads, -1, C // self.num_heads)
    k = self.k(context).view(B, self.num_heads, -1, C // self.num_heads)
    v = self.v(context).view(B, self.num_heads, -1, C // self.num_heads)

    # Attention
    scale = (C // self.num_heads) ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)

    out = torch.matmul(attn, v)
    out = self.proj(out)

    return out + residual
    
