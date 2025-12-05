import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from ..registry import register_pos_embedding

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    把最後一維拆成 (x_even, x_odd)，做 2D 旋轉用的 helper：
    (x0, x1) -> (-x1, x0)
    """
    
    x_even = x[..., ::2]   # 偶數維
    x_odd = x[..., 1::2]   # 奇數維
    
    # 拼回來：( -x_odd, x_even )
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    對 x 的前 rotary_dim 維套 RoPE，剩餘維度維持不變。

    參數:
        x: [batch, n_heads, seq_len, head_dim]
        cos, sin: [seq_len, 1, rotary_dim]（會自動 broadcast）
        rotary_dim: 使用多少維做 RoPE，若為 None 則用 head_dim 全部。
    """
    
    head_dim = x.shape[-1]
    if rotary_dim is None:
        rotary_dim = head_dim
    assert rotary_dim % 2 == 0, "rotary_dim 必須是偶數"

    x_rot = x[..., :rotary_dim]          # 要旋轉的部分
    x_pass = x[..., rotary_dim:]         # 不旋轉的部分

    # cos, sin shape: [seq_len, 1, rotary_dim] -> [1, 1, seq_len, rotary_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # RoPE: x_rot * cos + rotate_half(x_rot) * sin
    x_rotated = x_rot * cos + rotate_half(x_rot) * sin

    if x_pass.numel() == 0:
        return x_rotated
        
    return torch.cat([x_rotated, x_pass], dim=-1)
    
@register_pos_embedding("RoPE")
class RotaryPositionEmbedding(nn.Module):
    """
    Llama / Qwen / Mistral / SD3 風格的 RoPE 實作。

    用法（在 attention 裡）：
        rope = RotaryPositionEmbedding(head_dim, max_position_embeddings=4096)
        ...
        cos, sin = rope.get_cos_sin(position_ids)        # position_ids: [batch, seq_len]
        q = apply_rotary_pos_emb(q, cos, sin, rope.rotary_dim)
        k = apply_rotary_pos_emb(k, cos, sin, rope.rotary_dim)

    預設假設 Q/K shape = [batch, n_heads, seq_len, head_dim]
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        rotary_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            dim: head_dim（注意是單頭維度）
            max_position_embeddings: 支援的最大序列長度
            base: RoPE 的頻率 base（Llama/Qwen 用 10000 或 1000000 等）
            rotary_dim: 用多少維做 RoPE，None = 用 dim 全部
        """
        
        super().__init__()
        if rotary_dim is None:
            rotary_dim = dim
        assert rotary_dim % 2 == 0, "rotary_dim 必須是偶數"

        self.dim = dim
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 建立 inv_freq（頻率），跟 Llama/Qwen 做法一致
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        # [max_seq_len]
        t = torch.arange(max_position_embeddings, dtype=torch.float32)

        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_seq_len, rotary_dim/2]
        # 展開成 [max_seq_len, rotary_dim]，偶數/奇數維共享 freq
        emb = torch.cat([freqs, freqs], dim=-1)       # [max_seq_len, rotary_dim]

        cos_cached = emb.cos()
        sin_cached = emb.sin()

        if dtype is not None:
            cos_cached = cos_cached.to(dtype)
            sin_cached = sin_cached.to(dtype)
        if device is not None:
            cos_cached = cos_cached.to(device)
            sin_cached = sin_cached.to(device)

        # 註冊為 buffer，參與 device 移動，但不參與訓練
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def get_cos_sin(
        self,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        給定 position_ids（通常 shape = [batch, seq_len]），
        回傳對應的 cos/sin table。

        回傳:
            cos, sin: [seq_len, 1, rotary_dim]
            之後會被 apply_rotary_pos_emb 轉成 [1,1,seq_len,rotary_dim] broadcast。
        """
        
        # position_ids: [batch, seq_len]
        # 大部分模型會讓同一 batch 的 position 一樣，所以這裡取第 0 個即可
        if position_ids.dim() == 2:
            # 取 [seq_len]，假設每個 batch 的 position 一致
            pos = position_ids[0]
        else:
            # 允許直接給 [seq_len]
            pos = position_ids

        cos = self.cos_cached[pos]  # [seq_len, rotary_dim]
        sin = self.sin_cached[pos]  # [seq_len, rotary_dim]

        # 加上 head 維度 broadcast：[seq_len, 1, rotary_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        cos, sin = self.get_cos_sin(position_ids)
        rot = apply_rotary_pos_emb(x, cos, sin, self.rotary_dim)
        
        return rot
        
@register_pos_embedding("sinusoidal")
class SinusoidalTimeEmbedding(nn.Module):
    """
    標準的 Sinusoidal Position Embedding
    """
  
    def __init__(self, dim: int):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.dim = dim

    def forward(self, timesteps: torch.FloatTensor) -> torch.FloatTensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings + timesteps
      
