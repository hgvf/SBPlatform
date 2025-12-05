"""
Modality: Time Series
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from ..registry import register_time_embedding

@register_time_embedding("identical")
class IdenticalTimeEmbedding(nn.Module):
  """
  不採用 embedding, 以 raw data 型式輸出
  """

  def __init__(self, **kwargs):
    super(IdenticalTimeEmbedding, self).__init__()

    self.identity = nn.Identity()
    
  def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
    return self.identity(x)  

@register_time_embedding("linear")
class LinearTimeEmbedding(nn.Module):
  """
  Fully connected layer for embeddings
  """

  def __init__(
    self, 
    dim: int, 
    d_model: int,
    activate_fn: Optional[nn.Module] = None,
    **kwargs
  ):
    """
    Args:
      dim: The dimension of the output vector.
      d_model: The dimension of the input vector.
    """
    
    super(LinearTimeEmbedding, self).__init__()

    if activate_fn is None:
      activate_fn = nn.ReLU()
    
    self.embed = nn.Sequential(
      nn.Linear(d_model, dim),
      activate_fn
    )
    
  def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
    return self.embed(x)  
