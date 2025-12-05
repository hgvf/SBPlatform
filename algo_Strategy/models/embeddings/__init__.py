# 自動導入並註冊所有 embedding 模組
from .time_series import (
  IdenticalTimeEmbedding,
  LinearTimeEmbedding
)

from .position import (
  RotaryPositionEmbedding,
  SinusoidalTimeEmbedding
)

from .text import (
  Transformers,
  SentTransformer,
  LLMEmbedding
)

__all__ = [
    'IdenticalTimeEmbedding',
    'LinearTimeEmbedding',
    'RotaryPositionEmbedding',
    'SinusoidalTimeEmbedding',
    'Transformers',
    'SentTransformer',
    'LLMEmbedding'
]
