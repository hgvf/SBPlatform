"""
Flexible UNet1d built with registry components
"""
import torch
import torch.nn as nn
from diffusers import UNet1DModel

from .registry import (
    register_backbone,
    get_time_embedding,
    get_text_embedding,
    get_pos_embedding,
    get_block,
    get_attention
)

# TODO: 自製 Unet1DConditionModel
