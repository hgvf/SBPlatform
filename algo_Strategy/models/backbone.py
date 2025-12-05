import torch
import torch.nn as nn
from diffusers import UNet1DModel
from .registry import register_model

@register_model("unet1d")
class DiffUnet1D(UNet1DModel):
  """
  Extended UNet1D with custom features 
  (Only conditioned on time-series)
  """
  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @classmethod
    def from_config(cls, config):
        """Create model from config dict"""
        return cls(
            sample_size=config.get('sample_size', 128),
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            layers_per_block=config.get('layers_per_block', 2),
            block_out_channels=tuple(config.get('block_out_channels', [32, 64, 128, 256])),
            down_block_types=tuple(config.get('down_block_types', 
                ["DownBlock1D", "DownBlock1D", "DownBlock1D", "AttnDownBlock1D"])),
            up_block_types=tuple(config.get('up_block_types',
                ["AttnUpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"])),
        )

