"""Logger for training metrics"""

import os
from pathlib import Path
import json
import numpy as np


class Logger:
    """Simple logger with tensorboard support"""
    def __init__(self, log_dir, use_tensorboard=True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                print("Tensorboard not available, using basic logging")
                self.use_tensorboard = False
        
        # Text log
        self.log_file = self.log_dir / 'training.log'
        
    def log_scalar(self, tag, value, step):
        """Log scalar value"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        
        with open(self.log_file, 'a') as f:
            f.write(f"Step {step}: {tag} = {value}\n")
    
    def log_scalars(self, tag, values_dict, step):
        """Log multiple scalars"""
        if self.writer:
            self.writer.add_scalars(tag, values_dict, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_figure(self, tag, figure, step):
        """Log matplotlib figure"""
        if self.writer:
            self.writer.add_figure(tag, figure, step)
    
    def close(self):
        if self.writer:
            self.writer.close()
