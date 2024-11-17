import torch
import torch.nn as nn

from model.architecture.Block import *


class ConvDown(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                               kernel_size=4,stride=2,padding=1)
    
    def forward(self, x):
        return self.block(x)
