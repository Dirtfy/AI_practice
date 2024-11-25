import torch
import torch.nn as nn
import torch.nn.functional as F

from model.architecture.Block import *


class ConvDown(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                               kernel_size=2,stride=2)
    
    def forward(self, x):
        return self.block(x)

class MaxPoolDown(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)
    

class AvgPoolDown(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2,stride=2)