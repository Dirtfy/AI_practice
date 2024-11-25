import torch.nn as nn
import torch.nn.functional as F

class ConvTransUp(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super().__init__()

        self.block = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,
                               kernel_size=4,stride=2,padding=1)
    
    def forward(self, x):
        return self.block(x)
    
class ConvUp(nn.Module):
    def __init__(self,
                 channels) -> None:
        super().__init__()

        self.block = nn.Conv2d(in_channels=channels, out_channels=channels, 
                      kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = F.interpolate(x, size=(H * 2, W * 2), mode='nearest')
        
        return self.block(x)