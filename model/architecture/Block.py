import torch.nn as nn

class GSC(nn.Module):
    def __init__(self,
                 num_group,
                 in_channels, out_channels):
        super().__init__()

        assert in_channels%num_group == 0

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=num_group,num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.block(x)
    
class SF(nn.Module):
    def __init__(self,
                 in_features, out_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=in_features, out_features=out_features)
        )

    def forward(self, x):
        return self.block(x)
    
class GA(nn.Module):
    def __init__(self,
                 num_group, channels,
                 num_heads):
        super().__init__()

        assert channels%num_group == 0

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=num_group,num_channels=channels),
            nn.MultiheadAttention(embed_dim=channels,num_heads=num_heads,batch_first=True)
        )

    def forward(self, q, k, v):
        return self.block(q, k, v)
    

class ResidualBlock(nn.Module):
    def __init__(self,
                 module: nn.Module):
        self.module = module

    def forward(self, x):
        return x + self.module(x)
        
