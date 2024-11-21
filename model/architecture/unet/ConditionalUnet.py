import torch.nn as nn

from model.architecture.unet.base.Base import Base

from model.architecture.unet.base.ConvBlock import *
from model.architecture.unet.base.DownBlock import *
from model.architecture.unet.base.UpBlock import *

from model.architecture.embedding.SinusoidalPositionEmbedding import SinusoidalPositionEmbedding

class ConditionalUnet(Base):
    def __init__(self,
                 input_shape,
                 depth,
                 t_emb_dim,
                 t_max_position,
                 num_class,
                 c_emb_dim):
        super().__init__()

        channel_schedule = [input_shape[0], *[32*(2**i) for i in range(depth+1)]]
        group_schedule = [1, *[8 for _ in range(depth)]]

        self.down_block_list = nn.ModuleList([
            DownBlock(t_emb_dim=t_emb_dim,
                      c_emb_dim=c_emb_dim,
                      num_group=group_schedule[i],
                      in_channels=channel_schedule[i],
                      out_channels=channel_schedule[i+1])
                      for i in range(depth)
        ])
        self.mid_block = MidBlock(t_emb_dim=t_emb_dim,
                                  c_emb_dim=c_emb_dim,
                            in_channels=channel_schedule[-2],
                            out_channels=channel_schedule[-1])
        self.up_block_list = nn.ModuleList([
            UpBlock(t_emb_dim=t_emb_dim,
                    c_emb_dim=c_emb_dim,
                    num_group=group_schedule[i+1],
                    in_channels=channel_schedule[i+2],
                    out_channels=channel_schedule[i+1])
                    for i in reversed(range(depth))
        ])
        self.out_block = nn.Sequential(
            nn.GroupNorm(num_groups=group_schedule[1], num_channels=channel_schedule[1]),
            nn.Conv2d(in_channels=channel_schedule[1], out_channels=input_shape[0],
                      kernel_size=3, padding=1)
        )

        self.time_embedding = SinusoidalPositionEmbedding(
            dim=t_emb_dim,
            max_position=t_max_position
        )

        self.condition_embedding = nn.Sequential(
            nn.Embedding(num_class, c_emb_dim),
            nn.Linear(c_emb_dim, c_emb_dim),
            nn.ReLU(),
            nn.Linear(c_emb_dim, c_emb_dim)
        )

    def forward(self, x, time, condition):
        t_emb = self.time_embedding(time)
        c_emb = self.condition_embedding(condition)

        return super().forward(x, t_emb, c_emb)


class DownBlock(nn.Module):
    def __init__(self,
                 t_emb_dim,
                 c_emb_dim,
                 in_channels, out_channels,
                 num_group=8, num_head=1):
        super().__init__()
        
        self.block = CTCA(num_group=num_group,
                          t_emb_dim=t_emb_dim,
                          c_emb_dim=c_emb_dim,
                          num_head=num_head,
                          in_channels=in_channels,out_channels=out_channels)
        
        self.down = ConvDown(in_channels=out_channels,out_channels=out_channels)

        self.inc = in_channels

    def forward(self, x, t_emb, c_emb):
        h = self.block(x, t_emb, c_emb)
        x = self.down(h)
        return x, h
    
class MidBlock(nn.Module):
    def __init__(self,
                 t_emb_dim,
                 c_emb_dim,
                 in_channels, out_channels,
                 num_group=8, num_head=1):
        super().__init__()

        self.block = CAC(num_group=num_group,
                         t_emb_dim=t_emb_dim,
                         c_emb_dim=c_emb_dim,
                         num_head=num_head,
                         in_channels=in_channels,out_channels=out_channels)

    def forward(self, x, t_emb, c_emb):
        return self.block(x, t_emb, c_emb)
    
class UpBlock(nn.Module):
    def __init__(self,
                 t_emb_dim,
                 c_emb_dim,
                 in_channels, out_channels,
                 num_group=8,num_head=1):
        super().__init__()
        
        self.up = ConvUp(in_channels=in_channels, out_channels=in_channels//2)
        self.block = CTCA(num_group=num_group,
                          t_emb_dim=t_emb_dim,
                          c_emb_dim=c_emb_dim,
                          num_head=num_head,
                          in_channels=in_channels,out_channels=out_channels)
        
    def forward(self, skipped, x, t_emb, c_emb):
        x = self.up(x)
        x = torch.cat([x, skipped], dim=1)

        x = self.block(x, t_emb, c_emb)

        return x
