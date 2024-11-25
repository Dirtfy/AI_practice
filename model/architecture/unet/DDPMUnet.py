import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base.Base import Base

from model.architecture.unet.base.Base import Base

from model.architecture.unet.base.ConvBlock import *
from model.architecture.unet.base.DownBlock import *
from model.architecture.unet.base.UpBlock import *

from model.architecture.embedding.SinusoidalPositionEmbedding import SinusoidalPositionEmbedding

class Unet(Base):
    def __init__(self,
                 input_shape,
                 depth,
                 t_emb_dim,
                 t_max_position):
        super().__init__()

        assert input_shape[1]//(2**depth) >= 1 and input_shape[2]//(2**depth) >= 1

        channel_schedule = [input_shape[0], *[32*(2**i) for i in range(depth)]]
        group_schedule = [1, *[8 for _ in range(depth)]]

        self.down_block_list = nn.ModuleList([
            DownBlock(t_emb_dim=t_emb_dim,
                      num_group=group_schedule[i],
                      in_channels=channel_schedule[i],
                      out_channels=channel_schedule[i+1],
                      with_attention=channel_schedule[i]==32)
                      for i in range(depth)
        ])
        self.mid_block = MidBlock(t_emb_dim=t_emb_dim,
                                  channels=channel_schedule[-1],
                                  num_group=group_schedule[-1])
        self.up_block_list = nn.ModuleList([
            UpBlock(t_emb_dim=t_emb_dim,
                    num_group=group_schedule[i],
                    in_channels=channel_schedule[i+1],
                    out_channels=channel_schedule[i],
                    with_attention=channel_schedule[i+1] == 32)
                    for i in reversed(range(depth))
        ])
        self.out_block = nn.Identity()

        self.time_embedding = SinusoidalPositionEmbedding(
            dim=t_emb_dim,
            max_position=t_max_position
        )

    def forward(self, x, time):
        t_emb = self.time_embedding(time)

        return super().forward(x, t_emb, None)
    
class DownBlock(nn.Module):
    def __init__(self,
                 t_emb_dim,
                 in_channels, out_channels,
                 with_attention: bool,
                 num_group=8,num_haeds=1):
        super().__init__()
        
        self.block = RTR(num_group=num_group, channels=in_channels, t_emb_dim=t_emb_dim)

        self.with_attention = with_attention
        if with_attention:
            self.attention = GA(num_group=num_group,channels=in_channels,num_heads=num_haeds)

        self.down = ConvDown(in_channels=in_channels,out_channels=out_channels)

    def forward(self, x, t_emb):
        h = self.block(x, t_emb)
        if self.with_attention:
            h = self.attention(h)
        x = self.down(h)
        return x, h
    
class MidBlock(nn.Module):
    def __init__(self,
                 t_emb_dim,
                 channels,
                 num_group=8, num_head=1):
        super().__init__()

        self.first_block = RTR(num_group=num_group,channels=channels,t_emb_dim=t_emb_dim)
        self.attention = GA(num_group=num_group,channels=channels,num_heads=num_head)
        self.second_block = RTR(num_group=num_group,channels=channels,t_emb_dim=t_emb_dim)

    def forward(self, x, t_emb):
        x = self.first_block(x, t_emb)
        x = self.attention(x)
        x = self.second_block(x, t_emb)
        return x
    
class UpBlock(nn.Module):
    def __init__(self,
                 t_emb_dim,
                 in_channels, out_channels,
                 with_attention: bool,
                 num_group=8,num_head=1):
        super().__init__()
        
        self.up = ConvUp(channels=in_channels)
        self.block = RTR(num_group=num_group, channels=out_channels, t_emb_dim=t_emb_dim)

        self.with_attention = with_attention
        if with_attention:
            self.attention = GA(num_group=num_group,channels=out_channels, num_heads=num_head)
        
    def forward(self, skipped, x, t_emb):
        x = self.up(x)
        x = torch.cat([x, skipped], dim=1)

        x = self.block(x, t_emb)
        
        if self.with_attention:
            x = self.attention(x)

        return x


def nonlinearity(x):
    return F.silu(x)


class Normalize(nn.Module):
    def __init__(self, temb, name):
        super(Normalize, self).__init__()
        # Group normalization (assuming it's the same as tf.contrib.layers.group_norm)
        self.norm = nn.GroupNorm(32, temb.shape[-1])  # assuming 32 groups for simplicity
    
    def forward(self, x):
        return self.norm(x)


class Upsample(nn.Module):
    def __init__(self, name, with_conv):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        self.name = name

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.interpolate(x, size=(H * 2, W * 2), mode='nearest')
        if self.with_conv:
            x = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)(x)
        return x


class Downsample(nn.Module):
    def __init__(self, name, with_conv):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        self.name = name

    def forward(self, x):
        B, C, H, W = x.shape
        if self.with_conv:
            x = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, temb, name, out_ch=None, conv_shortcut=False, dropout=0.):
        super(ResNetBlock, self).__init__()
        self.temb = temb
        self.name = name
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels=temb.shape[0], out_channels=self.out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.out_ch, out_channels=self.out_ch, kernel_size=3, padding=1)

        self.norm1 = Normalize(self.temb, name="norm1")
        self.norm2 = Normalize(self.temb, name="norm2")

        self.temb_proj = nn.Linear(self.temb.shape[0], self.out_ch)
    
    def forward(self, x):
        h = self.norm1(nonlinearity(x))
        h = self.conv1(h)

        h += self.temb_proj(nonlinearity(self.temb))[:, :, None, None]

        h = self.norm2(nonlinearity(h))
        h = F.dropout(h, p=self.dropout)
        h = self.conv2(h)

        if self.out_ch != x.shape[1]:
            if self.conv_shortcut:
                shortcut = nn.Conv2d(x.shape[1], self.out_ch, kernel_size=1)(x)
            else:
                shortcut = nn.Conv2d(x.shape[1], self.out_ch, kernel_size=1)(x)
        else:
            shortcut = x

        return shortcut + h


class AttnBlock(nn.Module):
    def __init__(self, name, temb, C):
        super(AttnBlock, self).__init__()
        self.temb = temb
        self.name = name
        self.C = C
        self.norm = Normalize(self.temb, name='norm')
        self.q = nn.Conv2d(C, C, kernel_size=1)
        self.k = nn.Conv2d(C, C, kernel_size=1)
        self.v = nn.Conv2d(C, C, kernel_size=1)
        self.proj_out = nn.Conv2d(C, C, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        w = torch.einsum('bchw,bCHW->bhwHW', q, k) * (self.C ** (-0.5))
        w = w.view(w.shape[0], w.shape[1], w.shape[2], -1)
        w = F.softmax(w, dim=-1)
        w = w.view(w.shape[0], w.shape[1], w.shape[2], w.shape[3], w.shape[4])

        h = torch.einsum('bhwHW,bHWc->bhwc', w, v)
        h = self.proj_out(h)

        return x + h


class DDPMModel(nn.Module):
    def __init__(self, ch, out_ch, num_res_blocks, attn_resolutions, num_classes=1, dropout=0., resamp_with_conv=True):
        super(DDPMModel, self).__init__()
        self.ch = ch
        self.out_ch = out_ch
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv

        # Timestep embedding layers
        self.temb = nn.Linear(ch, ch * 4)
        self.temb_proj = nn.Linear(ch * 4, ch * 4)

        self.conv_in = nn.Conv2d(3, ch, kernel_size=3, padding=1)

        # Downsample layers
        self.downsample_blocks = nn.ModuleList()
        self.resnet_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            self.resnet_blocks.append(ResNetBlock(self.temb, f'block_{i}', out_ch=ch, dropout=self.dropout))

        self.attn_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        # Upsample layers
        self.conv_out = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        B, C, H, W = x.shape

        temb = self.temb(t)
        temb = nonlinearity(self.temb_proj(nonlinearity(temb)))

        hs = [self.conv_in(x)]

        # Downsampling
        for i_level in range(len(self.attn_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.resnet_blocks[i_block](hs[-1])
                if h.shape[2] in self.attn_resolutions:
                    h = AttnBlock(f'attn_{i_block}', temb, h.shape[1])(h)
                hs.append(h)
            if i_level != len(self.attn_resolutions) - 1:
                hs.append(Downsample(f'downsample_{i_level}', with_conv=self.resamp_with_conv)(hs[-1]))

        # Middle block
        h = self.resnet_blocks[0](hs[-1])
        h = AttnBlock('attn_1', temb, h.shape[1])(h)
        h = self.resnet_blocks[1](h)

        # Upsampling
        for i_level in reversed(range(len(self.attn_resolutions))):
            for i_block in range(self.num_res_blocks + 1):
                h = self.resnet_blocks[i_block](torch.cat([h, hs.pop()], dim=1))
                if h.shape[2] in self.attn_resolutions:
                    h = AttnBlock(f'attn_{i_block}', temb, h.shape[1])(h)
            if i_level != 0:
                h = Upsample('upsample', with_conv=self.resamp_with_conv)(h)

        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
