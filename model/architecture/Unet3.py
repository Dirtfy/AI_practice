import torch
import torch.nn as nn
import torch.nn.functional as F


# Non-linearity function (Swish)
def nonlinearity(x):
    return F.silu(x)


# Normalize function (Group Normalization)
class Normalize(nn.Module):
    def __init__(self, temb, num_groups=32):
        super(Normalize, self).__init__()
        self.num_groups = num_groups
        self.temb = temb  # Time Embedding, not directly used in normalizing here
        self.group_norm = nn.GroupNorm(num_groups, temb.shape[-1])

    def forward(self, x):
        return self.group_norm(x)


# Upsample function
class Upsample(nn.Module):
    def __init__(self, with_conv, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        if self.with_conv:
            x = self.conv(x)
        return x


# Downsample function
class Downsample(nn.Module):
    def __init__(self, with_conv, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = self.pool(x)
        return x


# ResNet Block (with timestep embedding)
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb, dropout=0.0, conv_shortcut=False):
        super(ResNetBlock, self).__init__()
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout

        # First layer
        self.norm1 = Normalize(temb, num_groups=32)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Second layer
        self.norm2 = Normalize(temb, num_groups=32)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Time Embedding projection
        self.temb_proj = nn.Linear(temb.shape[1], out_channels)

        # Shortcut conv
        if self.conv_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # Add timestep embedding
        temb_proj = nonlinearity(self.temb_proj(temb)).view(temb.size(0), -1, 1, 1)
        h = h + temb_proj

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.conv_shortcut:
            x = self.shortcut(x)

        return x + h


# Attention Block
class AttnBlock(nn.Module):
    def __init__(self, in_channels, temb, heads=1):
        super(AttnBlock, self).__init__()
        self.norm = Normalize(temb)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, temb):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        w = torch.einsum('bchw,bCHW->bHWCHW', q, k) * (q.size(1) ** -0.5)
        w = w.view(*w.shape[:3], -1)
        w = F.softmax(w, dim=-1)
        w = w.view(*w.shape[:3], *q.shape[-2:])

        h = torch.einsum('bHWCHW,bHWc->bchw', w, v)
        h = self.proj_out(h)

        return x + h


# Main Model
class DiffusionModel(nn.Module):
    def __init__(self, ch=64, out_ch=3, num_classes=1, num_res_blocks=2, num_resolutions=4, attn_resolutions=[16],
                 dropout=0.0, resamp_with_conv=True):
        super(DiffusionModel, self).__init__()

        # Time embedding layers
        self.temb_dense0 = nn.Linear(ch, ch * 4)
        self.temb_dense1 = nn.Linear(ch * 4, ch * 4)

        # Initial Conv Layer
        self.conv_in = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        # Downsampling layers
        self.downsample_layers = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                self.downsample_blocks.append(ResNetBlock(ch, ch * (2 ** i_level), temb=None, dropout=dropout))

            self.downsample_layers.append(Downsample(True, ch * (2 ** i_level), ch * (2 ** (i_level + 1))))

        # Middle layer
        self.mid_res_block1 = ResNetBlock(ch * (2 ** (num_resolutions - 1)), ch * (2 ** num_resolutions), temb=None, dropout=dropout)
        self.attn_block = AttnBlock(ch * (2 ** num_resolutions), temb=None)

        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks):
                self.upsample_layers.append(ResNetBlock(ch * (2 ** i_level), ch * (2 ** (i_level + 1)), temb=None, dropout=dropout))

            self.upsample_layers.append(Upsample(True, ch * (2 ** i_level), ch * (2 ** (i_level + 1))))

        # Output Conv
        self.conv_out = nn.Conv2d(ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        temb = self.temb_dense0(t)
        temb = nonlinearity(temb)
        temb = self.temb_dense1(temb)
        temb = nonlinearity(temb)

        # Downsampling
        hs = [self.conv_in(x)]
        for downsample, block in zip(self.downsample_layers, self.downsample_blocks):
            for b in block:
                h = b(hs[-1], temb)
                hs.append(h)
            hs.append(downsample(h))

        # Middle
        h = hs[-1]
        h = self.mid_res_block1(h, temb)
        h = self.attn_block(h, temb)

        # Upsampling
        for upsample, block in zip(self.upsample_layers, self.downsample_blocks[::-1]):
            for b in block:
                h = b(torch.cat([h, hs.pop()], dim=1), temb)
            h = upsample(h)

        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
