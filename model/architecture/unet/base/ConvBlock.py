import torch
import torch.nn as nn

from model.architecture.Block import *

class CTC(nn.Module):
    def __init__(self,
                num_group,
                t_emb_dim,
                in_channels, out_channels):
        super().__init__()

        self.first_conv = GSC(
            num_group=num_group,
            in_channels=in_channels,out_channels=out_channels)

        self.time_emb_layer = SF(in_features=t_emb_dim,out_features=out_channels)

        self.second_conv = GSC(
            num_group=num_group,
            in_channels=out_channels,out_channels=out_channels)
        
        self.residual_layer = nn.Conv2d(in_channels=in_channels,out_channels=out_channels
                                        ,kernel_size=1)


    def forward(self, x, t_emb, c_emb=None):
        residual = x

        x = self.first_conv(x)

        t_emb = self.time_emb_layer(t_emb)
        t_emb = t_emb.reshape(*x.shape[:2], 1, 1)
        x = x+t_emb+c_emb

        x = self.second_conv(x)

        residual = self.residual_layer(residual)
        x = x+residual
        
        return x

class GA(nn.Module):
    def __init__(self,
                num_group,
                num_head, channels):
        super().__init__()

        self.attention_norm = nn.GroupNorm(num_groups=num_group,num_channels=channels)
        self.attention = nn.MultiheadAttention(embed_dim=channels,num_heads=num_head,batch_first=True)

    def forward(self, x, c_emb=None):

        residual = x

        batch_size, channels, h, w = x.shape

        x = x.reshape(batch_size, channels, h*w)
        x = self.attention_norm(x)

        x = x.permute(0, 2, 1)
        if c_emb is not None:
            x, _ = self.attention(c_emb, x, x)
        else:
            x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).reshape(batch_size, channels, h, w)

        x = x+residual
        
        return x


class CTCA(nn.Module):
    def __init__(self,
                 num_group,
                 t_emb_dim,
                 num_head,
                 in_channels, out_channels):
        super().__init__()

        self.ctc = CTC(num_group=num_group, 
                       t_emb_dim=t_emb_dim,
                       in_channels=in_channels,out_channels=out_channels)

        self.ga = GA(num_group=num_group,
                       num_head=num_head,channels=out_channels)

    def forward(self, x, t_emb, c_emb=None):
        x = self.ctc(x, t_emb, c_emb)

        x = self.ga(x, c_emb)
        
        return x

class CAC(nn.Module):
    def __init__(self,
                 num_group,
                 t_emb_dim,
                 num_head,
                 in_channels, out_channels):
        super().__init__()

        self.first_ctc = CTC(num_group=num_group, 
                       t_emb_dim=t_emb_dim,
                       in_channels=in_channels,out_channels=out_channels)

        self.ga = GA(num_group=num_group,
                       num_head=num_head,channels=out_channels)
        
        self.second_ctc = CTC(num_group=num_group, 
                       t_emb_dim=t_emb_dim,
                       in_channels=out_channels,out_channels=out_channels)

    def forward(self, x, t_emb, c_emb=None):
        x = self.first_ctc(x, t_emb, c_emb)

        x = self.ga(x, c_emb)

        x = self.second_ctc(x, t_emb, c_emb)
        
        return x
