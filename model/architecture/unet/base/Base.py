import torch
import torch.nn as nn

from abc import *

from typing import List

class Base(nn.Module):
    def __init__(self, 
                 down_block_list=None,
                 mid_block=None,
                 up_block_list=None,
                 out_block=None) -> None:
        super().__init__()

        self.down_block_list = down_block_list
        self.mid_block = mid_block
        self.up_block_list = up_block_list
        self.out_block = out_block

    def forward(self, x: torch.Tensor, t_emb=None, condition=None) -> torch.Tensor:

        skip_list = []
        for down_block in self.down_block_list:
            x, h = self.down_condition_check(down_block, x, t_emb, condition)

            skip_list.append(h)

        x = self.down_condition_check(self.mid_block, x, t_emb, condition)

        for up_block, skip in zip(self.up_block_list, reversed(skip_list)):
            x = self.up_condition_check(up_block, skip, x, t_emb, condition)

        x = self.out_block(x)

        return x
    
    def down_condition_check(self, call, x, t_emb, condition):
        if t_emb is None and condition is None:
            h = call(x)
        elif t_emb is None:
            h = call(x, condition)
        elif condition is None:
            h = call(x, t_emb)
        else:
            h = call(x, t_emb, condition)

        return x, h
    
    def up_condition_check(self, call, skip, x, t_emb, condition):
        if t_emb is None and condition is None:
            h = call(skip, x)
        elif t_emb is None:
            h = call(skip, x, condition)
        elif condition is None:
            h = call(skip, x, t_emb)
        else:
            h = call(skip, x, t_emb, condition)

        return h
