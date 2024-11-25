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

    def forward(self, x: torch.Tensor, t_emb=None, c_emb=None) -> torch.Tensor:
        
        skip_list = []
        for down_block in self.down_block_list:
            x, h = self.down_condition_check(down_block, x, t_emb, c_emb)

            skip_list.append(h)
        
        x = self.mid_condition_check(self.mid_block, x, t_emb, c_emb)

        for up_block, skip in zip(self.up_block_list, reversed(skip_list)):
            x = self.up_condition_check(up_block, skip, x, t_emb, c_emb)

        x = self.out_block(x)
        
        return x
    
    def down_condition_check(self, call, x, t_emb, c_emb):
        if t_emb is None and c_emb is None:
            x, h = call(x)
        elif t_emb is None:
            x, h = call(x, c_emb)
        elif c_emb is None:
            x, h = call(x, t_emb)
        else:
            x, h = call(x, t_emb, c_emb)

        return x, h
    
    def mid_condition_check(self, call, x, t_emb, c_emb):
        if t_emb is None and c_emb is None:
            x = call(x)
        elif t_emb is None:
            x = call(x, c_emb)
        elif c_emb is None:
            x = call(x, t_emb)
        else:
            x = call(x, t_emb, c_emb)

        return x
    
    def up_condition_check(self, call, skip, x, t_emb, c_emb):
        if t_emb is None and c_emb is None:
            x = call(skip, x)
        elif t_emb is None:
            x = call(skip, x, c_emb)
        elif c_emb is None:
            x = call(skip, x, t_emb)
        else:
            x = call(skip, x, t_emb, c_emb)

        return x

