import torch
import torch.nn as nn

import torch
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, 
                 device,
                 dim, 
                 max_position):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim
        self.max_position = max_position

        # Position Embedding을 계산하는 함수
        self.position_embeddings = self._get_position_embeddings().to(device)

    def _get_position_embeddings(self):
        """
        sinusoidal 임베딩을 사인과 코사인을 결합하여 생성
        """
        # 각 position에 대해 임베딩을 계산
        position = torch.arange(self.max_position, dtype=torch.float).unsqueeze(1)  # shape: [max_position, 1]
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * -(math.log(10000.0) / self.dim))  # shape: [dim/2]

        # 사인과 코사인 값 계산
        embeddings = torch.zeros(self.max_position, self.dim)  # [max_position, dim]
        embeddings[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스: 사인
        embeddings[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스: 코사인

        return embeddings

    def forward(self, position_ids):
        """
        주어진 위치에 대해 임베딩을 반환
        position_ids: (batch_size, seq_len) 위치 인덱스 텐서
        """
        return self.position_embeddings[position_ids]

class Unet(nn.Module):

    def __init__(self,
                 image_shape,
                 max_channel: int,
                 time_embedding,
                 condition_embedding=None):
        super().__init__()

        channel = image_shape[0]
        image_size = image_shape[1:]

        encode_schedule_list, decode_schedule_list = self.channel_scheduling(channel, max_channel)

        mid_image_size = [s//(2**(len(encode_schedule_list)-1)) for s in image_size]

        print(f"encode schedule list:\n{encode_schedule_list}")
        print(f"decode schedule list:\n{decode_schedule_list}")

        self.encoder = UnetEncoder(
            channel_schedule_list=encode_schedule_list[:-1],
            image_size=image_size)
        self.encode_adapter = CBR_Block(
            channel_schedule=encode_schedule_list[-1],
            image_size=mid_image_size)

        self.decode_adapter = CBR_Block(
            channel_schedule=decode_schedule_list[0],
            image_size=mid_image_size)
        self.decoder = UnetDecoder(
            channel_schedule_list=decode_schedule_list[1:],
            image_size=image_size)

        self.out = nn.Conv2d(in_channels=decode_schedule_list[-1][-1], out_channels=channel, 
                            kernel_size=3, stride=1, padding=1)
        
        self.time_embedding = time_embedding
        self.condition_embedding = condition_embedding


    def channel_scheduling(self, start_channel, max_channel):
        encode_schedule_list = []
        now_schedule = [start_channel, 64, 64]

        while now_schedule[-1] < max_channel:
            encode_schedule_list.append(now_schedule)

            last_channel = now_schedule[-1]
            now_schedule = [last_channel, last_channel*2, last_channel*2]

        encode_schedule_list.append([
            now_schedule[0],
            now_schedule[-1]
            ])
        
        decode_schedule_list = [encode_schedule_list[-1][::-1]]
        for schedule in encode_schedule_list[-2::-1]:
            now_schedule = schedule[::]
            now_schedule[-1] *= 2
            decode_schedule_list.append(now_schedule[::-1])
        decode_schedule_list[-1][-1] = 64
        
        return encode_schedule_list, decode_schedule_list


    def forward(self, x, t, c=None):
        t_emb = self.time_embedding(t)
        c_emb = self.condition_embedding(c) \
            if self.condition_embedding is not None else None

        x, prev_list = self.encoder.residual_forward(x, t_emb, c_emb)
        x = self.encode_adapter(x, t_emb, c_emb)

        x = self.decode_adapter(x, t_emb, c_emb)
        x = self.decoder.residual_forward(x, prev_list[::-1], t_emb, c_emb)

        x = self.out(x)

        return x


class UnetEncoder(nn.Module):

    def __init__(self, 
                 channel_schedule_list,
                 image_size):
        super().__init__()

        self.encode_block_list = [
            UnetEncodeBlock(
                channel_schedule=channel_schedule, 
                image_size=[s//(2**i) for s in image_size])
            for i, channel_schedule in enumerate(channel_schedule_list)
        ]

        self.block = nn.Sequential(*self.encode_block_list)

    
    def forward(self, x):
        return self.block(x) 
    
    def residual_forward(self, x, t_emb, c_emb):
        residual_list = []

        for block in self.encode_block_list:
            x, h = block.residual_forward(x, t_emb, c_emb)
            residual_list.append(h)

        return x, residual_list


class UnetEncodeBlock(nn.Module):

    def __init__(self, 
                 channel_schedule, 
                 image_size,
                 pool_kernel_size=2):
        super().__init__()

        self.cbr_list = [
            CBR_Block(
                channel_schedule=channel_schedule,
                image_size=image_size
            ),

            nn.MaxPool2d(kernel_size=pool_kernel_size)
        ]

        self.cbr_block = nn.Sequential(*self.cbr_list)

    
    def forward(self, x):
        return self.cbr_block(x) 
    

    def residual_forward(self, x, t_emb, c_emb):
        h = self.cbr_list[0](x, t_emb, c_emb)
        x = self.cbr_list[1](h)
        return x, h


class UnetDecoder(nn.Module):

    def __init__(self,
                 channel_schedule_list,
                 image_size):
        super().__init__()

        self.decode_block_list = [
            UnetDecodeBlock(
                channel_schedule=channel_schedule,
                image_size=[s//(2**(len(channel_schedule_list)-1-i)) for s in image_size])
            for i, channel_schedule in enumerate(channel_schedule_list)
        ]

        self.block = nn.Sequential(*self.decode_block_list)


    def forward(self, x):
        return self.block(x)
    

    def residual_forward(self, x, prev_list, t_emb, c_emb):
        for block, prev in zip(self.decode_block_list, prev_list):
            x = block.residual_forward(x, prev, t_emb, c_emb)
        return x


class UnetDecodeBlock(nn.Module):

    def __init__(self, 
                 channel_schedule, 
                 image_size,
                 up_conv_kernel_size=2,
                 stride=2, padding=0, bias=True):
        super().__init__()

        assert channel_schedule[0]//2 == channel_schedule[1]

        up_conv_channel = channel_schedule[0]//2

        self.cbr_list = [
            nn.ConvTranspose2d(
                in_channels = up_conv_channel, 
                out_channels = up_conv_channel, 
                kernel_size = up_conv_kernel_size, 
                stride = stride, padding = padding, bias = bias),

            CBR_Block(
                channel_schedule=channel_schedule,
                image_size=image_size
            )
        ]

        self.block = nn.Sequential(*self.cbr_list)

    
    def forward(self, x):
        return self.block(x)


    def residual_forward(self, x, prev, t_emb, c_emb):
        up_conv = self.cbr_list[0](x)
        copy_and_crop = torch.cat((up_conv, prev),dim =1)
        return self.cbr_list[1](copy_and_crop, t_emb, c_emb)


class CBR_Block(nn.Module):

    def __init__(self, 
                 channel_schedule,
                 image_size, 
                 kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.time_embedding = nn.Sequential(
            nn.Linear(in_features=1, out_features=channel_schedule[0]),
            nn.ReLU()
            )

        cbr_list = []
        for idx in range(len(channel_schedule) - 1):
            cbr_list.append(
                CBR2d(
                    in_channels = channel_schedule[idx],
                    out_channels = channel_schedule[idx+1],
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    bias = bias
                    )
            )
        self.block = nn.Sequential(*cbr_list)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=image_size[0]*image_size[1], 
            num_heads=1, 
            batch_first=True)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb):
        t_emb = self.time_embedding(t_emb)
        
        t_emb = t_emb\
            .reshape(-1, t_emb.shape[1], 1, 1)\
                .expand(-1, -1, *x.shape[2:])
        
        x = x+t_emb

        x = self.block(x)

        x = x.flatten(2)

        c_emb = c_emb if c_emb is not None else x
        x, _ = self.cross_attention(x, c_emb, c_emb)
        
        x = x.reshape(t_emb.shape[0], -1, *t_emb.shape[2:])

        return x

    
class CBR2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        
        self.block = nn.Sequential(
            # nn.GroupNorm(num_groups=in_channels//4, num_channels=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    
