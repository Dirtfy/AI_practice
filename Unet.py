import torch
import torch.nn as nn

class Unet(nn.Module):

    def __init__(self,
                 channel: int,
                 max_channel: int):
        super().__init__()

        encode_schedule_list, decode_schedule_list = self.channel_scheduling(channel, max_channel)

        print(f"encode schedule list:\n{encode_schedule_list}")
        print(f"decode schedule list:\n{decode_schedule_list}")

        self.encoder = UnetEncoder(channel_schedule_list=encode_schedule_list[:-1])
        # 5층 파란색 화살표
        self.last_encoder = CBR_Block(channel_schedule=encode_schedule_list[-1])

        # Expansive path
        # 5층 파란색 2번쨰 화살표인데 디코더로
        self.first_decoder = CBR_Block(channel_schedule=decode_schedule_list[0])
        self.decoder = UnetDecoder(channel_schedule_list=decode_schedule_list[1:])

        # segmentation에 필요한 n개의 클래스에 대한 output 정의
        self.fc = nn.Conv2d(in_channels=decode_schedule_list[-1][-1], out_channels=1, 
                            kernel_size=1, stride=1, padding=0, bias=True)


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


    def forward(self, x):
        x, prev_list = self.encoder.residual_forward(x)
        x = self.last_encoder(x)
        x = self.first_decoder(x)
        x = self.decoder.residual_forward(x, prev_list[::-1])
        x = self.fc(x)
        return x


class UnetEncoder(nn.Module):

    def __init__(self, 
                 channel_schedule_list):
        super().__init__()

        self.encode_block_list = [
            UnetEncodeBlock(channel_schedule=channel_schedule)
            for channel_schedule in channel_schedule_list
        ]

        self.block = nn.Sequential(*self.encode_block_list)

    
    def forward(self, x):
        return self.block(x) 
    
    def residual_forward(self, x):
        residual_list = []

        for block in self.encode_block_list:
            x, h = block.residual_forward(x)
            residual_list.append(h)

        return x, residual_list


class UnetEncodeBlock(nn.Module):

    def __init__(self, 
                 channel_schedule, 
                 pool_kernel_size=2):
        super().__init__()

        self.cbr_list = [
            CBR_Block(
                channel_schedule=channel_schedule
            ),

            nn.MaxPool2d(kernel_size=pool_kernel_size)
        ]

        self.cbr_block = nn.Sequential(*self.cbr_list)

    
    def forward(self, x):
        return self.cbr_block(x) 
    

    def residual_forward(self, x):
        h = self.cbr_list[0](x)
        x = self.cbr_list[1](h)
        return x, h


class UnetDecoder(nn.Module):

    def __init__(self,
                 channel_schedule_list):
        super().__init__()

        self.decode_block_list = [
            UnetDecodeBlock(channel_schedule=channel_schedule)
            for channel_schedule in channel_schedule_list
        ]

        self.block = nn.Sequential(*self.decode_block_list)


    def forward(self, x):
        return self.block(x)
    

    def residual_forward(self, x, prev_list):
        for block, prev in zip(self.decode_block_list, prev_list):
            x = block.residual_forward(x, prev)
        return x


class UnetDecodeBlock(nn.Module):

    def __init__(self, 
                 channel_schedule, 
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
                channel_schedule=channel_schedule
            )
        ]

        self.block = nn.Sequential(*self.cbr_list)

    
    def forward(self, x):
        return self.block(x)


    def residual_forward(self, x, prev):
        up_conv = self.cbr_list[0](x)
        copy_and_crop = torch.cat((up_conv, prev),dim =1)
        return self.cbr_list[1](copy_and_crop)


class CBR_Block(nn.Module):

    def __init__(self, channel_schedule, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

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

    def forward(self, x):
        return self.block(x)

    
class CBR2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    
