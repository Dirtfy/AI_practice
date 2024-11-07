import torch
import torch.nn as nn

#Unet 구현
class Unet(nn.Module):

    def __init__(self):
        super().__init__()

        # Contracting path
        # 좌측 레이어 enc (인코더)
        # 1층 좌측 첫번째 레이어 두개 
        self.enc_1 = CBR_Block(channel_schedule=[1, 64, 64])

        # 다음 빨간색 화살표 max_pool 2*2
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        # 2층 파란색 화살표     
        self.enc_2 = CBR_Block(channel_schedule=[64, 128, 128])

        # 다음 빨간색 화살표 max_pool 2*2
        self.max_pool_2 = nn.MaxPool2d(kernel_size =2)

        # 3층 파란색 화살표 
        self.enc_3 = CBR_Block(channel_schedule=[128, 256, 256])

        # 다음 빨간색 화살표 max_pool 2*2
        self.max_pool_3 = nn.MaxPool2d(kernel_size =2)


        # 4층 파란색 화살표 
        self.enc_4 = CBR_Block(channel_schedule=[256, 512, 512])

        # 다음 빨간색 화살표 max_pool 2*2
        self.max_pool_4 = nn.MaxPool2d(kernel_size =2)

        # 5층 파란색 화살표
        self.enc_5 = CBR_Block(channel_schedule=[512, 1024])

        # Expansive path
        # 5층 파란색 2번쨰 화살표인데 디코더로
        self.dec_5 = CBR_Block(channel_schedule=[1024, 512])

        # 초록색 화살표
        self.up_conv_5 = nn.ConvTranspose2d(in_channels =512, out_channels = 512, kernel_size = 2, stride = 2, padding = 0, bias = True)

        # enc4_2와 대칭이되는 점을 보면 dec4_2 input값은 512가 맞는데, unet 아키텍쳐를 보니
        # enc4_2 에서 회색 화살표로 dec4_2로 와서 copy and crop이 일어남
        # 따라서 dec4_2 in_channels = 1024로 설정
        self.dec_4 = CBR_Block(channel_schedule=[2*512, 512, 256])

        # 초록색 화살표
        self.up_conv_4 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # 3층 파란색 화살표
        self.dec_3 = CBR_Block(channel_schedule=[2*256, 256, 128])

        # 초록색 화살표
        self.up_conv_3 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # 2층 파란색 화살표
        self.dec_2 = CBR_Block(channel_schedule=[2*128, 128, 64])
        
        # 초록색 화살표
        self.up_conv_2 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # 1층 파란색 화살표
        self.dec_1 = CBR_Block(channel_schedule=[2*64, 64, 64])

        # segmentation에 필요한 n개의 클래스에 대한 output 정의
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # 좌측 1층 레이어 2개 연결 및 빨간색 화살표
        enc_1 = self.enc_1(x)
        max_pool_1 = self.max_pool_1(enc_1)

        # 좌측 2층 레이어 2개 연결 및 빨간색 화살표
        enc_2 = self.enc_2(max_pool_1)
        maxs_pool_2 = self.max_pool_2(enc_2)

        # 좌측 3층 레이어 2개 연결 및 빨간색 화살표
        enc_3 = self.enc_3(maxs_pool_2)
        max_pool_3 = self.max_pool_3(enc_3)

        # 좌측 4층 레이어 2개 연결 및 빨간색 화살표
        enc_4 = self.enc_4(max_pool_3)
        max_pool_4 = self.max_pool_4(enc_4)

        # 좌측 5층 레이어
        enc_5 = self.enc_5(max_pool_4)

        # 우측 5층 레이어 및 초록색 화살표
        dec_5 = self.dec_5(enc_5)
        up_conv_5 = self.up_conv_5(dec_5)

        # 하얀색 부분 연결하기
        copy_and_crop_4 = torch.cat((up_conv_5, enc_4),dim =1)

        # 파란색 화살표 실행
        # cat에서 512 + 512 로 1024의 레이어 만들고 파란색 화살표 수행후 아웃풋값을 512로 만듬
        dec_4 = self.dec_4(copy_and_crop_4)
        # 여기까지 하면 우측 4층 레이어까지 생성

        # 반복 3층
        up_conv_4 = self.up_conv_4(dec_4)
        copy_and_crop_3 = torch.cat((up_conv_4, enc_3),dim =1)
        dec_3 = self.dec_3(copy_and_crop_3)

        # 반복 2층
        up_conv_3 = self.up_conv_3(dec_3)
        copy_and_crop_2 = torch.cat((up_conv_3, enc_2),dim =1)
        dec_2 = self.dec_2(copy_and_crop_2)

        # 반복 1층
        up_conv_2 = self.up_conv_2(dec_2)
        copy_and_crop_1 = torch.cat((up_conv_2, enc_1),dim =1)
        dec_1 = self.dec_1(copy_and_crop_1)

        x = self.fc(dec_1)

        return x
   
class ConditionalUnet(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x):
        return x
    
class UnetEncoder(nn.Module):

    def __init__(self, 
                 channel_schedule_list):
        super.__init__()

        self.cbr_list = []
        for idx in range(channel_schedule_list):
            self.cbr_list.append(
                ResidualCBR_Block(channel_schedule=channel_schedule_list[idx])
            )

        self.cbr_block = nn.Sequential(*self.cbr_list)

    
    def forward(self, x):
        return self.cbr_block(x) 
    
    def residual_forward(self, x):
         # 좌측 1층 레이어 2개 연결 및 빨간색 화살표
        enc_1 = self.enc_1(x)
        max_pool_1 = self.max_pool_1(enc_1)

        # 좌측 2층 레이어 2개 연결 및 빨간색 화살표
        enc_2 = self.enc_2(max_pool_1)
        maxs_pool_2 = self.max_pool_2(enc_2)

        # 좌측 3층 레이어 2개 연결 및 빨간색 화살표
        enc_3 = self.enc_3(maxs_pool_2)
        max_pool_3 = self.max_pool_3(enc_3)

        # 좌측 4층 레이어 2개 연결 및 빨간색 화살표
        enc_4 = self.enc_4(max_pool_3)
        max_pool_4 = self.max_pool_4(enc_4)

        # 좌측 5층 레이어
        enc_5 = self.enc_5(max_pool_4)

class ResidualCBR_Block(nn.Module):

    def __init__(self, 
                 channel_schedule, 
                 pool_kernel_size=2,
                 kernel_size=3, stride=1, padding=1, bias=True):
        super.__init__()

        self.cbr_list = [
            CBR_Block(
                channel_schedule=channel_schedule,
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, bias=bias
            ),

            nn.MaxPool2d(kernel_size=pool_kernel_size)
        ]

        self.cbr_block = nn.Sequential(*self.cbr_list)

    
    def forward(self, x):
        return self.cbr_block(x) 


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

        self.cbr_block = nn.Sequential(*cbr_list)

    def forward(self, x):
        return self.cbr_block(x)

    
class CBR2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cbr(x)
    
