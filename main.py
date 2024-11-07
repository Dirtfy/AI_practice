import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

import UNet

# 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize((64, 64)),  # 이미지 크기를 64x64로 조정
        transforms.ToTensor()         # 이미지를 Tensor 형식으로 변환
    ]),
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform= transforms.Compose([
        transforms.Resize((64, 64)),  # 이미지 크기를 64x64로 조정
        transforms.ToTensor()         # 이미지를 Tensor 형식으로 변환
    ]),
)

batch_size = 64

# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

model = UNet.ConditionalUnet(channel=1, max_channel=1024).to(device)
print(model)
for X, y in test_dataloader:
    print(model(X.to(device)))
    break