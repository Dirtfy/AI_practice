import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from model.architecture.Unet import Unet
from model.method.diffusion.DDPM import DDPM
from model.DiffusionModel import DiffusionModel


# 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

image_shape = (64, 64)

image_transform = transforms.Compose([
        transforms.Resize(image_shape),  # 이미지 크기를 64x64로 조정
        transforms.ToTensor()         # 이미지를 Tensor 형식으로 변환
    ])

# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=image_transform
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
validation_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform= image_transform
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform= image_transform
)

batch_size = 64

# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

unet = Unet(channel=1, max_channel=1024).to(device)
print(unet)
for X, y in test_dataloader:
    print(unet(X.to(device)))
    break

diffusion_step = 1000
ddpm = DDPM(
    image_shape=image_shape,
    diffusion_steps=diffusion_step,
    beta_schedule=torch.linspace(0.0001, 0.02, diffusion_step)
)

model = DiffusionModel(
    architecture=unet,
    method=ddpm,
    optimizer=optim.Adam(unet.parameters(), lr=1e-4),
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    test_dataloader=test_dataloader
)

model.train(device=device)