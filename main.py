import torch
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToPILImage
from torchvision import transforms

import os.path as path

from model.architecture.Unet import Unet
from model.method.diffusion.DDPM import DDPM
from model.DiffusionModel import DiffusionModel
from model.DataSetLoader import DataSetLoader


# 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

image_shape = (1, 64, 64)

image_transform = transforms.Compose([
        transforms.Resize(image_shape[1:]),  # 이미지 크기를 64x64로 조정
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
dataset_loader = DataSetLoader(
    batch_size=batch_size,
    train_data=training_data,
    validatioin_data=validation_data,
    test_data=test_data
)

unet = Unet(channel=image_shape[0]*2, max_channel=1024).to(device)

diffusion_step = 1000
ddpm = DDPM(
    device=device,
    image_shape=image_shape,
    diffusion_steps=diffusion_step,
    beta_schedule=torch.linspace(0.0001, 0.02, diffusion_step)
)

model = DiffusionModel(
    architecture=unet,
    method=ddpm,
    optimizer=optim.Adam(unet.parameters(), lr=1e-4),
    dataset_loader=dataset_loader
)

result_path = path.join(".", "result")
train_path = path.join(result_path, "train")
test_path = path.join(result_path, "test")
architecture_saved_path = path.join(train_path, "trained")
sample_saved_path = path.join(test_path, "test.png")

try:
    model.load(architecture_saved_path)
except:
    model.train(device=device, epochs=1)
    model.save(architecture_saved_path)

model.test(device=device)

sample = ddpm.generate(unet, 1)
print(type(sample))
tf_pil = ToPILImage()
image = tf_pil(sample.squeeze(dim=0))
print(type(image))
image.save(sample_saved_path)
