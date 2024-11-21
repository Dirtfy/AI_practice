import torch
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToPILImage
from torchvision import transforms

import os.path as path

import matplotlib.pyplot as plt

from model.architecture.embedding.SinusoidalPositionEmbedding import SinusoidalPositionEmbedding
from model.architecture.unet.Unet import Unet as nUnet

from model.method.diffusion.DDPM import DDPM
from model.DiffusionModel import DiffusionModel


from trainer.Trainer import CI_scenario
from trainer.scheduler.CI import CI

from logger.FileLogger import FileLogger

from utils.convert import tensorToPIL

device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

image_shape = (1, 32, 32)

diffusion_step = 1000

t_emb_dim = image_shape[0]
unet = nUnet(input_shape=image_shape, depth=5,
             t_emb_dim=t_emb_dim,
             t_max_position=diffusion_step).to(device)

ddpm = DDPM(
    device=device,
    image_shape=image_shape,
    diffusion_steps=diffusion_step,
    beta_schedule=torch.linspace(0.0001, 0.02, diffusion_step)
)

model = DiffusionModel(
    architecture=unet,
    method=ddpm,
    optimizer=optim.Adam(unet.parameters(), lr=1e-4)
)

model.load("/home/mskim/project/AI_practice/result/train/ci_test/task_1/trained_nUnet_epoch=1000")

sample = model.generate()
print(sample.min())
print(sample.max())
print(sample)
image = tensorToPIL(sample[0])
image.save("/home/mskim/project/AI_practice/result/tt.png")