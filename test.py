import torch
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToPILImage
from torchvision import transforms

import os.path as path

import matplotlib.pyplot as plt

from model.architecture.Unet import Unet
from model.embedding.SinusoidalPositionEmbedding import SinusoidalPositionEmbedding
from model.architecture.Unet2 import UNet as Unet2
from model.architecture.unet.Unet import Unet as nUnet

from model.method.diffusion.DDPM import DDPM
from model.DiffusionModel import DiffusionModel

from dataloader.SplitedDataSet import SplitedDataSet
from dataloader.SplitedDataLoader import SplitedDataSetLoader

from trainer.CI_scenario import CI_scenario
from trainer.scheduler.CI import CI

from logger.FileLogger import FileLogger

from utils.convert import tensorToPIL

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

image_shape = (1, 32, 32)

diffusion_step = 1000

t_emb_dim = image_shape[0]
unet = nUnet(shape=image_shape, depth=5,
             t_emb_dim=t_emb_dim,
             time_embedding=SinusoidalPositionEmbedding(
                 device=device,
                 dim=t_emb_dim,max_position=diffusion_step)).to(device)

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

model.load("/home/mskim/project/AI_practice/result/train/best")

sample = model.generate()
print(sample.min())
print(sample.max())
print(sample)
image = tensorToPIL(sample[0])
image.save("/home/mskim/project/AI_practice/result/tt.png")