import torch
import torch.optim as optim

from torch.utils.data import ConcatDataset

from torchvision import datasets
from torchvision import transforms

import os
import os.path as path

import matplotlib.pyplot as plt

from dataloader.SplitedDataLoader import SplitedDataLoader

from model.architecture.unet.ConditionalUnet import ConditionalUnet
from model.method.diffusion.ConditionalDDPM import ConditionalDDPM
from model.architecture.unet.Unet import Unet
from model.method.diffusion.DDPM import DDPM
from model.method.diffusion.base.BetaScheduler import cosine_beta_schedule
from model.DiffusionModel import DiffusionModel

from logger.FileLogger import FileLogger

from utils.convert import tensorToPIL


# config
result_path = path.join(".", "result", "condition")
train_path = path.join(result_path, "train")
test_path = path.join(result_path, "test")

graph_path = path.join(train_path, "loss_line_plot.png")
sample_saved_path = path.join(test_path, "test.png")

image_shape = (1, 32, 32)
epoch = 100
batch_size = 128

os.makedirs(train_path)
os.makedirs(test_path)

logger = FileLogger(
    file_path=result_path,
    file_name="log.txt")
logger.on()

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# dataset setting
image_transform = transforms.Compose([
        transforms.Resize(image_shape[1:]),
        transforms.ToTensor()
    ])

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=image_transform
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform= image_transform
)

whole_dataset = ConcatDataset([
    training_data,
    test_data
])

dataset_loader = SplitedDataLoader(
    batch_size=batch_size,
    portion=(0.7,0.2,0.1),
    dataset=whole_dataset
)


# model setting
diffusion_step = 1000

t_emb_dim = image_shape[0]
c_emb_dim = image_shape[0]
unet = ConditionalUnet(
    input_shape=image_shape, 
    depth=5,
    t_emb_dim=t_emb_dim,
    t_max_position=diffusion_step,
    num_class=10,
    c_emb_dim=c_emb_dim).to(device)
# unet = Unet(
#     input_shape=image_shape, 
#     depth=5,
#     t_emb_dim=t_emb_dim,
#     t_max_position=diffusion_step).to(device)
ddpm = ConditionalDDPM(
    device=device,
    image_shape=image_shape,
    beta_schedule=cosine_beta_schedule(diffusion_step)
)
# ddpm = DDPM(
#     device=device,
#     image_shape=image_shape,
#     beta_schedule=cosine_beta_schedule(diffusion_step)
# )
model = DiffusionModel(
    architecture=unet,
    method=ddpm,
    optimizer=optim.Adam(unet.parameters(), lr=1e-4)
)
params = sum(p.numel() for p in model.architecture.parameters())
print(f"Model parameters memory usage: {params * 4 / 1024 ** 2} MB")

model_name = f"trained_nUnet_epoch={epoch}"

loss_list, validation_loss = model.run(
    splited_dataloader=dataset_loader,
    epochs=epoch,
    save_file_path=train_path,
    save_file_name=model_name
)

# loss plot
x = list(range(1, epoch+1))

train = loss_list
plt.plot(x, train, marker='o', linestyle='-', linewidth=2, markersize=6, label='train')
val = validation_loss
plt.plot(x, val, marker='o', linestyle='-', linewidth=2, markersize=6, label='val')

plt.title("epoch loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()

plt.savefig(graph_path, format="png", dpi=300)


# sampling

model.load(path.join(train_path, f"{model_name}_best"))

sample = model.generate(num_sample=1,y=6)
image = tensorToPIL(sample[0])
image.save(sample_saved_path)

logger.off()
