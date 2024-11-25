import torch
import torch.optim as optim

from torch.utils.data import ConcatDataset

from torchvision import datasets
from torchvision import transforms

import os
import sys
import os.path as path

import matplotlib.pyplot as plt

from dataloader.SplitedDataLoader import SplitedDataLoader

from model.architecture.unet.DDPMUnet import Unet
from model.architecture.unet.ConditionalUnet import ConditionalUnet
from model.method.diffusion.DDPM import DDPM
from model.method.diffusion.ConditionalDDPM import ConditionalDDPM
from model.method.diffusion.base.BetaScheduler import cosine_beta_schedule
from model.DiffusionModel import DiffusionModel

from trainer.Trainer import Trainer
from trainer.scheduler.CI import CI
from trainer.strategy.Replay import Replay
from trainer.strategy.Naive import Naive

from logger.FileLogger import FileLogger

from utils.convert import tensorToPIL


# config
result_path = path.join(".", "result", "ci")
train_path = path.join(result_path, "train")
test_path = path.join(result_path, "test")

graph_path = path.join(train_path, "loss_line_plot.png")
sample_saved_path = path.join(test_path, "test.png")

image_shape = (1, 32, 32)
epoch = 1
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


# model setting
diffusion_step = 1000

t_emb_dim = image_shape[0]
c_emb_dim = image_shape[0]
unet = ConditionalUnet(input_shape=image_shape, depth=5,
             t_emb_dim=t_emb_dim,
             t_max_position=diffusion_step,
             num_class=10,
             c_emb_dim=c_emb_dim).to(device)
ddpm = ConditionalDDPM(
    device=device,
    image_shape=image_shape,
    beta_schedule=cosine_beta_schedule(diffusion_step)
)
model = DiffusionModel(
    architecture=unet,
    method=ddpm,
    optimizer=optim.Adam(unet.parameters(), lr=1e-4)
)
params = sum(p.numel() for p in model.architecture.parameters())
print(f"Model parameters memory usage: {params * 4 / 1024 ** 2} MB")

model_name = f"trained_nUnet_epoch={epoch}"

dataset_by_label = {}
for label in range(10):
    dataset_by_label[label] = torch.load(f"./data/MNIST/raw/byLabels/mnist_{label}.pth")
print(f"Dataset memory usage: {sum([sys.getsizeof(ds) for ds in dataset_by_label.values()]) / 1024 ** 2} MB")

label_schedule_list = [[0, 2, 4, 6, 8], [1, 3,5], [7, 9]]
# label_schedule_list = [[0, 8], [1], [9]]
# label_schedule_list = [list(range(10))]
scheduler = CI(label_schedule_list=label_schedule_list,
               dataset_by_label=dataset_by_label)

strategy = Replay(
    batch_size=batch_size,
    portion=(0.7,0.2,0.1),
    generator=model,
    label_schedule_list=label_schedule_list,
    replay_percentage=0.2)

# strategy = Naive(
#     batch_size=batch_size,
#     portion=(0.7,0.2,0.1))

scenario = Trainer(model=model, 
            scheduler=scheduler,
            strategy=strategy)

task_running_list = scenario.run(
    epochs=epoch, 
    save_file_path=train_path, save_file_name=model_name)

for i, task_result in enumerate(task_running_list):
    x = list(range(1, epoch+1))
    train, val = tuple(task_result)
    plt.plot(x, train, marker='o', linestyle='-', linewidth=2, markersize=6, label='train')
    plt.plot(x, val, marker='o', linestyle='-', linewidth=2, markersize=6, label='val')

    plt.title("epoch loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    task_graph_path = path.join(train_path, f"task_{i+1}","loss_line_plot.png")
    
    plt.savefig(task_graph_path, format="png", dpi=300)

    plt.clf()

x = list(range(1, len(label_schedule_list)+1))

y = [sum(i[0]) for i in task_running_list]
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label='train')

y = [sum(i[1]) for i in task_running_list]
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label='validation')

plt.title("task loss")
plt.xlabel("task")
plt.ylabel("loss")
plt.legend()

plt.savefig(graph_path, format="png", dpi=300)


model.load(path.join(train_path, f"task_{len(label_schedule_list)}", f"{model_name}_best"))

sample = model.generate()
image = tensorToPIL(sample[0])
image.save(sample_saved_path)

logger.off()
