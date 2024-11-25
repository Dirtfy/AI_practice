import os
import sys
import os.path as path

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from model.architecture.unet.Unet import Unet
from model.method.diffusion.DDPM import DDPM
from model.method.diffusion.base.BetaScheduler import cosine_beta_schedule

from model.architecture.CnnClassifier import CnnClassifier
from model.method.classifier.Classic import Classic

from model.DGRModel import DGRModel

from trainer.Trainer import Trainer
from trainer.scheduler.CI import CI
from trainer.strategy.ReplayWithoutLabel import ReplayWithoutLabel

from logger.FileLogger import FileLogger

from utils.convert import tensorToPIL


# config
result_path = path.join(".", "result", "dgr")
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
unet = Unet(input_shape=image_shape, depth=5,
             t_emb_dim=t_emb_dim,
             t_max_position=diffusion_step).to(device)
ddpm = DDPM(
    device=device,
    image_shape=image_shape,
    beta_schedule=cosine_beta_schedule(diffusion_step)
)

cnn = CnnClassifier(
    input_shape=image_shape, depth=5,
    num_class=10)
classifier = Classic()

model = DGRModel(
    generator_archtecture=unet,
    generator_method=ddpm,
    generator_optimizer=optim.Adam(unet.parameters(), lr=1e-4),

    solver_archtecture=cnn,
    solver_method=classifier,
    solver_optimizer=optim.Adam(unet.parameters(), lr=1e-4),

    replay_ratio=0.2
)
params = sum(p.numel() for p in model.generator_architecture.parameters()) \
    + sum(p.numel() for p in model.solver_archtecture.parameters())
print(f"Model parameters memory usage: {params * 4 / 1024 ** 2} MB")

model_name = f"trained_nUnet_epoch={epoch}"

# trainer setting

dataset_by_label = {}
for label in range(10):
    dataset_by_label[label] = torch.load(f"./data/MNIST/raw/byLabels/mnist_{label}.pth")
print(f"Dataset memory usage: {sum([sys.getsizeof(ds) for ds in dataset_by_label.values()]) / 1024 ** 2} MB")

label_schedule_list = [[0, 2, 4, 6, 8], [1, 3, 5], [7, 9]]
# label_schedule_list = [[0, 8], [1], [9]]
# label_schedule_list = [list(range(10))]
scheduler = CI(label_schedule_list=label_schedule_list,
               dataset_by_label=dataset_by_label)

strategy = ReplayWithoutLabel(
    batch_size=batch_size,
    portion=(0.7,0.2,0.1),
    generator=model,
    classifier=model,
    label_schedule_list=label_schedule_list,
    replay_percentage=0.1)

scenario = Trainer(model=model, 
            scheduler=scheduler,
            strategy=strategy)

task_running_list = scenario.run(
    epochs=epoch, 
    save_file_path=train_path, save_file_name=model_name)

# result

for task_number, task_result in enumerate(task_running_list):
    x = list(range(1, epoch+1))
    train, val = tuple(task_result)
    plt.plot(x, train, marker='o', linestyle='-', linewidth=2, markersize=6, label='train')
    plt.plot(x, val, marker='o', linestyle='-', linewidth=2, markersize=6, label='val')

    plt.title("epoch loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    task_graph_path = path.join(train_path, f"task_{task_number}","loss_line_plot.png")
    
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


model.load(path.join(train_path, f"task_{len(label_schedule_list)-1}", f"{model_name}_best"))

sample = model.generate(1, 6)
image = tensorToPIL(sample[0])
image.save(sample_saved_path)

logger.off()
