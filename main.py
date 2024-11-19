import torch
import torch.optim as optim

from torch.utils.data import ConcatDataset

from torchvision import datasets
from torchvision import transforms

import os.path as path

import matplotlib.pyplot as plt

from model.architecture.unet.Unet import Unet
from model.method.diffusion.DDPM import DDPM
from model.DiffusionModel import DiffusionModel

from trainer.CI_scenario import CI_scenario
from trainer.scheduler.CI import CI

from logger.FileLogger import FileLogger

from utils.convert import tensorToPIL

logger = FileLogger("log.txt")
logger.on()

# 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

image_shape = (1, 32, 32)

image_transform = transforms.Compose([
        transforms.Resize(image_shape[1:]),  # 이미지 크기를 64x64로 조정
        transforms.ToTensor(),         # 이미지를 Tensor 형식으로 변환
        transforms.Normalize(0.5, 0.5)
    ])

# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=image_transform
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform= image_transform
)

diffusion_step = 1000

t_emb_dim = image_shape[0]
unet = Unet(input_shape=image_shape, depth=5,
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

result_path = path.join(".", "result")
train_path = path.join(result_path, "train", "ci_test")

epoch = 1000
batch_size = 128

model_name = f"trained_nUnet_epoch={epoch}"
# loss_list, validation_loss = model.run(device=device, epochs=epoch, save_file_path=model_save_path)

whole_dataset = ConcatDataset([
    training_data,
    test_data
])

dataset_by_label = {}
for label in range(10):
    dataset_by_label[label] = torch.load(f"./data/MNIST/raw/byLabels/mnist_{label}.pth")

# label_schedule_list = [[5]]
label_schedule_list = [list(range(10))]
scenario = CI_scenario(model=model, 
            scheduler=CI(
                batch_size=batch_size,
                portion=(0.7, 0.2, 0.1),
                label_schedule_list=label_schedule_list,
                dataset=whole_dataset,
                dataset_by_label=dataset_by_label))

task_running_list = scenario.run(
    epochs=epoch, 
    save_file_path=train_path, save_file_name=model_name)

for i, task_result in enumerate(task_running_list):
    x = list(range(1, epoch+1))
    train, val = tuple(task_result)
    plt.plot(x, train, marker='o', linestyle='-', linewidth=2, markersize=6, label='train')
    plt.plot(x, val, marker='o', linestyle='-', linewidth=2, markersize=6, label='val')

    # 그래프에 제목과 라벨 추가
    plt.title("epoch loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    graph_path = path.join(train_path, f"task_{i+1}","loss_line_plot.png")
    # 그래프 이미지 파일로 저장
    plt.savefig(graph_path, format="png", dpi=300)  # png 형식으로 저장, 해상도 설정

    plt.clf()

x = list(range(1, len(label_schedule_list)+1))

y = [sum(i[0]) for i in task_running_list]
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label='train')

y = [sum(i[1]) for i in task_running_list]
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, label='validation')

# 그래프에 제목과 라벨 추가
plt.title("task loss")
plt.xlabel("task")
plt.ylabel("loss")
plt.legend()

graph_path = path.join(train_path, "loss_line_plot.png")
# 그래프 이미지 파일로 저장
plt.savefig(graph_path, format="png", dpi=300)  # png 형식으로 저장, 해상도 설정


test_path = path.join(result_path, "test")

sample_saved_path = path.join(test_path, "test.png")

model.load(path.join(train_path, f"task_{len(label_schedule_list)}", f"{model_name}_best"))

sample = model.generate()
image = tensorToPIL(sample[0])
image.save(sample_saved_path)

logger.off()
