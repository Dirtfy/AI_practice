import os

import torch

from torchvision import datasets
# from torchvision import transforms

# image_shape = (1, 32, 32)

# image_transform = transforms.Compose([
#         transforms.Resize(image_shape[1:]),  # 이미지 크기를 64x64로 조정
#         transforms.ToTensor(),         # 이미지를 Tensor 형식으로 변환
#         transforms.Normalize(0.5, 0.5)
#     ])

# training_data = datasets.MNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=image_transform
# )



# label_list = torch.tensor([label for _, label in training_data]).unique()
# print(label_list)

# tt = {"d":1, "e":2}
# for key, value in tt.items():
#     print(key, value)

os.makedirs(os.path.join(".", "dofn", "aos"))