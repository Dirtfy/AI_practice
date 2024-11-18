import os.path as path

import torch
from torchvision.transforms import ToPILImage

from PIL import Image

def tensorToPIL(tensor: torch.Tensor) -> Image:
    tensor = tensor.clone()
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # 정규화

    return ToPILImage()(tensor)

def dictToPath(dict: dict) -> path:
    now_path = ""
    for key, value in dict:
        now_path = path.join(now_path, f"{key}={value}")
    return now_path