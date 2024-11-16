import torch
from torchvision.transforms import ToPILImage

from PIL import Image

def tensorToPIL(tensor: torch.Tensor) -> Image:
    tensor = tensor.clone()
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # 정규화

    return ToPILImage()(tensor)

