from abc import *

import torch

class Generator(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate(self, num_sample, *condition) -> torch.Tensor:
        raise NotImplementedError
    
class Classifier(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def inference(self, x) -> torch.Tensor:
        raise NotImplementedError
