from abc import *


class GaussianDiffusion(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, 
                 image_size,
                 diffusion_steps, 
                 beta_schedule, 
                 time_embedder=None):
        pass

    @abstractmethod
    def forward_diffusion(self, x0, t):
        pass

    @abstractmethod
    def reverse_diffusion(self, architecture, xt):
        pass

    @abstractmethod
    def training_step(self, architecture, x0):
        pass

    @abstractmethod
    def generate(self, architecture, num_images):
        pass
