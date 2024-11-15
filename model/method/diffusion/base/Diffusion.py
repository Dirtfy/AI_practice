import torch.nn as nn

from abc import *

class Diffusion(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, 
                 device,
                 image_shape,
                 diffusion_steps, 
                 beta_schedule):
        self.device = device
        
        self.image_shape = image_shape
        self.diffusion_steps = diffusion_steps
        
        self.beta_schedule = beta_schedule.to(device)

        # Time Embedding
        self.time_embedding = nn.Embedding(diffusion_steps, image_shape[0]).to(device=device)

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
