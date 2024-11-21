import torch
import torch.nn as nn

from abc import *

class Diffusion(metaclass=ABCMeta):
    
    def __init__(self, 
                 device,
                 image_shape,
                 beta_schedule):
        self.device = device
        
        self.image_shape = image_shape
        
        self.diffusion_steps = len(beta_schedule)
        self.beta_schedule = beta_schedule.to(device)
        alphas = 1.0 - self.beta_schedule
        self.alpha_bars = torch.cumprod(alphas, dim=0)


    @abstractmethod
    def q_sample(self, x0, t):
        raise NotImplementedError

    @abstractmethod
    def p_sample(self, architecture, xt, y):
        raise NotImplementedError

    @abstractmethod
    def train_batch(self, architecture, x0, y):
        raise NotImplementedError

    @abstractmethod
    def generate(self, architecture, num_images, y):
        raise NotImplementedError
