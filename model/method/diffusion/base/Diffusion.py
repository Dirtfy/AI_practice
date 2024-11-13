import torch.nn as nn

from abc import *

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, diffusion_steps):
        super(TimeEmbedding, self).__init__()
        # nn.Embedding을 사용하여 시간 단계 t를 임베딩 벡터로 변환
        self.embedding = nn.Embedding(diffusion_steps, embedding_dim)

    def forward(self, t):
        # t는 배치 크기만큼의 diffusion step
        return self.embedding(t)  # t를 임베딩 벡터로 변환

class Diffusion(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, 
                 device,
                 image_shape,
                 diffusion_steps, 
                 beta_schedule, 
                 time_embedder=None):
        self.image_shape = image_shape
        self.diffusion_steps = diffusion_steps
        
        self.beta_schedule = beta_schedule.to(device)

        # Time Embedding
        self.time_embedding = TimeEmbedding(image_shape[0], diffusion_steps).to(device=device) if time_embedder == None else time_embedder

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
