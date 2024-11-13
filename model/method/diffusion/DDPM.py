import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../base'))
print(f"Adding base path to sys.path: {base_path}")
sys.path.append(base_path)

from base.GaussianDiffusion import GaussianDiffusion

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, diffusion_steps):
        super(TimeEmbedding, self).__init__()
        # nn.Embedding을 사용하여 시간 단계 t를 임베딩 벡터로 변환
        self.embedding = nn.Embedding(diffusion_steps, embedding_dim)

    def forward(self, t):
        # t는 배치 크기만큼의 diffusion step
        return self.embedding(t)  # t를 임베딩 벡터로 변환


class DDPM(GaussianDiffusion):
    def __init__(self, 
                 image_shape, 
                 diffusion_steps, 
                 beta_schedule,
                 time_embedder=None):

        self.image_shape = image_shape
        self.diffusion_steps = diffusion_steps
        
        self.beta_schedule = beta_schedule
        self.alphas = 1.0 - self.beta_schedule
        self.alpha_bars = torch.cumprod(self.alphas, 0)
        
        # Time Embedding
        self.time_embedding = TimeEmbedding(128, diffusion_steps) if time_embedder == None else time_embedder

    def forward_diffusion(self, x0, t):
        # 노이즈 생성
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return xt, noise

    def reverse_diffusion(self, architecture,  xt):
        current_image = xt
        for step in reversed(range(self.diffusion_steps)):
            t = torch.full((xt.size(0),), step, dtype=torch.long, device=xt.device)
            time_emb = self.time_embedding(t)  # 시간 임베딩 계산
            # 이미지와 시간 임베딩을 결합 (채널 차원에서 concat)
            current_image_and_emb = torch.cat([
                current_image, 
                time_emb\
                    .reshape(time_emb.size(0), -1, 1, 1)\
                    .expand(-1, -1, current_image.size(2), current_image.size(3))
                ], dim=1)
            predicted_noise = architecture(current_image_and_emb)  # 결합된 입력을 architecture에 전달
            alpha_bar_t = self.alpha_bars[step].reshape(-1, 1, 1, 1)
            current_image = (current_image - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        return current_image

    def training_step(self, architecture, x0):
        t = torch.randint(0, self.diffusion_steps, (x0.size(0),), device=x0.device).long()
        xt, noise = self.forward_diffusion(x0, t)
        
        time_emb = self.time_embedding(t)  # 시간 임베딩 계산
        # 이미지와 시간 임베딩을 결합 (채널 차원에서 concat)
        xt_and_emb = torch.cat([
            xt, 
            time_emb\
                .reshape(time_emb.size(0), -1, 1, 1)\
                .expand(-1, -1, xt.size(2), xt.size(3))
            ], dim=1)
        predicted_noise = architecture(xt_and_emb)  # 결합된 입력을 architecture에 전달
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def generate(self, architecture, num_images):
        initial_noise = torch.randn(
            num_images, 3, 
            self.image_shape[0], self.image_shape[1]
            ).to(next(architecture.parameters()).device)
        return self.reverse_diffusion(initial_noise)
