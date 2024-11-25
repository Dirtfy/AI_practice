import torch

import torch.nn.functional as F

from .base.Diffusion import Diffusion

from utils.convert import tensorToPIL

class Mock(Diffusion):
    def __init__(self, 
                 device,
                 image_shape,
                 beta_schedule):
        super().__init__(
            device=device,
            image_shape=image_shape,
            beta_schedule=beta_schedule
        )

    def q_sample(self, x0, t):
        # 노이즈 생성
        noise = torch.randn_like(x0)
        xt = self.noising(x0, t, noise)
        return xt, noise

    def p_sample(self, architecture,  xt, y):
        pass

    def train_batch(self, architecture, x0, y):
        t = torch.randint(0, self.diffusion_steps, (x0.size(0),), device=x0.device).long()

        xt, noise = self.q_sample(x0, t)

        predicted_noise = architecture(xt, t, y)

        loss = F.mse_loss(predicted_noise, noise)

        return loss

    def generate(self, architecture, num_images, y):
        initial_noise = torch.randn(
            num_images, *self.image_shape
            ).to(self.device)
        
        y = torch.tensor([y for _ in range(num_images)]).to(self.device)
        
        return initial_noise
