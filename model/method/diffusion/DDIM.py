import torch

import torch.nn.functional as F

from .base.Diffusion import Diffusion

from utils.convert import tensorToPIL

class DDIM(Diffusion):
    def __init__(self, 
                device,
                image_shape,
                beta_schedule):
        super().__init__(
            device=device,
            image_shape=image_shape,
            beta_schedule=beta_schedule
        )

    def noising(self, x0, t, noise):        
        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    def q_sample(self, x0, t):
        # 노이즈 생성
        noise = torch.randn_like(x0)
        xt = self.noising(x0, t, noise)
        return xt, noise
    
    def denoising(self, xt, t, predicted_noise):
        if t > 0:
            noise = torch.randn_like(xt)
        else:
            noise = torch.zeros_like(xt)

        beta_t = self.beta_schedule[t]
        alpha_t = 1-beta_t
        alpha_bar_t = self.alpha_bars[t].reshape(1, 1, 1, 1)
        sigma_t = beta_t

        return (1/torch.sqrt(alpha_t)) * \
            (xt - \
             ((1-alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) \
                + sigma_t*noise

    def p_sample(self, architecture,  xt):
        current_image = xt

        batch_size = xt.shape[0]

        for step in reversed(range(self.diffusion_steps)):
            t = torch.full((batch_size, ), step, dtype=torch.long, device=self.device)

            predicted_noise = architecture(current_image, t)

            current_image = self.denoising(current_image, t, predicted_noise)

            if current_image.shape[0] == 1:
                import os.path as path

                t_path = path.join("result", "temp", f"t_{step}.png")

                image = tensorToPIL(current_image[0])
                image.save(t_path)

        return current_image

    def train_batch(self, architecture, x0):
        t = torch.randint(0, self.diffusion_steps, (x0.size(0),), device=x0.device).long()

        xt, noise = self.forward_diffusion(x0, t)

        predicted_noise = architecture(xt, t)

        loss = F.mse_loss(predicted_noise, noise)

        return loss

    def generate(self, architecture, num_images):
        initial_noise = torch.randn(
            num_images, *self.image_shape
            ).to(next(architecture.parameters()).device)
        
        return self.reverse_diffusion(architecture, initial_noise)
