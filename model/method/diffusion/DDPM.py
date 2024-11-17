import torch

import torch.nn.functional as F

from .base.Diffusion import Diffusion

from utils.convert import tensorToPIL

class DDPM(Diffusion):
    def __init__(self, 
                 device,
                 image_shape, 
                 diffusion_steps, 
                 beta_schedule):
        super().__init__(
            device=device,
            image_shape=image_shape,
            diffusion_steps=diffusion_steps,
            beta_schedule=beta_schedule
        )

        assert len(beta_schedule) == diffusion_steps

        alphas = 1.0 - self.beta_schedule
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def noising(self, x0, alpha_bar_t, noise):        
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    def forward_diffusion(self, x0, t):
        # 노이즈 생성
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        xt = self.noising(x0, alpha_bar_t, noise)
        return xt, noise
    
    def denoising(self, xt, alpha_bar_t, noise):        
        return (xt - torch.sqrt(1 - alpha_bar_t) * noise) / torch.sqrt(alpha_bar_t)

    def reverse_diffusion(self, architecture,  xt):
        current_image = xt

        batch_size = xt.shape[0]

        for step in reversed(range(self.diffusion_steps)):
            t = torch.full((batch_size, ), step, dtype=torch.long, device=self.device)

            predicted_noise = architecture(current_image, t)

            alpha_bar_t = self.alpha_bars[step].reshape(1, 1, 1, 1)
            current_image = self.denoising(current_image, alpha_bar_t, predicted_noise)
            
            current_image = current_image.flatten(2)
            current_image = F.normalize(current_image, dim=2)
            current_image = current_image.reshape(batch_size, 1, xt.shape[2], xt.shape[3])


            if current_image.shape[0] == 1:
                import os.path as path

                t_path = path.join("result", "temp", f"t_{step}.png")

                image = tensorToPIL(current_image[0])
                image.save(t_path)

        return current_image

    def training_step(self, architecture, x0):
        t = torch.randint(0, self.diffusion_steps, (x0.size(0),), device=x0.device).long()

        xt, noise = self.forward_diffusion(x0, t)

        predicted_noise = architecture(xt, t)

        loss = F.mse_loss(predicted_noise, noise)

        lv = loss.item()

        if lv < 0.005:
            import os.path as path

            t_prev_path = path.join("result", "sss", f"t_{t[0]}_{lv}_prev.png")
            t_path = path.join("result", "sss", f"t_{t[0]}_{lv}.png")

            image_prev = tensorToPIL(xt[0])
            image_prev.save(t_prev_path)
            image = tensorToPIL((xt-predicted_noise)[0])
            image.save(t_path)

        return loss

    def generate(self, architecture, num_images):
        initial_noise = torch.randn(
            num_images, *self.image_shape
            ).to(next(architecture.parameters()).device)
        
        return self.reverse_diffusion(architecture, initial_noise)
