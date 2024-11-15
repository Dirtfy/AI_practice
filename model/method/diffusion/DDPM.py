import torch

import torch.nn.functional as F

from .base.Diffusion import Diffusion


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
            time_emb = self.time_embedding(t)  # 시간 임베딩 계산
            time_emb_dim_size = time_emb.shape[1]

            # 이미지와 시간 임베딩을 결합 (채널 차원에서 concat)
            current_image_and_emb = torch.cat([
                current_image, 
                time_emb\
                    .reshape(-1, time_emb_dim_size, 1, 1)\
                    .expand(-1, -1, current_image.shape[2], current_image.shape[3])
                ], dim=1)
            
            predicted_noise = architecture(current_image_and_emb)  # 결합된 입력을 architecture에 전달

            alpha_bar_t = self.alpha_bars[step].reshape(1, 1, 1, 1)
            current_image = self.denoising(current_image, alpha_bar_t, predicted_noise)

            if current_image.shape[0] == 1:
                import os.path as path
                from torchvision.transforms import ToPILImage

                t_path = path.join("result", "temp", f"t_{step}.png")
                tf_pil = ToPILImage()
                image = tf_pil(current_image.squeeze(dim=0))
                image.save(t_path)

        return current_image

    def training_step(self, architecture, x0):
        t = torch.randint(0, self.diffusion_steps, (x0.size(0),), device=x0.device).long()
        time_emb = self.time_embedding(t)  # 시간 임베딩 계산
        time_emb_dim_size = time_emb.shape[1]

        xt, noise = self.forward_diffusion(x0, t)
        
        # 이미지와 시간 임베딩을 결합 (채널 차원에서 concat)
        xt_and_emb = torch.cat([
            xt, 
            time_emb\
                .reshape(-1, time_emb_dim_size, 1, 1)\
                .expand(-1, -1, xt.shape[2], xt.shape[3])
            ], dim=1)
        
        predicted_noise = architecture(xt_and_emb)  # 결합된 입력을 architecture에 전달

        loss = F.mse_loss(predicted_noise, noise)

        return loss

    def generate(self, architecture, num_images):
        initial_noise = torch.randn(
            num_images, *self.image_shape
            ).to(next(architecture.parameters()).device)
        
        return self.reverse_diffusion(architecture, initial_noise)
