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

    def noising(self, x0, t, noise):        
        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    def denoising(self, xt, t, predicted_noise):
        # t가 0인 경우는 noise가 0이고, 그렇지 않은 경우 noise 생성
        noise = torch.zeros_like(xt)  # 기본적으로 noise를 0으로 초기화
        mask = t > 0  # t가 0보다 큰 경우에만 noise를 생성하도록 마스크를 만듬
        noise[mask] = torch.randn_like(xt)[mask]  # t > 0인 경우에만 noise를 생성

        beta_t = self.beta_schedule[t]
        alpha_t = (1-beta_t).reshape(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        sigma_t = (beta_t).reshape(-1, 1, 1, 1)

        print(xt.shape)
        print(predicted_noise.shape)
        print(t.shape)
        print(alpha_t.shape)
        print(alpha_bar_t.shape)
        print(beta_t.shape)

        assert False

        return (1/torch.sqrt(alpha_t)) * \
            (xt - \
             ((1-alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) \
                + sigma_t*noise

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
