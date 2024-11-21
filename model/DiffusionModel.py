import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .Model import Model
from .method.diffusion.base.Diffusion import Diffusion
from .Category import Generator

class DiffusionModel(Model, Generator):

    def __init__(self,
                 architecture: nn.Module,
                 method: Diffusion,
                 optimizer: Optimizer):
        super().__init__()
        
        self.architecture = architecture
        self.optimizer = optimizer

        self.method = method

        self.device = next(architecture.parameters()).device


    def train_epoch(self, train_dataloader, total_epoch, now_epoch):
        self.architecture.train()

        epoch_loss = 0.0
        batch_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(self.device)  # GPU로 이동
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            
            # 모델의 학습 단계
            loss = self.method.train_batch(self.architecture, images, labels)
            
            # 손실 역전파
            loss.backward()
            self.optimizer.step()
            
            batch_loss += loss.item()
            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:  # 100 배치마다 손실 출력
                print(f"Epoch [{now_epoch+1}/{total_epoch}], "+
                      f"Step [{i+1}/{len(train_dataloader)}], "+
                      f"Loss: {batch_loss/100:.4f}")
                batch_loss = 0.0

        return epoch_loss



    # Validation 루프
    def validate_epoch(self, dataloader):
        self.architecture.eval()

        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 모델의 예측 단계
            loss = self.method.train_batch(self.architecture, images, labels)
            total_loss += loss.item()

        return total_loss

    # Test 루프
    def test_loop(self, dataloader):
        self.architecture.eval()
        
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 모델의 예측 단계
            loss = self.method.train_batch(self.architecture, images, labels)
            total_loss += loss.item()

        return total_loss
    
    def generate(self, num_sample, y) -> torch.Tensor:
        return self.method.generate(
            self.architecture, 
            num_images=num_sample,
            y=y)
    
    def save(self, file_path):
        torch.save(self.architecture.state_dict(), file_path)

    def load(self, file_path):
        self.architecture.load_state_dict(torch.load(file_path))
