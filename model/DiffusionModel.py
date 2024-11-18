import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .Model import Model
from .method.diffusion.base.Diffusion import Diffusion

class DiffusionModel(Model):

    def __init__(self,
                 architecture: nn.Module,
                 method: Diffusion,
                 optimizer: Optimizer):
        super().__init__(
            architecture=architecture
            )
        
        self.method = method

        self.optimizer = optimizer

    def train_loop(self, train_dataloader, total_epoch, now_epoch):
        epoch_loss = 0.0
        batch_loss = 0.0

        for i, (images, _) in enumerate(train_dataloader):
            images = images.to(self.device)  # GPU로 이동

            self.optimizer.zero_grad()
            
            # 모델의 학습 단계
            loss = self.method.training_step(self.architecture, images)
            
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
    def validate_loop(self, dataloader):
        total_loss = 0.0
        for images, _ in dataloader:
            images = images.to(self.device)
            
            # 모델의 예측 단계
            loss = self.method.training_step(self.architecture, images)
            total_loss += loss.item()

        return total_loss

    # Test 루프
    def test_loop(self, dataloader):
        
        total_loss = 0.0
        for images, _ in dataloader:
            images = images.to(self.device)
            
            # 모델의 예측 단계
            loss = self.method.training_step(self.architecture, images)
            total_loss += loss.item()

        return total_loss
    
    def generate(self):
        return self.method.generate(self.architecture, 1)
