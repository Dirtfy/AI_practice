import torch
from torch._tensor import Tensor
import torch.nn as nn
from torch.optim.optimizer import Optimizer


from .Category import Generator, Classifier

from model.method.diffusion.base.Diffusion import Diffusion
from model.method.classifier.Classic import Classic

from model.Model import Model

class DGRModel(Model, Generator, Classifier):
    def __init__(self,
                 generator_archtecture: nn.Module,
                 generator_method: Diffusion,
                 generator_optimizer: Optimizer,

                 solver_archtecture: nn.Module,
                 solver_method: Classic,
                 solver_optimizer: Optimizer,

                 replay_ratio=0.3):
        super().__init__()

        self.generator_architecture = generator_archtecture
        self.generator_method = generator_method
        self.generator_optimizer = generator_optimizer

        self.solver_archtecture = solver_archtecture
        self.solver_method = solver_method
        self.solver_optimizer = solver_optimizer

        self.replay_ratio = replay_ratio

        self.device = next(generator_archtecture.parameters()).device    


    def train_epoch(self, train_dataloader, total_epoch, now_epoch):
        self.generator_architecture.train()
        self.solver_archtecture.train()

        epoch_loss = 0.0
        batch_loss = 0.0

        for i, (xs, ys) in enumerate(train_dataloader):
            xs = xs.to(self.device)  # GPU로 이동
            ys = ys.to(self.device)

            self.generator_optimizer.zero_grad()
            generator_loss = self.generator_method.train_batch(self.generator_architecture, xs, ys)
            generator_loss.backward()
            self.solver_optimizer.zero_grad()
            self.generator_optimizer.step()
            
            # 모델의 학습 단계
            self.solver_optimizer.zero_grad()
            solver_loss = self.solver_method.train_batch(self.solver_archtecture, xs, ys)
            solver_loss.backward()
            self.generator_optimizer.step()
            
            loss_sum = generator_loss.item() + solver_loss.item()
            batch_loss += loss_sum
            epoch_loss += loss_sum

            if (i + 1) % 100 == 0:  # 100 배치마다 손실 출력
                print(f"Epoch [{now_epoch+1}/{total_epoch}], "+
                      f"Step [{i+1}/{len(train_dataloader)}], "+
                      f"Loss: {batch_loss/100:.4f}")
                batch_loss = 0.0

        return epoch_loss
    
    def validate_epoch(self, dataloader):
        self.solver_archtecture.eval()

        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 모델의 예측 단계
            loss = self.solver_method.train_batch(self.solver_archtecture, images, labels)
            total_loss += loss.item()

        return total_loss
    
    def test_loop(self, dataloader):
        return self.validate_epoch(dataloader)

    def generate(self, num_sample, y) -> Tensor:
        with torch.no_grad():
            return self.generator_method.generate(self.generator_architecture, num_sample, y)
    
    def inference(self, x) -> Tensor:
        with torch.no_grad():
            return self.solver_method.inference(self.solver_archtecture, x)
        
    def load(self, file_path):
        return super().load(file_path)
    
    def save(self, file_path):
        return super().save(file_path)
