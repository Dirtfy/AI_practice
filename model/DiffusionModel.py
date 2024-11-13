import torch
import torch.nn as nn

import method.diffusion.base.GaussianDiffusion as GaussianDiffusion


class DiffusionModel():

    def __init__(self,
                 architecture: nn.Module,
                 method: GaussianDiffusion,
                 optimizer,
                 train_dataloader,
                 validation_dataloader,
                 test_dataloader):
        
        self.architecture = architecture
        self.method = method

        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def train(self, device, epochs):
        self.architecture.train()  # 모델을 학습 모드로 설정

        for epoch in range(epochs):
            running_loss = 0.0

            for i, (images, _) in enumerate(self.train_dataloader):
                images = images.to(device)  # GPU로 이동

                self.optimizer.zero_grad()
                
                # 모델의 학습 단계
                loss = self.method.training_step(self.architecture, images)
                
                # 손실 역전파
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()

                if (i + 1) % 100 == 0:  # 100 배치마다 손실 출력
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(self.train_dataloader)}], Loss: {running_loss/100:.4f}")
                    running_loss = 0.0

            print(f"Epoch [{epoch+1}/{epochs}] Training loss: {running_loss/len(self.train_dataloader):.4f}")

            # Validation step
            val_loss = self.validation_loop(device)
            print(f"Validation Loss after Epoch [{epoch+1}/{epochs}]: {val_loss:.4f}")

        print("Finished Training")

    # Validation 루프
    def validation_loop(self, device):
        self.architecture.eval()  # 평가 모드로 전환

        total_loss = 0.0
        with torch.no_grad():  # 그래디언트 계산을 하지 않음
            for images, _ in self.validation_dataloader:
                images = images.to(device)
                
                # 모델의 예측 단계
                loss = self.method.training_step(self.architecture, images)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.validation_dataloader)
        self.architecture.train()  # 다시 학습 모드로 전환
        return avg_loss

    # Test 루프
    def test_loop(self, device):
        self.architecture.eval()  # 평가 모드로 전환
        total_loss = 0.0
        with torch.no_grad():  # 그래디언트 계산을 하지 않음
            for images, _ in self.test_dataloader:
                images = images.to(device)
                
                # 모델의 예측 단계
                loss = self.method.training_step(self.architecture, images)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_dataloader)
        print(f"Test Loss: {avg_loss:.4f}")