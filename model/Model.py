import torch
import torch.nn as nn

from abc import *

from ..dataloader.SplitedDataSetLoader import SplitedDataSetLoader

class Model(metaclass=ABCMeta):

    def __init__(self,
                 architecture: nn.Module,
                 dataset_loader: SplitedDataSetLoader):
        
        self.architecture = architecture
        self.dataset_loader = dataset_loader

    def train(self, device, epochs):
        self.architecture.train()  # 모델을 학습 모드로 설정

        epoch_loss_list = []
        epoch_validation_loss_list = []

        for epoch in range(epochs):
            epoch_loss = self.train_loop(
                total_epoch=epochs, 
                now_epoch=epoch, 
                device=device
                )
            
            epoch_loss_list.append(epoch_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] Training loss: {epoch_loss/len(self.dataset_loader.train):.4f}")

            # Validation step
            val_loss = self.validate(device)
            epoch_validation_loss_list.append(val_loss)
            print(f"Validation Loss after Epoch [{epoch+1}/{epochs}]: {val_loss:.4f}")

        print("Finished Training")

        return epoch_loss_list, epoch_validation_loss_list

    @abstractmethod
    def train_loop(self, total_epoch, now_epoch, device):
        pass

    def validate(self, device):
        self.architecture.eval()  # 평가 모드로 전환

        with torch.no_grad():  # 그래디언트 계산을 하지 않음
            total_loss = self.validate_loop(device=device)
        
        avg_loss = total_loss / len(self.dataset_loader.validatioin)
        self.architecture.train()  # 다시 학습 모드로 전환
        return avg_loss
    
    @abstractmethod
    def validate_loop(self, device):
        pass

    def test(self, device):
        self.architecture.eval()  # 평가 모드로 전환
        
        with torch.no_grad():  # 그래디언트 계산을 하지 않음
            total_loss = self.test_loop(device=device)
        
        avg_loss = total_loss / len(self.dataset_loader.test)
        print(f"Test Loss: {avg_loss:.4f}")

    @abstractmethod
    def test_loop(self, device):
        pass


    def save(self, file_name):
        torch.save(self.architecture.state_dict(), file_name)
    

    def load(self, file_name):
        self.architecture.load_state_dict(torch.load(file_name))


    def run(self, device, epochs, file_name):
        self.train(device, epochs)

        self.test(device)

        self.save(file_name)