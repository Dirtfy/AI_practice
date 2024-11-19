import torch.nn as nn
from torch.optim.optimizer import Optimizer

from model.method.dgr.Generator import Generator
from model.method.dgr.Solver import Solver
from model.Model import Model

class DGRModel(Model):
    def __init__(self,
                 generator_archtecture: nn.Module,
                 generator_method: Generator,
                 generator_optimizer: Optimizer,

                 solver_archtecture: nn.Module,
                 solver_method: Solver,
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

        self.is_first_task = True

    def train_g(self):
        pass
    def train_s(self):
        pass
    


    def train_epoch(self, train_dataloader, total_epoch, now_epoch):
        self.generator_architecture.train()
        self.solver_archtecture.train()

        epoch_loss = 0.0
        batch_loss = 0.0

        label_list = []
        for _, labels in train_dataloader:
            label_list.extend(labels.tolist())
        label_list = list(set(label_list))
        
        batch_size = train_dataloader[0][0][0]
        total_dataset_size = len(train_dataloader)*batch_size

        for _ in range(int(total_dataset_size*self.replay_ratio)):
            self.generator_method.generate(self.generator_architecture, None)
        
        self.train_s()

        for i, (xs, ys) in enumerate(train_dataloader):
            xs = xs.to(self.device)  # GPU로 이동

            num_replay = batch_size*self.replay_ratio
            
            self.generator_method.generate(self.generator_architecture, ys, num_replay)

            self.solver_optimizer.zero_grad()
            
            # 모델의 학습 단계
            loss = self.solver_method.train_batch(self.solver_archtecture, xs, ys)
            
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

        self.train_g()

        return epoch_loss
    
    def validate_epoch(self, dataloader):
        return super().validate_epoch(dataloader)
    
    def test_loop(self, dataloader):
        return super().test_loop(dataloader)

