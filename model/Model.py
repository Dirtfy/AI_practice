import os.path as path

import torch
import torch.nn as nn

from abc import *

from dataloader.SplitedDataLoader import SplitedDataLoader

class Model(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    def train(self, train_dataloader, validation_dataloader, 
              epochs, 
              save_name,
              save_path):

        epoch_loss_list = []
        epoch_validation_loss_list = []

        for epoch in range(epochs):
            epoch_loss = self.train_epoch(
                train_dataloader=train_dataloader,
                total_epoch=epochs, 
                now_epoch=epoch
                )
            
            train_data_size = len(train_dataloader)
            avg_loss = epoch_loss / train_data_size
            epoch_loss_list.append(avg_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] Training loss: {epoch_loss/train_data_size:.4f}")

            # Validation step
            val_loss = self.validate(dataloader=validation_dataloader)
            epoch_validation_loss_list.append(val_loss)

            if min(epoch_validation_loss_list) == val_loss:
                best_path = path.join(save_path, save_name+"_best")
                self.save(best_path)

            # epoch_save_path = path.join(save_path, save_name+f"_epoch_{epoch}")
            # self.save(epoch_save_path)

            print(f"Validation Loss after Epoch [{epoch+1}/{epochs}]: {val_loss:.4f}")

        print("Finished Training")

        return epoch_loss_list, epoch_validation_loss_list

    @abstractmethod
    def train_epoch(self, train_dataloader, total_epoch, now_epoch):
        raise NotImplementedError

    def validate(self, dataloader):

        with torch.no_grad():  # 그래디언트 계산을 하지 않음
            total_loss = self.validate_epoch(dataloader=dataloader)
        
        avg_loss = total_loss / len(dataloader)

        return avg_loss
    
    @abstractmethod
    def validate_epoch(self, dataloader):
        raise NotImplementedError

    def test(self, dataloader):
        
        with torch.no_grad():  # 그래디언트 계산을 하지 않음
            total_loss = self.test_loop(dataloader=dataloader)
        
        avg_loss = total_loss / len(dataloader)
        print(f"Test Loss: {avg_loss:.4f}")

    @abstractmethod
    def test_loop(self, dataloader):
        raise NotImplementedError

    @abstractmethod
    def save(self, file_path):
        raise NotImplementedError
    
    @abstractmethod
    def load(self, file_path):
        raise NotImplementedError


    def run(self,
            splited_dataloader: SplitedDataLoader,
            epochs, save_file_path, save_file_name):
        
        epoch_loss_list, validation_loss_list = self.train(
            train_dataloader=splited_dataloader.train, 
            validation_dataloader=splited_dataloader.validatioin,
            epochs=epochs, 
            save_name=save_file_name,
            save_path=save_file_path)

        self.test(dataloader=splited_dataloader.test)

        final_save_path = path.join(save_file_path, save_file_name)
        self.save(final_save_path)

        return epoch_loss_list, validation_loss_list