import os
import os.path as path

from model.Model import Model
from .scheduler.CI import CI
from dataloader.SplitedDataLoader import SplitedDataLoader

class CI_scenario():
    def __init__(self,
                 model: Model,
                 scheduler: CI):
        self.model = model
        self.scheduler = scheduler

    def run(self,
            epochs, save_file_path, save_file_name):
        task_running_list = []

        for i, task_loader in enumerate(self.scheduler):
            print(f"task {i+1} run start")

            task_number = i+1
            task_path = path.join(save_file_path, f"task_{task_number}")

            os.makedirs(task_path, exist_ok=True)

            loss_list, validatioin_list = self.model.run(
                splited_dataloader=task_loader,
                epochs=epochs, 
                save_file_path=task_path,
                save_file_name=save_file_name)
            
            task_running_list.append([
                loss_list,
                validatioin_list
            ])
            
            print(f"task {i+1} run finish")

        return task_running_list