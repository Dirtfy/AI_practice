import os.path as path

from model.Model import Model
from .scheduler.CI import CI

class CI_scenario():
    def __init__(self,
                 model: Model,
                 scheduler: CI):
        self.model = model
        self.scheduler = scheduler

    def run(self, device, epochs, save_file_path):
        task_running_list = []

        for i, task_loader in enumerate(self.scheduler):
            print(f"task {i+1} run start")

            self.model.dataset_loader = task_loader

            task_number = i+1
            save_file_name = f"task_{task_number}_model"

            fianl_save_file_path = path.join(save_file_path, save_file_name)

            loss_list, validatioin_list = self.model.run(
                device=device, 
                epochs=epochs, 
                save_file_path=fianl_save_file_path)
            
            task_running_list.append([
                loss_list,
                validatioin_list
            ])
            
            print(f"task {i+1} run finish")

        return task_running_list