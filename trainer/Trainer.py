import os
import os.path as path

from model.Model import Model
from .scheduler.base.Scheduler import Scheduler
from .strategy.base.Strategy import Strategy

class Trainer():
    def __init__(self,
                 model: Model,
                 scheduler: Scheduler,
                 strategy: Strategy):
        self.model = model
        self.scheduler = scheduler
        self.strategy = strategy


    def run(self, 
            epochs, save_file_path, save_file_name):
        task_running_list = []

        for i, task_dataset in enumerate(self.scheduler):
            print(f"task {i+1} run start")

            task_number = i+1
            task_path = path.join(save_file_path, f"task_{task_number}")

            os.makedirs(task_path, exist_ok=True)

            task_loader = self.strategy.make_loader(task_number, task_dataset)

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