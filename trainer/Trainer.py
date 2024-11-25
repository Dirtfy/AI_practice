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
            epochs, save_file_path, save_file_name,
            before_run=lambda x: None,
            after_run=lambda x: None):
        task_running_list = []

        for task_number, task_dataset in enumerate(self.scheduler):
            print(f"task {task_number} run start")

            task_path = path.join(save_file_path, f"task_{task_number}")

            os.makedirs(task_path, exist_ok=True)
            
            task_loader = self.strategy.make_loader(task_number, task_dataset)

            before_run(task_number)

            loss_list, validatioin_list = self.model.run(
                splited_dataloader=task_loader,
                epochs=epochs, 
                save_file_path=task_path,
                save_file_name=save_file_name)
            
            after_run(task_number)
            
            task_running_list.append([
                loss_list,
                validatioin_list
            ])
            
            print(f"task {task_number} run finish")

        return task_running_list