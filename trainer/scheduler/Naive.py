from torch.utils.data import Dataset
from .base.Scheduler import Scheduler

class Naive(Scheduler):
    def __init__(self,
                 whole_dataset: Dataset) -> None:
        super().__init__()

        self.whole_dataset = whole_dataset

        self.now_task = 0
        self.last_task = 0

    def __next__(self) -> Dataset:
        if self.now_task <= self.last_task:
            self.now_task += 1
            return self.whole_dataset
        else:
            raise StopIteration
