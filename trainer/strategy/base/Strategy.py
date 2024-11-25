from abc import *

from dataloader.SplitedDataLoader import SplitedDataLoader

class Strategy(metaclass=ABCMeta):
    def __init__(self,
                 batch_size,
                 portion) -> None:
        self.batch_size = batch_size
        self.portion = portion

    @abstractmethod
    def make_loader(self, num_task, dataset) -> SplitedDataLoader:
        raise NotImplementedError