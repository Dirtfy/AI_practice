from abc import *

class Method(metaclass=ABCMeta):
    @abstractmethod
    def train_batch(self, architecture, x, y):
        raise NotImplementedError