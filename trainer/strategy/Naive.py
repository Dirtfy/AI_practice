from dataloader.SplitedDataLoader import SplitedDataLoader

from .base.Strategy import Strategy

class Naive(Strategy):
    def __init__(self,
                 batch_size,
                 portion) -> None:
        super().__init__(
            batch_size=batch_size,
            portion=portion
        )

    def make_loader(self, num_task, dataset) -> SplitedDataLoader:
        return SplitedDataLoader(
            batch_size=self.batch_size,
            portion=self.portion,
            dataset=dataset)