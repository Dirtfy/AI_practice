import random

from torch.utils.data import ConcatDataset


from dataloader.SplitedDataLoader import SplitedDataLoader
from .base.Strategy import Strategy
from model.Category import Generator
from dataloader.CustomDataSet import CustomDataSet

class Replay(Strategy):
    def __init__(self,
                 batch_size,
                 portion,
                 generator: Generator,
                 label_schedule_list,
                 replay_percentage) -> None:
        super().__init__(
            batch_size=batch_size,
            portion=portion)

        self.generator = generator
        self.label_schedule_list = label_schedule_list
        self.replay_percentage = replay_percentage


    def make_loader(self, num_task, dataset) -> SplitedDataLoader:
        num_sample = int(len(dataset)*self.replay_percentage)

        label_pool = []
        for label_schedule in self.label_schedule_list[:num_task]:
            label_pool.extend(label_schedule)

        num_sample = int(num_sample/len(label_pool))

        conacted_dataset = CustomDataSet(data=None, labels=None)
        for label in label_pool:
            generated_batch = self.generator.generate(num_sample, label)
            data = []
            for i in range(num_sample):
                data.append(generated_batch[i])
            labels = [label]*num_sample

            conacted_dataset = ConcatDataset([
                conacted_dataset,
                CustomDataSet(data=data,labels=labels)
            ])

        return SplitedDataLoader(
            batch_size=self.batch_size,
            portion=self.portion,
            dataset=conacted_dataset)