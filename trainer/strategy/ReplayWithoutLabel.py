import random

from torch.utils.data import ConcatDataset


from dataloader.SplitedDataLoader import SplitedDataLoader
from .base.Strategy import Strategy
from model.Category import Generator, Classifier
from dataloader.CustomDataSet import CustomDataSet

class ReplayWithoutLabel(Strategy):
    def __init__(self,
                 batch_size,
                 portion,
                 generator: Generator,
                 classifier: Classifier,
                 label_schedule_list,
                 replay_percentage) -> None:
        super().__init__(
            batch_size=batch_size,
            portion=portion
        )

        self.generator = generator
        self.classifier = classifier

        self.label_schedule_list = label_schedule_list
        self.replay_percentage = replay_percentage


    def make_loader(self, num_task, dataset) -> SplitedDataLoader:
        if num_task == 0:
            return SplitedDataLoader(
                batch_size=self.batch_size,
                portion=self.portion,
                dataset=dataset)
    
        num_sample = int(len(dataset)*self.replay_percentage)

        generated_data = self.generator.generate(num_sample)

        data = []
        labels = []
        for i in range(num_sample):
            generated = generated_data[i]
            pred = self.classifier.inference(generated)

            data.append(generated)
            labels.append(pred)

        generated_dataset = CustomDataSet(data=data, labels=labels)
        
        return SplitedDataLoader(
            batch_size=self.batch_size,
            portion=self.portion,
            dataset=generated_dataset)