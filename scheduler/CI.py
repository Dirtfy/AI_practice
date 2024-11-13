from typing import List

from torch.utils.data import ConcatDataset

from ..dataloader.SplitedDataSet import SplitedDataSet
from ..dataloader.SplitedDataSetLoader import SplitedDataSetLoader

class CI():
    def __init__(self,
                 batch_size,
                 class_schedule_list: List[List[int]],
                 class_dataset_list: List[SplitedDataSet]):
        self.batch_size = batch_size
        self.class_schedule_list = class_schedule_list
        self.class_dataset_list = class_dataset_list

        self.now_task = 0
        self.last_task = len(class_schedule_list)-1

    
    def __iter__(self):
        return self


    def __next__(self):
        if self.now_task <= self.last_task:

            train_dataset = ConcatDataset([
                self.class_dataset_list[class_number].train
                for class_number in self.class_schedule_list[self.now_task]
                ])
            validation_dataset = ConcatDataset([
                self.class_dataset_list[class_number].validatioin
                for class_number in self.class_schedule_list[self.now_task]
                ])
            test_dataset = ConcatDataset([
                self.class_dataset_list[class_number].test
                for class_number in self.class_schedule_list[self.now_task]
                ])
            
            splited_dataset = SplitedDataSet(
                train_data=train_dataset,
                validatioin_data=validation_dataset,
                test_data=test_dataset
            )

            self.now_task += 1

            return SplitedDataSetLoader(
                batch_size=self.batch_size,
                splited_dataset=splited_dataset
                )
        else:
            raise StopIteration
