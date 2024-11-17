from typing import List
import random

import torch

from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset

from dataloader.SplitedDataSet import SplitedDataSet
from dataloader.SplitedDataSetLoader import SplitedDataSetLoader

class CI():
    def __init__(self,
                 batch_size,
                 portion,
                 label_schedule_list: List,
                 dataset: Dataset):
        self.batch_size = batch_size
        self.label_schedule_list = label_schedule_list
        self.dataset = dataset
        self.portion = portion

        self.dataset_by_label = self.split_by_class()

        self.now_task = 0
        self.last_task = len(label_schedule_list)-1

    def split_by_class(self):
        """
        주어진 클래스(target_class)에 해당하는 데이터만 포함된 하위 데이터셋 반환
        """

        label_list = torch.tensor([label for _, label in self.dataset]).unique()
        splited_dataset = {}
        for target_label in label_list:
            indices = [i for i, (_, label) in enumerate(self.dataset) if label == target_label]
            splited_dataset[target_label] = Subset(self.dataset, indices)
        
        return splited_dataset
    
    def __iter__(self):
        return self


    def __next__(self):
        if self.now_task <= self.last_task:

            dataset = ConcatDataset([
                self.dataset_by_label[label]
                for label in self.label_schedule_list[self.now_task]
            ])

            self.now_task += 1

            return SplitedDataSetLoader(
                batch_size=self.batch_size,
                portion=self.portion,
                dataset=dataset)
        else:
            raise StopIteration
