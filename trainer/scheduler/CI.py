from typing import List
import random

import torch

from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset

from .base.Scheduler import Scheduler

class CI(Scheduler):
    def __init__(self,
                 whole_dataset: Dataset = None,
                 dataset_by_label: dict = None,
                 label_schedule_list: List = None,
                 count_schedule_list: List = None):
        super().__init__()

        assert not (label_schedule_list is None and count_schedule_list is None)
        assert not (label_schedule_list is not None and count_schedule_list is not None)

        self.dataset = whole_dataset

        if dataset_by_label is not None:
            self.dataset_by_label = dataset_by_label
        else:
            print("extracting labels from dataset")
            self.label_list = torch.tensor([label for _, label in dataset]).unique().tolist()
            self.dataset_by_label = self.split_by_class()

        self.count_schedule_list = count_schedule_list
        self.label_schedule_list = label_schedule_list \
            if label_schedule_list is not None else self.make_schedule()

        self.now_task = 0
        self.last_task = len(label_schedule_list)-1

    def split_by_class(self):
        """
        주어진 클래스(target_class)에 해당하는 데이터만 포함된 하위 데이터셋 반환
        """

        splited_dataset = {}
        print("splitting dataset by label")
        for target_label in self.label_list:
            print(f"by {target_label}")
            indices = [i for i, (_, label) in enumerate(self.dataset) if label == target_label]
            splited_dataset[target_label] = Subset(self.dataset, indices)
        
        return splited_dataset
    
    def sample(self, list, count):
        selected = random.sample(list, count)
        for i in selected:
            list.remove(i)
        return selected, list

    def make_schedule(self):
        label_list = self.label_list[::]

        label_schedule_list = []
        for count in self.count_schedule_list:
            selected, _ = self.sample(label_list, count)
            label_schedule_list.append(selected)

        return label_schedule_list


    def __iter__(self):
        return self


    def __next__(self) -> Dataset:
        if self.now_task <= self.last_task:

            dataset = ConcatDataset([
                self.dataset_by_label[label]
                for label in self.label_schedule_list[self.now_task]
            ])

            self.now_task += 1

            return dataset
        else:
            raise StopIteration
