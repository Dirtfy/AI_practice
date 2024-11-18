import random

from torch.utils.data import Dataset, Subset, DataLoader

from .SplitedDataSet import SplitedDataSet


class SplitedDataLoader():
    def __init__(self,
                 batch_size,
                 portion,
                 dataset: Dataset,
                 shuffle=True):
        
        self.portion = portion
        
        splited_dataset = self.split_by_portion(dataset=dataset)
        
        self.train = DataLoader(splited_dataset.train, batch_size=batch_size, shuffle=shuffle)
        self.validatioin = DataLoader(splited_dataset.validatioin, batch_size=batch_size, shuffle=shuffle)
        self.test = DataLoader(splited_dataset.test, batch_size=batch_size, shuffle=shuffle)

    def sample(self, list, count):
        selected = random.sample(list, count)
        for i in selected:
            list.remove(i)
        return selected, list


    def split_by_portion(self, dataset):
        dataset_size = len(dataset)

        train = int(dataset_size*self.portion[0])
        val = int(dataset_size*self.portion[1])
        test = dataset_size-(train+val)

        indices = list(range(dataset_size))

        train_indices, indices = self.sample(indices, train)
        val_indicies, indices = self.sample(indices, val)
        test_indicies, indices = self.sample(indices, test)

        return SplitedDataSet(
            train_data=Subset(dataset, train_indices),
            validation_data=Subset(dataset, val_indicies),
            test_data=Subset(dataset, test_indicies))