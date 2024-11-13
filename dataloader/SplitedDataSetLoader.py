from SplitedDataSet import SplitedDataSet
from torch.utils.data import DataLoader

class SplitedDataSetLoader():
    def __init__(self,
                 batch_size,
                 splited_dataset: SplitedDataSet,
                 shuffle=True):
        
        self.train = DataLoader(splited_dataset.train, batch_size=batch_size, shuffle=shuffle)
        self.validatioin = DataLoader(splited_dataset.validatioin, batch_size=batch_size, shuffle=shuffle)
        self.test = DataLoader(splited_dataset.test, batch_size=batch_size, shuffle=shuffle)