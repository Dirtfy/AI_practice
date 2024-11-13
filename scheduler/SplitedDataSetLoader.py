from SplitedDataSet import SplitedDataSet
from torch.utils.data import DataLoader

class SplitedDataSetLoader():
    def __init__(self,
                 batch_size,
                 splited_dataset: SplitedDataSet):
        
        self.train = DataLoader(splited_dataset.train, batch_size=batch_size)
        self.validatioin = DataLoader(splited_dataset.validatioin, batch_size=batch_size)
        self.test = DataLoader(splited_dataset.test, batch_size=batch_size)