from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataSetLoader():
    def __init__(self,
                 batch_size,
                 train_data: Dataset,
                 validatioin_data: Dataset,
                 test_data: Dataset):
        
        self.train = DataLoader(train_data, batch_size=batch_size)
        self.validatioin = DataLoader(validatioin_data, batch_size=batch_size)
        self.test = DataLoader(test_data, batch_size=batch_size)

        