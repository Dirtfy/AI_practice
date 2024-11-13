from torch.utils.data import Dataset

class SplitedDataSet():
    def __init__(self,
                 train_data: Dataset,
                 validatioin_data: Dataset,
                 test_data: Dataset):
        
        self.train = train_data
        self.validatioin = validatioin_data
        self.test = test_data