from torch.utils.data import Subset

class SplitedDataSet():
    def __init__(self,
                 train_data: Subset,
                 validation_data: Subset,
                 test_data: Subset):
        
        self.train = train_data
        self.validatioin = validation_data
        self.test = test_data