from torch.utils.data import Dataset

class Scheduler():
    def __init__(self) -> None:
        pass
    
    def __iter__(self):
        return self


    def __next__(self) -> Dataset:
        raise NotImplementedError
    