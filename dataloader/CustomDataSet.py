from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx):
        if self.data is not None:
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label
        else:
            return None, None