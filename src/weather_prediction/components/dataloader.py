from torch.utils.data import Dataset
import pandas as pd
import torch

class WeatherDataLoader(Dataset):
    def __init__(self, data:pd.DataFrame, features:list, target:list):
        self.X = torch.tensor(data[features].values,
                              dtype=torch.float32)
        self.y = torch.tensor(data[target].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
