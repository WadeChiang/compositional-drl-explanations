import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LunarLanderDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx],dtype=torch.float32)


def testData():
    state = np.load('rl_compexp/save/state.npy')
    count = np.sum((state[:, -2:] == 0).all(axis=1))
    print(f"Number of rows where the last two elements are 0: {count}")
    print(state.shape)
    lunar_loader = DataLoader(LunarLanderDataset(
        state), batch_size=32, shuffle=True)
    sample = next(iter(lunar_loader))
    print(sample)
