import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class LandmarkDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.landmarks = torch.tensor(data['landmarks'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.labels[idx]

def get_landmark_dataloader(npz_path: str, batch_size: int = 64, shuffle: bool = True):
    dataset = LandmarkDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
