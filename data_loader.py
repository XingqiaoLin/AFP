import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AntifreezeProteinDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        figure_str = self.data_frame.iloc[idx, 0]
        figure = np.array(figure_str.split(), dtype=np.float32).reshape(64, 64)

        embedding_str = self.data_frame.iloc[idx, 3].strip('[]')
        embedding = np.fromstring(embedding_str, sep=',', dtype=np.float32)

        label = self.data_frame.iloc[idx, 1]

        return (torch.tensor(figure, dtype=torch.float32),
                torch.tensor(embedding, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))

def get_data_loader(csv_path, batch_size, shuffle=True):
    dataset = AntifreezeProteinDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)