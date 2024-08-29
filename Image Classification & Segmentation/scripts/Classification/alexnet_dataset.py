import pandas as pd
from torch.utils.data import Dataset
import torch

class CatDogDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, :-2].values.astype('float32').reshape(120, 120)
        label = self.data.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# example usage：
# from torch.utils.data import DataLoader
# dataset = CatDogDataset(csv_file='../../data/cat_dog/train.csv')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
