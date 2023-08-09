import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split as sk_train_test_split
from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx, 0]
        features = self.data.iloc[idx, 3:].values.astype(np.float32)
        target = self.data.iloc[idx, 2]

        image = read_image(path)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'features': features, 'target': target}