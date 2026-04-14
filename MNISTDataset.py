import torch
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (numpy.ndarray): The pixel data (X).
            labels (numpy.ndarray): The target digits (y).
        """
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

        if torch.max(self.X) > 1.0:
            self.X = self.X / 255.0

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Fetches the feature and label at the specified index.
        This is called automatically by the DataLoader during the training loop.
        """
        image = self.X[idx]
        label = self.y[idx]
        
        return image, label