import torch.nn as nn


class CNN(nn.Module):
    """
    Simple CNN for MNIST digit classification.
    Accepts flat 784-dim input (rescaled pixels) and reshapes
    internally to 1×28×28 before the convolutional layers.
    """

    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # → 32×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → 32×14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → 64×14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → 64×7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # reshape flat 784 → 1×28×28
        return self.classifier(self.features(x))
