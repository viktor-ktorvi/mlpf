import torch
from torch import nn


class StandardScaler(nn.Module):
    """
    Standard scaler transform as a torch module.
    """

    def __init__(self, X):
        super(StandardScaler, self).__init__()

        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0)
        self.std[self.std < 1e-9] = 1.0

    def forward(self, x):
        return (x - self.mean) / self.std

    def _apply(self, fn):
        super(StandardScaler, self)._apply(fn)
        self.mean = fn(self.mean)
        self.std = fn(self.std)

        return self

    def inverse(self, z):
        return z * self.std + self.mean
