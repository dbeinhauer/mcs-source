import torch
import torch.nn as nn


class LeakyTanh(nn.Module):
    def __init(self, scale: float = 5., leak: float = .001):
        self.scale = scale
        self.leak = leak

    def forward(self, x):
        return self.scale * torch.tanh(x / self.scale) + self.leak * x
