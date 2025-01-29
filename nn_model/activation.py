import torch
import torch.nn as nn


class LeakyTanh(nn.Module):
    def __init__(self, scale: float = 2.5, leak: float = .001, steepness: float = .2, beta: float = 20):
        super().__init__()
        self.scale = scale
        self.leak = leak
        self.steepness = steepness
        self.beta = beta

    def forward(self, x):
        return (self.scale * torch.tanh(self.steepness * x)  # tanh
                + self.scale  # move up so that lim -inf is 0
                + self.leak * torch.nn.functional.softplus(x, beta=self.beta))  # introduce leak for x -> inf
