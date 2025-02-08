"""
This source code contains definition of all final activation functions used in neuron models.
"""

import torch
import torch.nn as nn


class SigmoidTanh(torch.nn.Module):
    """
    Custom neuron activation function designed the way that should better capture the
    expected output values of the neurons.

    We expect that neurons return values in interval (0, 5) because when we inspect our
    training data there is no such time interval that contains more than 4 spikes in the
    20 ms time window (largest window that we use). We then safely assume that our values
    should be smaller than 5 (in case it is not true it is permissible exception).

    Alongside with this, the majority of data lays in the interval (0, 1). So, we use
    sigmoid function for all predictions smaller than 1 (for those we would get value
    from 0 to 1 always). When this threshold is reached we apply scaled tanh to spread the
    rest of the input values to interval (0, 5).
    """

    def forward(self, x):
        # TODO: improve the function definition (especially better tanh case)
        # Apply sigmoid to get values between 0 and 1
        sigmoid_output = torch.sigmoid(x)
        # Apply scaled tanh for values greater than 1
        tanh_output = 5 * torch.tanh(x / 5)  # Scale tanh to range [0, 5]
        # Combine the two outputs
        return torch.where(x <= 1, sigmoid_output, tanh_output)


class LeakyTanh(nn.Module):
    def __init__(
        self,
        scale: float = 2.5,
        leak: float = 0.001,
        steepness: float = 0.2,
        beta: float = 20,
    ):
        super().__init__()
        self.scale = scale
        self.leak = leak
        self.steepness = steepness
        self.beta = beta

    def forward(self, x):
        return (
            self.scale * torch.tanh(self.steepness * x)  # tanh
            + self.scale  # move up so that lim -inf is 0
            + self.leak * torch.nn.functional.softplus(x, beta=self.beta)
        )  # introduce leak for x -> inf
