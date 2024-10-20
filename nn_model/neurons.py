"""
This source code defines multiple variants of complex neurons
that can be used in the model. Neuron here is represented by 
some small model.
"""

import torch.nn as nn


# Shared complexity module
class FeedForwardNeuron(nn.Module):
    """
    TODO:
    Class defining shared complexity of the layer that should represent
    one neuron (more complex neuron than just one operation).
    """

    def __init__(
        self,
        num_layers: int = 5,
        layer_size: int = 10,
        residual: bool = True,
    ):

        super(FeedForwardNeuron, self).__init__()

        # self.num_layers = num_layers
        # self.layer_size = layer_size
        self.residual = residual

        # Create a list to hold layers
        layers = []

        # First layer: input size is 1 (based on your earlier example)
        layers.append(nn.Linear(1, layer_size))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))

        # Final output layer: output size is 1
        layers.append(nn.Linear(layer_size, 1))

        # Use nn.Sequential to combine all layers into a single network
        self.network = nn.Sequential(*layers)

    def forward(self, hidden):
        out = self.network(hidden)
        if self.residual:
            out += hidden
        # Pass the input through the network
        return out
