"""
This source code defines multiple variants of complex neurons
that can be used in the model. Neuron here is represented by 
some small model.
"""

import torch
import torch.nn as nn


class CustomReLU(nn.Module):
    """
    Custom ReLU class as the classical one has problems with computational graph.
    """

    def __init__(self, min_val=0.0):
        """
        Initializes ReLU module.

        :param min_val: ReLU clamp value (minimum).
        """
        super().__init__()
        self.min_val = min_val

    def forward(self, x):
        """
        Does simple ReLU computation. All negative values are equal to 0.

        :param x: Input value.
        :return: Input value after ReLU application.
        """
        return torch.clamp(x, min=self.min_val)


# Shared complexity module
class FeedForwardNeuron(nn.Module):
    """
    Class defining shared complexity of the layer that should represent
    one neuron (more complex neuron than just one operation).
    """

    def __init__(
        self,
        num_layers: int = 5,
        layer_size: int = 10,
        residual: bool = True,
    ):
        """
        Initializes DNN model of the neuron.

        :param num_layers: Number of layers of the model.
        :param layer_size: Size of the layer of the model.
        :param residual: Flag whether there is a residual connection used in the model.
        """
        super(FeedForwardNeuron, self).__init__()

        self.residual: bool = residual

        self.network = self._init_model_architecture(layer_size, num_layers)

    def _init_model_architecture(self, layer_size: int, num_layers: int):
        """
        Initializes architecture of the DNN neuron model.

        :param layer_size: Size of one layer.
        :param num_layers: Total number of layers.
        :return: Returns sequential model of the neuron model.
        """
        layers = nn.ModuleList()

        # Input layer
        layers.append(nn.Linear(1, layer_size))
        # layers.append(nn.Linear(2, layer_size))
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            # TODO:
            # layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.LayerNorm(layer_size))
            layers.append(nn.ReLU())  # Non-linear activation after each layer.

        # Final output layer: output size is 1
        layers.append(nn.Linear(layer_size, 1))

        # Use nn.Sequential to combine all layers into a single network
        return nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Performs forward step of the DNN model of neuron.

        :param hidden: Input of the model.
        :return: Returns output of the DNN model.
        """
        # Pass the input through the network.
        out = self.network(hidden)
        if self.residual:
            # We want to use residual connection.
            out += hidden
            # out += hidden.sum(
            #     dim=1, keepdim=True
            # )  # Sum over second dimension (sum excitatory and inhibitory outputs)

        # out = torch.nn.functional.relu(out)
        out = torch.nn.functional.hardtanh(out, min_val=0.0, max_val=20.0)

        return out
