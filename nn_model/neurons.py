"""
This source code defines multiple variants of complex neurons
that can be used in the model. Neuron here is represented by 
some small model.
"""

import torch
import torch.nn as nn


class CustomActivation(torch.nn.Module):
    def forward(self, x):
        # Apply sigmoid to get values between 0 and 1
        sigmoid_output = torch.sigmoid(x)
        # Apply scaled tanh for values greater than 1
        tanh_output = 5 * torch.tanh(x / 5)  # Scale tanh to range [0, 5]
        # Combine the two outputs
        return torch.where(x <= 1, sigmoid_output, tanh_output)


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

        self.custom_activation = CustomActivation()

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
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
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

        # out = torch.nn.functional.hardtanh(out, min_val=0.0, max_val=5.0)

        # out = CustomActivation()
        out = self.custom_activation(out)

        return out
