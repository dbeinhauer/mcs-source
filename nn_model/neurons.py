"""
This source code defines multiple variants of complex neurons
that can be used in the model. Neuron here is represented by 
some small model.
"""

import torch
import torch.nn as nn

from nn_model.type_variants import ModelTypes


# Shared complexity module
class FeedForwardNeuron(nn.Module):
    """
    Class defining shared complexity of the layer that should represent
    one neuron (more complex neuron than just one operation).
    """

    def __init__(
        self,
        model_type: str,
        num_layers: int = 5,
        layer_size: int = 10,
        residual: bool = True,
    ):
        """
        Initializes DNN model of the neuron.

        :param model_type: Variant of the complex neuron (value from `ModelTypes`).
        :param num_layers: Number of layers of the model.
        :param layer_size: Size of the layer of the model.
        :param residual: Flag whether there is a residual connection used in the model.
        """
        super(FeedForwardNeuron, self).__init__()

        # Check that the model type is the one we would expect.
        assert model_type in [
            ModelTypes.COMPLEX_JOINT.value,
            ModelTypes.COMPLEX_SEPARATE.value,
        ]
        self.model_type: str = model_type

        self.input_size = (
            1 if model_type == ModelTypes.COMPLEX_JOINT.value else 2
        )  # Either joint (one input) or separate exc/inh layers (two inputs).

        self.residual: bool = residual

        self.network = self._init_model_architecture(model_type, layer_size, num_layers)

    def _init_model_architecture(
        self, model_type: str, layer_size: int, num_layers: int
    ):
        """
        Initializes architecture of the DNN neuron model.

        :param model_type: Variant of the complex neuron (value from `ModelTypes`).
        :param layer_size: Size of one layer.
        :param num_layers: Total number of layers.
        :return: Returns sequential model of the neuron model.
        """
        layers = nn.ModuleList()

        # layers.append(nn.Linear(1, layer_size))
        layers.append(nn.Linear(self.input_size, layer_size))
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
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
            # out += hidden
            out += hidden.sum(
                dim=1, keepdim=True
            )  # Sum over second dimension (sum excitatory and inhibitory outputs)

        # out = torch.nn.functional.relu(out)
        out = torch.nn.functional.hardtanh(out, min_val=0.0, max_val=20.0)

        return out
