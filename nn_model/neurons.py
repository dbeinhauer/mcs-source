"""
This source code defines multiple variants of complex neurons
that can be used in the model. Neuron here is represented by 
some small model.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import nn_model.globals
from nn_model.type_variants import ModelTypes


class SharedNeuronBase(nn.Module, ABC):
    """
    Base class of the shared neuron module.
    """

    joint_input_models = [ModelTypes.DNN_JOINT.value, ModelTypes.RNN_JOINT.value]
    separate_input_models = [
        ModelTypes.DNN_SEPARATE.value,
        ModelTypes.RNN_SEPARATE.value,
    ]

    def __init__(self, model_type: str, residual: bool = True):
        super(SharedNeuronBase, self).__init__()
        # Check that the model type is valid
        assert model_type in nn_model.globals.COMPLEX_MODELS, "Invalid model type"
        self.model_type = model_type

        # Set module input size.
        self.input_size = self._select_input_size()

        # Flag whether we want to use the residual connection.
        self.residual = residual

    def _select_input_size(self) -> int:
        """
        Based on the model type set the model input size.

        :return: Returns size of the input layer (typically 1 for joint
        inputs and 2 for separate excitatory/inhibitory layers).
        """
        if self.model_type in SharedNeuronBase.joint_input_models:
            # Joint model -> input is sum of excitatory and inhibitory layers.
            return 1
        if self.model_type in SharedNeuronBase.separate_input_models:
            # Separate model -> input are two values one for excitatory one for inhibitory layer.
            return 2

        class WrongModelException(Exception):
            """
            Exception in case wrong model type was selected.
            """

        raise WrongModelException(
            "Wrong model type was selected for shared complexity."
        )

    # def forward_residual(self, hidden: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    #     """
    #     Applies residual connection if enabled.

    #     :param hidden: Original input tensor.
    #     :param out: Output from the model.
    #     :return: Residual output tensor.
    #     """
    #     if self.residual:
    #         out += hidden.sum(dim=1, keepdim=True)

    #     return out

    def forward_non_linearity(self, input_data: torch.Tensor):
        """
        Applies final non-linearity (bounder ReLU) function in the forward step.

        :param input_data: Tensor to apply the non linearity on.
        :return: Returns the input tensor after the non-linearity is applied.
        """
        return torch.nn.functional.hardtanh(input_data, min_val=0.0, max_val=20.0)

    # @abstractmethod
    # def forward(self) -> torch.Tensor:
    #     """
    #     Abstract forward method of the module.
    #     """


# Shared complexity module
# class FeedForwardNeuron(nn.Module):
class FeedForwardNeuron(SharedNeuronBase):
    """
    Class defining shared complexity of the layer that should represent
    one neuron (more complex neuron than just one operation). This
    type of neuron consist of feed-forward DNN.
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
        super(FeedForwardNeuron, self).__init__(model_type, residual)

        # Check that the model type is the one we would expect.
        # assert model_type in nn_model.globals.DNN_MODELS
        # self.model_type: str = model_type

        # self.input_size = (
        #     1 if model_type == ModelTypes.DNN_JOINT.value else 2
        # )  # Either joint (one input) or separate exc/inh layers (two inputs).

        # self.residual: bool = residual

        self.network = self._init_model_architecture(layer_size, num_layers)

    def _init_model_architecture(self, layer_size: int, num_layers: int):
        """
        Initializes architecture of the DNN neuron model.

        :param layer_size: Size of one layer.
        :param num_layers: Total number of layers.
        :return: Returns sequential model of the neuron model.
        """
        layers = nn.ModuleList()

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
        # Apply potential residual connection.
        # out = self.forward_residual(hidden, out)
        if self.residual:
            out += hidden.sum(dim=1, keepdim=True)

        # Apply module final non-linearity function.
        return self.forward_non_linearity(out)


class LSTMNeuron(SharedNeuronBase):
    """
    Memory-enabled neuron module using an LSTM cell for processing scalar inputs.
    """

    def __init__(
        self,
        model_type: str,
        hidden_size: int = 10,
        num_layers: int = 1,
        residual: bool = True,
    ):
        """
        Initialize the neuron module.

        :param input_size: Size of the input feature (scalar).
        :param hidden_size: Size of the hidden state in the LSTM.
        :param num_layers: Number of LSTM layers for richer memory.
        :param residual: Whether to use a residual connection.
        """
        super(LSTMNeuron, self).__init__(model_type, residual)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.residual = residual

        # LSTM for memory processing
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Scalar output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor, hidden: tuple) -> torch.Tensor:
        """
        Forward pass of the memory neuron.

        :param inputs: Tensor of shape (batch_size, 1), one scalar per input.
        :param hidden: Optional hidden state (h_0, c_0) for the LSTM.
        :return: Processed output scalar and updated hidden state.
        """
        # Reshape input to mimic a sequence of length 1
        inputs = inputs.unsqueeze(1)  # Shape: (batch_size, seq_len=1, input_size)

        # LSTM processing
        lstm_output, hidden = self.lstm(
            inputs, hidden
        )  # Shape: (batch_size, 1, hidden_size)

        # Extract last output and map to scalar
        lstm_output = lstm_output[:, -1, :]  # Shape: (batch_size, hidden_size)
        output = self.output_layer(lstm_output)  # Shape: (batch_size, 1)

        # Apply residual connection if enabled
        if self.residual:
            output = output + inputs.squeeze(1)  # Use original input as residual

        return output, hidden
