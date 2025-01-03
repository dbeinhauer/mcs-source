"""
This source code defines multiple variants of complex neurons
that can be used in the model. Neuron here is represented by 
some small model.
"""

from typing import Optional, Tuple
from abc import ABC

import torch
import torch.nn as nn

import nn_model.globals
from nn_model.type_variants import ModelTypes


class CustomActivation(torch.nn.Module):
    def forward(self, x):
        # Apply sigmoid to get values between 0 and 1
        sigmoid_output = torch.sigmoid(x)
        # Apply scaled tanh for values greater than 1
        tanh_output = 5 * torch.tanh(x / 5)  # Scale tanh to range [0, 5]
        # Combine the two outputs
        return torch.where(x <= 1, sigmoid_output, tanh_output)


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

        self.custom_activation = CustomActivation()

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
            layers.append(nn.LayerNorm(layer_size))
            layers.append(nn.Linear(layer_size, layer_size))
            # layers.append(nn.LayerNorm(layer_size))
            layers.append(nn.ReLU())  # Non-linear activation after each layer.
            # layers.append(nn.GELU())
            # layers.append(nn.Dropout(0.2))

        # Final output layer: output size is 1
        layers.append(nn.Linear(layer_size, 1))

        # Use nn.Sequential to combine all layers into a single network
        return nn.Sequential(*layers)

    def forward(
        self, hidden: torch.Tensor, complexity_hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward step of the DNN model of neuron.

        :param hidden: Input of the model.
        :return: Returns tuple of output of the DNN mode and empty tensor of hidden states
        (to have same output as RNN variants).
        """
        # Pass the input through the network.
        out = self.network(hidden)
        # Apply potential residual connection.
        # out = self.forward_residual(hidden, out)
        if self.residual:
            out += hidden.sum(dim=1, keepdim=True)

        # Apply module final non-linearity function.
        # return self.forward_non_linearity(out)

        out = self.custom_activation(out)

        return out, torch.zeros(0)


# class RNNNeuron(SharedNeuronBase):
#     """
#     Class defining shared complexity of the layer that should represent
#     one neuron (more complex neuron than just one operation). This
#     type of neuron consists of an RNN.
#     """

#     def __init__(
#         self,
#         model_type: str,
#         hidden_size: int = 10,
#         num_layers: int = 1,
#         residual: bool = True,
#     ):
#         """
#         Initializes RNN model of the neuron.

#         :param model_type: Variant of the complex neuron (value from `ModelTypes`).
#         :param hidden_size: Size of the hidden state in the RNN.
#         :param num_layers: Number of RNN layers.
#         :param residual: Flag whether there is a residual connection used in the model.
#         """
#         super(RNNNeuron, self).__init__(model_type, residual)
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # RNN for processing
#         self.rnn = nn.RNN(
#             input_size=self.input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#         )

#         # Scalar output layer
#         self.output_layer = nn.Linear(hidden_size, 1)

#     def forward(
#         self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Performs forward step of the RNN model of neuron.

#         :param inputs: Input tensor of shape (batch_size, seq_len, input_size).
#         :param hidden: Optional hidden state for the RNN.
#         :return: Returns output of the RNN model.
#         """
#         # Add a sequence dimension to the input
#         inputs = inputs.unsqueeze(1)  # Shape: (batch_size, 1, input_size)

#         # RNN processing
#         rnn_output, hidden = self.rnn(
#             inputs, hidden
#         )  # Shape: (batch_size, seq_len, hidden_size)

#         # Extract last output and map to scalar
#         rnn_output = rnn_output[:, -1, :]  # Shape: (batch_size, hidden_size)
#         output = self.output_layer(rnn_output)  # Shape: (batch_size, 1)

#         # Apply residual connection if enabled
#         if self.residual:
#             output = output + inputs[:, -1, :].sum(
#                 dim=1, keepdim=True
#             )  # Use original input as residual

#         # Apply module final non-linearity function
#         output = self.custom_activation(output)

#         return output, hidden


class LSTMNeuron(SharedNeuronBase):
    """
    Memory-enabled neuron module using an LSTM cell for processing scalar inputs.
    """

    def __init__(
        self,
        model_type: str,
        # input_size: int,
        num_layers: int = 1,
        layer_size: int = 10,
        residual: bool = True,
    ):
        """
        Initialize the neuron module.

        :param model_type: Variant of the complex neuron (value from `ModelTypes`).
        :param input_size: Size of the input feature (scalar).
        :param layer_size: Size of the hidden state in the LSTM.
        :param num_layers: Number of LSTM layers for richer memory.
        :param residual: Whether to use a residual connection.
        """
        super(LSTMNeuron, self).__init__(model_type, residual)
        self.layer_size = layer_size
        self.num_layers = num_layers

        # LSTM cells for memory processing
        self.lstm_cells = nn.ModuleList(
            [
                nn.LSTMCell(self.input_size if i == 0 else layer_size, layer_size)
                for i in range(num_layers)
            ]
        )

        # Scalar output layer
        self.output_layer = nn.Linear(layer_size, 1)

        # Custom activation function
        self.custom_activation = CustomActivation()

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the memory neuron.

        :param inputs: Tensor of shape (batch_size, input_size).
        :param hidden: Optional hidden state (h_0, c_0) for the LSTM cells.
        :return: Processed output scalar and updated hidden state.
        """
        batch_size = inputs.size(0)

        if hidden is None:
            hidden = [
                (
                    torch.zeros(batch_size, self.layer_size).to(inputs.device),
                    torch.zeros(batch_size, self.layer_size).to(inputs.device),
                )
                for _ in range(self.num_layers)
            ]

        current_input = inputs
        h, c = hidden[0]
        for i, lstm_cell in enumerate(self.lstm_cells):
            h, c = lstm_cell(current_input, (h, c))
            current_input = (
                h  # The output of the current cell is the input to the next cell
            )

        # Apply the output layer to the last hidden state
        output = self.output_layer(h)

        # Apply residual connection if enabled
        # if self.residual:
        #     output = output + inputs  # Use original input as residual

        if self.residual:
            output += inputs.sum(dim=1, keepdim=True)

        # Apply module final non-linearity function
        output = self.custom_activation(output)

        return output, (h, c)
