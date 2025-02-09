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
from nn_model.type_variants import NeuronActivationTypes
from nn_model.activation_functions import SigmoidTanh, LeakyTanh


class SharedNeuronBase(nn.Module, ABC):
    """
    Base class of the shared neuron module.
    """

    def __init__(
        self,
        model_type: str,
        activation_function: str,
        residual: bool = True,
    ):
        """
        Initializes neuron model base parameters (shared across all neuron models).

        :param model_type: Type of the neuron model.
        :param activation_function: Final activation function of the neuron model.
        :param residual: Flag whether we want to use residual connection in the neuron model.
        """
        super(SharedNeuronBase, self).__init__()
        # Check that the model type is valid
        assert model_type in nn_model.globals.COMPLEX_MODELS, "Invalid model type"
        self.model_type = model_type

        # Set module input size.
        self.input_size = self._select_input_size()

        # Flag whether we want to use the residual connection.
        self.residual = residual

        self.custom_activation = (
            SigmoidTanh()
            if activation_function == NeuronActivationTypes.SIGMOIDTANH.value
            else LeakyTanh()
        )

    def _select_input_size(self) -> int:
        """
        Based on the model type set the model input size.

        :return: Returns size of the input layer (typically 1 for joint
        inputs and 2 for separate excitatory/inhibitory layers).
        """
        if self.model_type in nn_model.globals.JOINT_MODELS:
            # Joint model -> input is sum of excitatory and inhibitory layers.
            return 1
        elif self.model_type in nn_model.globals.SEPARATE_MODELS:
            # Separate model -> input values are two values one for excitatory
            # one for inhibitory layer.
            return 2

        class WrongModelException(Exception):
            """
            Exception in case wrong model type was selected.
            """

        raise WrongModelException(
            "Wrong model type was selected for shared complexity."
        )


class DNNNeuron(SharedNeuronBase):
    """
    Class defining shared complexity of the layer that should represent
    one neuron (more complex neuron than just one operation). This
    type of neuron consist of feed-forward DNN.
    """

    def __init__(
        self,
        model_type: str,
        activation_function: str,
        num_layers: int = 5,
        layer_size: int = 10,
        residual: bool = True,
    ):
        """
        Initializes DNN model of the neuron.

        :param model_type: Variant of the complex neuron (value from `ModelTypes`).
        :param activation_function: Final activation function of the neuron model.
        :param num_layers: Number of layers of the model.
        :param layer_size: Size of the layer of the model.
        :param residual: Flag whether there is a residual connection used in the model.
        """
        super(DNNNeuron, self).__init__(model_type, activation_function, residual)

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
            layers.append(nn.ReLU())  # Non-linear activation after each layer.
            # layers.append(nn.Dropout(0.2)) # TODO: decide whether we want to include dropout

        # Final output layer: output size is 1
        layers.append(nn.Linear(layer_size, 1))

        # Use nn.Sequential to combine all layers into a single network
        return nn.Sequential(*layers)

    def forward(
        self, hidden: torch.Tensor, complexity_hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Performs forward step of the DNN model of neuron.

        :param hidden: Input of the model.
        :param complexity_hidden: Not used argument of complexity hidden states
        (used to have same forward header as RNN variants).
        :return: Returns tuple of output of the DNN mode and empty
        tensor of hidden states (to have same output as RNN variants).
        """
        # Pass the input through the network.
        out = self.network(hidden)

        if self.residual:
            # Apply residual connection.
            out += hidden.sum(dim=1, keepdim=True)

        # Apply module final non-linearity function.
        out = self.custom_activation(out)

        # Return
        return out, torch.zeros(0)


class LSTMNeuron(SharedNeuronBase):
    """
    Memory-enabled neuron module using an LSTM cell for processing scalar inputs.
    """

    # TODO: Look deeply at the LSTM module definition (num_layers)

    def __init__(
        self,
        model_type: str,
        activation_function: str,
        num_layers: int = 1,
        layer_size: int = 10,
        residual: bool = True,
    ):
        """
        Initialize the neuron module.

        :param model_type: Variant of the complex neuron (value from `ModelTypes`).
        :param activation_function: Final activation function of the neuron model.
        :param num_layers: Number of LSTM layers for richer memory (neuron hidden time steps).
        :param layer_size: Size of the hidden state in the LSTM.
        :param residual: Whether to use a residual connection.
        """
        super(LSTMNeuron, self).__init__(model_type, activation_function, residual)
        self.layer_size = layer_size
        self.num_layers = num_layers  # Number of hidden time steps

        self.input_layer = nn.Linear(self.input_size, layer_size)
        self.lstm_cell = nn.LSTMCell(layer_size, layer_size)
        # LSTM cells for memory processing
        # self.lstm_cells = nn.ModuleList(
        #     # [
        #     #     nn.LSTMCell(self.input_size if i == 0 else layer_size, layer_size)
        #     #     for i in range(num_layers)
        #     # ]
        #     [
        #         # nn.LSTMCell(self.input_size, layer_size),
        #         nn.LSTMCell(layer_size, layer_size),
        #     ]
        # )

        # Scalar output layer
        self.output_layer = nn.Linear(layer_size, 1)

        # Custom activation function
        # self.custom_activation = SigmoidTanh()

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:

        # TODO: WHole definition is strange somehow (we have layers but we work with only the one of them as time step layer).
        """
        Forward pass of the memory neuron.

        :param inputs: Tensor of shape (batch_size, input_size).
        :param hidden: Optional hidden state (h_0, c_0) for the LSTM cells.
        :return: Processed output scalar and updated hidden state.
        """
        batch_size = inputs.size(0)

        if hidden is None:
            hidden = (
                torch.zeros(batch_size, self.layer_size).to(inputs.device),
                torch.zeros(batch_size, self.layer_size).to(inputs.device),
            )
            #     for _ in range(self.num_layers)
            # ]

        current_input = self.input_layer(inputs)
        h, c = hidden
        for i in range(self.num_layers):
            # h, c = lstm_cell(current_input, (h, c))
            h, c = self.lstm_cell(current_input, (h, c))
            current_input = (
                h  # The output of the current cell is the input to the next cell
            )

        h, c = self.lstm_cell(current_input, (h, c))

        # Apply the output layer to the last hidden state
        output = self.output_layer(h)

        if self.residual:
            # Apply residual connection if enabled
            output += inputs.sum(dim=1, keepdim=True)

        # Apply module final non-linearity function
        output = self.custom_activation(output)

        return output, (h, c)
