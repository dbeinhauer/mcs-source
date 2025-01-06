"""
This source code defines variants of possible model layers
typically in some form of `RNNCell` module with additional
operations and complexities.
"""

from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn

import nn_model.globals
from nn_model.type_variants import ModelTypes
from nn_model.custom_rnn import CustomRNNCell
from nn_model.neurons import SharedNeuronBase
from nn_model.models import PrimaryVisualCortexModel
from nn_model.weights_constraints import WeightConstraint


class ModelLayer(nn.Module):
    """
    Class defining one neuronal layer of the model constrained with the model
    assumptions (inhibitory/excitatory layers).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layer_name: str,
        weight_constraint: WeightConstraint,
        input_constraints: List[Dict],
        weight_initialization_type: str,
        neuron_model: Optional[SharedNeuronBase] = None,
    ):
        """
        Initializes layer parameters and constraints.

        :param input_size: input size of the layer.
        :param hidden_size: hidden (output) size of the layer.
        :param layer_name: Name of the layer (to determine whether it is excitatory or inhibitory).
        :param weight_constraint: constraint object of the layer.
        :param input_constraints: List of properties of each input layer
        (its size and type (inhibitory/excitatory)).
        :param weight_initialization_type: Which type of weight initialization we want to use.
        :param neuron_model: Shared complexity neuron model used as a neuron non-linearity
        activation in the layer. In case we would like to use simple model the value
        should be `None`.
        """
        super(ModelLayer, self).__init__()

        # Layer size parameters:
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Check layer type is one of our expected output layers.
        assert layer_name in [PrimaryVisualCortexModel.layers_input_parameters.keys()]
        self.layer_name = layer_name

        # Assign the model type.
        self.model_type = (
            ModelTypes.SIMPLE.value if not neuron_model else neuron_model.model_type
        )

        # Initialize the layer model.
        self.rnn_cell = CustomRNNCell(
            input_size,
            hidden_size,
            layer_name,
            input_constraints,
            self.model_type,
            weight_initialization_type,
        )
        # Weights constraint object.
        self.constraint = weight_constraint

        # Shared neuron model.
        self.neuron_model = neuron_model

        # Flag whether we want to return also linear outputs (before non-linearity
        # function calling) of the layer (for model analysis).
        self.return_recurrent_state = False

    def switch_to_return_recurrent_state(self):
        """
        Changes state of the layer to return recurrent time steps.

        NOTE: This functionality is typically used for neuron model analysis.
        """
        self.return_recurrent_state = True

    def apply_constraints(self):
        """
        Applies the layer constraint on the weights.
        """
        self.constraint.apply(self.rnn_cell)

    def apply_complexity(
        self,
        rnn_output: Tuple[torch.Tensor, ...],
        neuron_hidden: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Applies complexity layer (shared DNN of neuron) on RNN linearity output.

        :param rnn_output: Linearity output of RNN layer to apply complexity on.
        :param neuron_hidden: Tuple of hidden states of the neurons (needed for RNN models).
        :return: Returns tuple of output after shared complexity is applied (first value)
        and tuple of all updated hidden states of the neuron model (if RNN neuron model is used).
        """

        neuron_model_input: torch.Tensor = rnn_output
        if len(rnn_output) == 2:
            # In case we work with separate excitatory and inhibitory inputs
            # -> stack their values to one tensor.
            neuron_model_input = torch.stack(rnn_output, dim=2)

        # If shared complexity is defined (model is complex) -> apply complexity.
        if self.neuron_model:
            # Reshape the layer output to [batch_size * hidden_size, neuron_model_input_size]
            # for batch processing (parallel application of the neuron module for all
            # the layer output values).
            complexity_result = neuron_model_input.view(
                -1, self.neuron_model.input_size
            )

            # Apply the neuron model to all values at parallel.
            complexity_result, neuron_hidden = self.neuron_model(
                complexity_result, neuron_hidden
            )

            # Define the output shape.
            viewing_shape: torch.Tensor = rnn_output
            if self.model_type in nn_model.globals.SEPARATE_MODELS:
                # In case we use separate models we want the output shape to be same as
                # one of the input tensors (we do not want two output values
                # (for exc/inh layer) anymore).
                viewing_shape = rnn_output[0]

            # Reshape back to [batch_size, hidden_size]
            complexity_result = complexity_result.view_as(viewing_shape)

            return complexity_result, neuron_hidden

        # No shared complexity (is `None` -> apply default tanh).
        return torch.tanh(neuron_model_input), tuple(torch.zeros(0))

    def forward(
        self,
        input_data: torch.Tensor,
        hidden: torch.Tensor,
        neuron_hidden: Tuple[torch.Tensor, ...],
    ) -> Tuple[
        torch.Tensor, Optional[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]
    ]:
        """
        Forward step the cell. One time step.

        :param input_data: input data of the model (current time step).
        :param hidden: output data of the previous step (or zeros if first step).
        :param neuron_hidden: Tuple of hidden states of the neurons (needed for RNN neuron models).
        :return: Returns tuple of the output of the layer model (first value), second value is
        optional output of the RNN model linearity (in case we want to return it), otherwise `None`,
        and the third value is tuple of new neuron model hidden states (needed in RNN neuron models).
        """
        # Apply RNN linearity step.
        rnn_out = self.rnn_cell(input_data, hidden)

        # Apply non-linearity (either function or DNN model of neuron).
        complexity_result, neuron_hidden = self.apply_complexity(rnn_out, neuron_hidden)
        # complexity_result = self.apply_complexity(hidden_exc, hidden_inh)

        if not self.return_recurrent_state:
            # In case we do not want ot return RNN results -> return None
            return complexity_result, None, neuron_hidden

        if self.model_type in nn_model.globals.SEPARATE_MODELS:
            # Ensuring returning 1D vector of neuron model inputs (for model analysis).
            return complexity_result, rnn_out[0] + rnn_out[1], neuron_hidden

        return complexity_result, rnn_out, neuron_hidden
