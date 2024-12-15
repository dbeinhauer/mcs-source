"""
This source code defines variants of possible model layers
typically in some form of `RNNCell` module with additional
operations and complexities.
"""

from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn

from nn_model.type_variants import LayerType, ModelTypes
from nn_model.custom_rnn import CustomRNNCell


class ConstrainedRNNCell(nn.Module):
    """
    Class defining RNNCell (model layer) constrained with
    model assumptions (inhibitory/excitatory layers).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layer_name: str,
        weight_constraint,
        input_constraints: List[Dict],  # TODO: add docstring
        shared_complexity=None,
    ):
        """
        Initializes layer parameters and constraints.

        :param input_size: input size of the layer.
        :param hidden_size: hidden (output) size of the layer.
        :param layer_name: Name of the layer (to determine whether it is excitatory or inhibitory).
        :param weight_constraint: constraints of the layer.
        :param input_constraints: List of properties of each input layer
        (its size and type (inhibitory/excitatory)).
        :param shared_complexity: placeholder for shared complexity
        model used in more complex models. Here only for proper header
        definition.
        """
        super(ConstrainedRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Check layer type is one of our expected output layers.
        assert layer_name in [
            LayerType.V1_EXC_L4.value,
            LayerType.V1_INH_L4.value,
            LayerType.V1_EXC_L23.value,
            LayerType.V1_INH_L23.value,
        ]
        self.layer_name = layer_name

        self.model_type = (
            ModelTypes.SIMPLE.value
            if not shared_complexity
            else shared_complexity.model_type
        )
        self.rnn_cell = CustomRNNCell(
            input_size, hidden_size, layer_name, input_constraints, self.model_type
        )
        self.constraint = weight_constraint
        self.shared_complexity = shared_complexity

        # Flag whether we want to return also RNN outputs of the layer (for model analysis).
        self.return_recurrent_state = False

    def switch_to_return_recurrent_state(self):
        """
        Changes state of the layer to return recurrent time steps.
        """
        self.return_recurrent_state = True

    def apply_constraints(self):
        """
        Applies the layer constraint on the weights.
        """
        self.constraint.apply(self.rnn_cell)

    def apply_complexity(self, rnn_output: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # def apply_complexity(
        #     self, excitatory_input: torch.Tensor, inhibitory_input: torch.Tensor
        # ) -> torch.Tensor:
        """
        Applies complexity layer (shared DNN of neuron) on RNN output.

        :param rnn_output: Output of RNN layer to apply complexity on.
        :return: Returns output after shared complexity is applied.
        """

        combined_input = rnn_output
        # combined_input = torch.zeros(0)
        if len(rnn_output) == 2:
            # In case we work with separate excitatory and inhibitory inputs.
            # combined_input = torch.stack((rnn_output[0], rnn_output[1]), dim=2)
            combined_input = torch.stack(rnn_output, dim=2)
        # else:
        #     combined_input =

        # If shared complexity is defined (model is complex) -> apply complexity.
        if self.shared_complexity:
            # Reshape the layer output to [batch_size * hidden_size, 2] for batch processing
            # complexity_result = rnn_output.view(-1, 1)
            complexity_result = combined_input.view(
                -1, self.shared_complexity.input_size
            )

            # Apply the small network to all elements in parallel
            complexity_result = self.shared_complexity(complexity_result)
            # complexity_result = self.shared_complexity(complexity_result)

            # Reshape back to [batch_size, hidden_size]
            complexity_result = complexity_result.view_as(rnn_output)
            # complexity_result = complexity_result.view_as(rnn_output)

            return complexity_result

        # No shared complexity (is `None` -> apply default tanh).
        # TODO: maybe add option to apply different complexity functions.
        return torch.tanh(combined_input)
        # return torch.tanh(excitatory_input + inhibitory_input)

    def forward(
        self, input_data: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward step the cell. One time step.

        :param input_data: input data.
        :param hidden: output data of the previous step (or zeros if first step).
        :return: Returns tuple of the output of the forward step and optionally
        output of the RNN layer in case we switch the layer to do so.
        """
        # RNN linear step.
        rnn_out = self.rnn_cell(input_data, hidden)
        # hidden_exc, hidden_inh = self.rnn_cell(input_data, hidden)

        # if self.model_type != ModelTypes.COMPLEX_SEPARATE.value:
        #     rnn_out = tuple(rnn_out)

        # Apply non-linearity (either function or DNN model of neuron).
        complexity_result = self.apply_complexity(rnn_out)
        # complexity_result = self.apply_complexity(hidden_exc, hidden_inh)

        if not self.return_recurrent_state:
            # In case we do not want ot return RNN results -> return None
            return complexity_result, None

        if self.model_type == ModelTypes.COMPLEX_SEPARATE.value:
            # Ensuring returning 1D vector of input complexity.
            return complexity_result, rnn_out[0] + rnn_out[1]

        return complexity_result, rnn_out
        # return complexity_result, hidden_exc + hidden_inh/
