"""
This source code defines variants of possible model layers
typically in some form of `RNNCell` module with additional
operations and complexities.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn

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
        weight_constraint,
        shared_complexity=None,
    ):
        """
        Initializes layer parameters and constraints.

        :param input_size: input size of the layer.
        :param hidden_size: hidden (output) size of the layer.
        :param weight_constraint: constraints of the layer.
        :param shared_complexity: placeholder for shared complexity
        model used in more complex models. Here only for proper header
        definition.
        """
        super(ConstrainedRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = CustomRNNCell(input_size, hidden_size)
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

    def apply_complexity(self, rnn_output: torch.Tensor) -> torch.Tensor:
        """
        Applies complexity layer (shared DNN of neuron) on RNN output.

        :param rnn_output: Output of RNN layer to apply complexity on.
        :return: Returns output after shared complexity is applied.
        """
        # If shared complexity is defined (model is complex) -> apply complexity.
        if self.shared_complexity:
            # Reshape the layer output to [batch_size * hidden_size, 1] for batch processing
            complexity_result = rnn_output.view(-1, 1)

            # Apply the small network to all elements in parallel
            complexity_result = self.shared_complexity(complexity_result)

            # Reshape back to [batch_size, hidden_size]
            complexity_result = complexity_result.view_as(rnn_output)

            return complexity_result

        # No shared complexity (is `None` -> apply default tanh).
        # TODO: maybe add option to apply different complexity functions.
        return torch.tanh(rnn_output)

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
        hidden = self.rnn_cell(input_data, hidden)

        # Apply non-linearity (either function or DNN model of neuron).
        complexity_result = self.apply_complexity(hidden)

        if not self.return_recurrent_state:
            # In case we do not want ot return RNN results -> return None
            return complexity_result, None

        return complexity_result, hidden
