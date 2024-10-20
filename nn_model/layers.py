"""
This source code defines variants of possible model layers
typically in some form of `RNNCell` module with additional
operations and complexities.
"""

import torch.nn as nn


class ConstrainedRNNCell(nn.Module):
    """
    Class defining RNNCell (model layer) constrained with
    model assumptions (inhibitory/excitatory layers).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
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
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.constraint = weight_constraint
        self.shared_complexity = shared_complexity

    def _apply_complexity(self, rnn_output):
        if self.shared_complexity:
            batch_size = rnn_output.size(0)
            layer_size = rnn_output.size(1)

            # Reshape the layer output to [batch_size * hidden_size, 1] for batch processing
            complexity_result = rnn_output.view(batch_size * layer_size, 1)

            # Apply the small network to all elements in parallel
            # processed_output = self.layers_configs[layer].neuron_model(reshaped_input)
            complexity_result = self.shared_complexity(complexity_result)

            # Reshape back to [batch_size, hidden_size]
            complexity_result = complexity_result.view(batch_size, self.hidden_size)

            return complexity_result

        # No shared complexity (is `None` -> apply identity)
        return rnn_output

    def forward(self, input_data, hidden):
        """
        Forward step the cell. One time step.

        :param input_data: input data.
        :param hidden: output data of the previous step (or zeros if first step).
        :return: Returns the output of the forward step.
        """
        hidden = self.rnn_cell(input_data, hidden)
        return self._apply_complexity(hidden)
        # return hidden

    def apply_constraints(self):
        """
        Applies the layer constraint on the weights.
        """
        self.constraint.apply(self.rnn_cell)


# class ComplexConstrainedRNNCell(ConstrainedRNNCell):
#     """
#     TODO: Applying shared complexity.
#     """

#     def __init__(self, input_size, hidden_size, weight_constraint, shared_complexity):
#         # Inherit from ConstrainedRNNCell
#         super(ComplexConstrainedRNNCell, self).__init__(
#             input_size, hidden_size, weight_constraint
#         )
#         self.shared_complexity = shared_complexity  # Shared complexity module

#     def forward(self, input_data, hidden):
#         # Apply the RNN operation
#         hidden = self.rnn_cell(input_data, hidden)

#         # Apply the shared complexity transformation
#         complex_hidden = self.shared_complexity(hidden)

#         # Combine the original hidden state and the transformed one
#         combined_hidden = hidden + complex_hidden

#         return combined_hidden

#     def apply_constraints(self):
#         # Apply weight constraints inherited from ConstrainedRNNCell
#         self.constraint.apply(self.rnn_cell)
