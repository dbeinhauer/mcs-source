"""
This script contains definition of the custom RNN class used in the model. 
It is especially needed due the need to play with custom neuron models instead 
of just simple non-linearity function.
"""

import torch
import torch.nn as nn


class CustomRNNCell(nn.Module):
    """
    Class which defines custom RNNCell module. It should be same as in the implementation
    of pytorch `RNNCell` from there: https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html

    There is one major modification and it is that we do not use any non-linearity at the end.
    We expect that the non-linearity function would be applied after forward step outside
    of the module.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ):
        """
        Initializes the custom RNNCell module.

        :param input_size: Size of the input.
        :param hidden_size: Size of the layer itself.
        """
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define module weights and biases.
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))

        self._init_weights()

    def _init_weights(self):
        """
        Initializes module weights and biases. Weights are initialized using
        uniform distribution. Biases are initialized with zeros.
        """
        nn.init.kaiming_uniform_(self.W_ih.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.W_hh.weight, nonlinearity="linear")
        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

    def forward(self, input_data: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward step the cell. One time step.

        :param input_data: Input data (all inputs without itself).
        :param hidden: Output data of the previous step (previous predictions of itself
        modified by non-linearity function).
        :return: Returns linearity step of the RNNCell module.
        NOTE: It expects that the non-linearity would be used outside of the module.
        """
        in_linear = self.W_ih(input_data) + self.b_ih
        hidden_linear = self.W_hh(hidden) + self.b_hh
        return in_linear + hidden_linear
