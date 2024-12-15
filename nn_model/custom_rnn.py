"""
This script contains definition of the custom RNN class used in the model. 
It is especially needed due the need to play with custom neuron models instead 
of just simple non-linearity function.
"""

from typing import List, Dict, Tuple

import torch
import torch.nn as nn

import nn_model.globals
from nn_model.type_variants import LayerConstraintFields, WeightTypes


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
        layer_name: str,
        input_constraints: List[Dict],
    ):
        """
        Initializes the custom RNNCell module.

        :param input_size: Size of the input.
        :param hidden_size: Size of the layer itself.
        :param layer_name: Name of the layer which this module represents
        (to determine whether it is excitatory or inhibitory).
        :param input_constraints:
        """
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.layer_name = layer_name
        self.input_constraints = input_constraints

        # Select excitatory and inhibitory indices and determine their sizes.
        self.excitatory_indices, self.inhibitory_indices = (
            self._get_excitatory_inhibitory_layer_indices()
        )
        self.excitatory_size = len(self.excitatory_indices)
        self.inhibitory_size = len(self.inhibitory_indices)

        # Define module weights and biases.
        # TODO: Adding exc/inh weights.
        self.weights_ih = nn.Linear(input_size, hidden_size)
        # self.weights_ih_exc = nn.Linear(
        #     self.excitatory_size, hidden_size
        # )  # Input excitatory
        # self.weights_ih_inh = nn.Linear(
        #     self.inhibitory_size, hidden_size
        # )  # Input inhibitory
        self.weights_hh = nn.Linear(hidden_size, hidden_size)

        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
        # self.b_ih_exc = nn.Parameter(torch.Tensor(hidden_size))  # Input excitatory
        # self.b_ih_inh = nn.Parameter(torch.Tensor(hidden_size))  # Input inhibitory
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))

        self._init_weights()

    def _get_excitatory_inhibitory_layer_indices(self) -> Tuple[List[int], List[int]]:
        """
        Split indices of the input layer to those that belongs to excitatory
        and those that belong to inhibitory part.

        :return: Returns tuple on excitatory and inhibitory lists of indices of
        these parts of input layer (without self-recurrent connection).
        """
        # Lists of indices:
        excitatory_indices: List[int] = []
        inhibitory_indices: List[int] = []
        # Start index of the following part.
        start_index = 0
        for constraint in self.input_constraints:
            # Iterate all input layers and its properties.
            layer_type = constraint[LayerConstraintFields.TYPE.value]  # Exc/Inh
            layer_size = constraint[LayerConstraintFields.SIZE.value]

            indices_to_add = []  # Placeholder for indices list.
            if layer_type == WeightTypes.EXCITATORY.value:
                # Current layer is excitatory -> add indices to excitatory part
                indices_to_add = excitatory_indices
            elif layer_type == WeightTypes.INHIBITORY.value:
                # Current layer is inhibitory -> add indices to inhibitory part
                indices_to_add = inhibitory_indices

            # Add indices to
            indices_to_add.extend(range(start_index, start_index + layer_size))
            start_index += layer_size

        return excitatory_indices, inhibitory_indices

    def _init_weights(self):
        """
        Initializes module weights and biases. Weights are initialized using
        uniform distribution. Biases are initialized with zeros.
        """
        nn.init.kaiming_uniform_(self.weights_ih.weight, nonlinearity="linear")
        # nn.init.kaiming_uniform_(self.weights_ih_exc.weight, nonlinearity="linear")
        # nn.init.kaiming_uniform_(self.weights_ih_inh.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.weights_hh.weight, nonlinearity="linear")
        nn.init.zeros_(self.b_ih)
        # nn.init.zeros_(self.b_ih_exc)
        # nn.init.zeros_(self.b_ih_inh)
        nn.init.zeros_(self.b_hh)

    def forward(
        self, input_data: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward step the cell. One time step.

        :param input_data: Input data (all inputs without itself).
        :param hidden: Output data of the previous step (previous predictions of itself
        modified by non-linearity function).
        :return: Returns linearity step of the RNNCell module.
        NOTE: It expects that the non-linearity would be used outside of the module.
        """
        # # Take excitatory and inhibitory parts
        # input_excitatory = input_data[:, self.excitatory_indices]
        # input_inhibitory = input_data[:, self.inhibitory_indices]

        # # Apply linear step to inhibitory and excitatory part
        # in_exc_linear = self.weights_ih_exc(input_excitatory) + self.b_ih_exc
        # in_inh_linear = self.weights_ih_inh(input_inhibitory) + self.b_ih_inh

        # # Apply linear step to self recurrent connection and
        # # decide whether it is excitatory or inhibitory.
        # hidden_linear = self.weights_hh(hidden) + self.b_hh
        # if self.layer_name in nn_model.globals.EXCITATORY_LAYERS:
        #     # In case excitatory layer -> add to excitatory part the self recurrent part
        #     in_exc_linear += hidden_linear
        # elif self.layer_name in nn_model.globals.INHIBITORY_LAYERS:
        #     # Inhibitory layer -> add self recurrent part to inhibitory
        #     in_inh_linear += hidden_linear

        # return in_exc_linear, in_inh_linear

        in_linear = self.weights_ih(input_data) + self.b_ih
        hidden_linear = self.weights_hh(hidden) + self.b_hh
        return in_linear + hidden_linear
