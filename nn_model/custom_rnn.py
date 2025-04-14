"""
This script contains definition of the custom RNN class used in the model.
It is especially needed due the need to play with custom neuron models instead
of just simple non-linearity function.
"""

from typing import List, Dict, Tuple

import torch
import torch.nn as nn

import nn_model.globals
from nn_model.type_variants import (
    LayerConstraintFields,
    WeightTypes,
    WeightsInitializationTypes,
)


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
        model_type: str,
        weight_initialization_type: str,
    ):
        """
        Initializes the custom RNN module.

        :param input_size: Size of the input.
        :param hidden_size: Size of the layer itself.
        :param layer_name: Name of the layer which this module represents
        (to determine whether it is excitatory or inhibitory).
        :param input_constraints: List of properties of each input layer
        (its size and type (inhibitory/excitatory)).
        :param model_type: Variant of the complex neuron (value from `ModelTypes`).
        :param weight_initialization_type: Which type of weight initialization we want to use.
        """
        super(CustomRNNCell, self).__init__()

        # Layer type
        self.layer_name = layer_name

        # List of properties of input.
        self.input_constraints = input_constraints

        # assert model_type in nn_model.globals.COMPLEX_MODELS
        self.model_type = model_type

        # Select excitatory and inhibitory indices and determine their sizes.
        self.excitatory_indices, self.inhibitory_indices = (
            self._get_excitatory_inhibitory_layer_indices()
        )

        # Layer sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.excitatory_size = len(self.excitatory_indices)
        self.inhibitory_size = len(self.inhibitory_indices)

        # Define module weights and biases.
        self.weights_ih_exc = nn.Linear(
            self.excitatory_size, hidden_size
        )  # Input excitatory
        self.weights_ih_inh = nn.Linear(
            self.inhibitory_size, hidden_size
        )  # Input inhibitory
        self.weights_hh = nn.Linear(hidden_size, hidden_size)  # Self-connection

        self._init_weights(weight_initialization_type)

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

    def _init_pytorch_default_weights(self):
        """
        Initializes module weights using `nn.init.kaiming_uniform_` function.
        """
        nn.init.kaiming_uniform_(self.weights_ih_exc.weight)
        nn.init.kaiming_uniform_(self.weights_ih_inh.weight)
        nn.init.kaiming_uniform_(self.weights_hh.weight)

    def _init_normal_weights(self, mean: float = 0.02, std: float = 0.01):
        """
        TODO: more general usage
        Initializes module weights using normal distribution that has larger
        mean in inhibitory part.

        :param mean: excitatory part mean.
        :param std: excitatory part std.
        """
        # Multiplier of the recurrent connection weights (based on inh/exc)
        recurrent_layer_multiplier = (
            1 if self.layer_name in nn_model.globals.EXCITATORY_LAYERS else -1
        )
        # Multiplier of the means (based on inh/exc). - inh has larger mean
        mean_multiplier = (
            1 if self.layer_name in nn_model.globals.EXCITATORY_LAYERS else 4
        )
        # Multiplier of the std (based on inh/exc). - inh has larger std
        std_multiplier = (
            mean_multiplier
            if self.layer_name in nn_model.globals.EXCITATORY_LAYERS
            else 2
        )

        nn.init.normal_(
            self.weights_ih_exc.weight,
            mean=mean * 1 * 1,
            std=std,
        )
        nn.init.normal_(
            self.weights_ih_inh.weight,
            mean=mean * -1 * 4,
            std=std * 2,
        )
        torch.nn.init.normal_(
            self.weights_hh.weight,
            mean=mean * recurrent_layer_multiplier * mean_multiplier,
            std=std * std_multiplier,
        )

    @torch.no_grad()
    def _flip_weights_signs(self, constraint_multiplier: int):
        """
        Flips weights of the model based on the layer it belongs to either
        non-positive (inhibitory) or non-negative (excitatory) values.

        :param constraint_multiplier: Multiplier used on self-recurrent weights
        (either `-1` if inhibitory or `1` if excitatory).
        """
        self.weights_ih_exc.weight.abs_()  # in-place
        self.weights_ih_inh.weight.copy_(-self.weights_ih_inh.weight.abs())
        self.weights_hh.weight.copy_(
            constraint_multiplier * self.weights_hh.weight.abs()
        )

    def _init_weights(self, weight_initialization_type: str):
        """
        Initializes module weights and biases. Weights are initialized using
        uniform distribution. Biases are initialized with zeros.

        :param weight_initialization_type: Which type of weight initialization we want to use.
        """
        # Set self-recurrent weight constraint based on the layer type.
        self_recurrent_multiplier = (
            1 if self.layer_name in nn_model.globals.EXCITATORY_LAYERS else -1
        )

        # Init weights:
        if weight_initialization_type == WeightsInitializationTypes.DEFAULT.value:
            # Use default pytorch weight initialization.
            self._init_pytorch_default_weights()
        elif weight_initialization_type == WeightsInitializationTypes.NORMAL.value:
            # Use Normal distribution initialization (with shifted larger mean in inhibitory).
            self._init_normal_weights()
        else:

            class WrongWeightsInitializationType(Exception):
                """
                Exception raised when wrong weights initialization type is used.
                """

            raise WrongWeightsInitializationType("Wrong weights initialization type.")

        # Flip weights to its correct sign (exc/inh).
        self._flip_weights_signs(self_recurrent_multiplier)

    def forward(
        self, input_data: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward step the cell. One time step.

        :param input_data: Input data (all inputs without itself).
        :param hidden: Output data of the previous step (previous predictions of itself
        modified by non-linearity function).
        :return: Returns result of linearity step of the RNN layer (either sum of inhibitory and
        excitatory (`_joint` models) or tuple of both outputs separately (`_separate` models).

        In case it returns excitatory and inhibitory results separately, it returns tuple where
        first value is excitatory result and second value is inhibitory.
        NOTE: It expects that the non-linearity would be used outside of the module.
        """
        # Take excitatory and inhibitory parts
        input_excitatory = input_data[:, self.excitatory_indices]
        input_inhibitory = input_data[:, self.inhibitory_indices]

        # Apply linear step to inhibitory and excitatory part.
        in_exc_linear = self.weights_ih_exc(input_excitatory)
        in_inh_linear = self.weights_ih_inh(input_inhibitory)

        # Apply linear step to self recurrent connection and
        # decide whether it is excitatory or inhibitory.
        hidden_linear = self.weights_hh(hidden)
        if self.layer_name in nn_model.globals.EXCITATORY_LAYERS:
            # In case excitatory layer -> add to excitatory part the self recurrent part
            in_exc_linear += hidden_linear
        elif self.layer_name in nn_model.globals.INHIBITORY_LAYERS:
            # Inhibitory layer -> add self recurrent part to inhibitory
            in_inh_linear += hidden_linear

        if self.model_type in nn_model.globals.JOINT_MODELS:
            # In case we want to return the sum of inhibitory and excitatory part.
            return in_exc_linear + in_inh_linear

        # In case we want to return the tuple of excitatory and inhibitory linear part.
        return in_exc_linear, in_inh_linear
