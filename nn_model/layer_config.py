"""
This script defined the class `LayerConfig` that is used to store the configuration
of the model layer. The class is used to determine the weight constraints of the layer and
serves to apply synaptic adaptation to the input tensor.
"""

from typing import List, Tuple, Dict, Optional

import torch

import nn_model.globals
from nn_model.type_variants import LayerConstraintFields, WeightTypes, WeightConstraint
from nn_model.weights_constraints import ConstraintRegistrar


class LayerConfig:
    """
    Class for storing configuration of model layer. The class is used to determine the weight
    constraints of the layer and serves to apply synaptic adaptation to the input tensor.
    """

    def __init__(
        self,
        size: int,
        layer_type: str,
        input_layers_parameters: List[Tuple[str, str]],
        neuron_model=None,
        synaptic_activation_models=None,
    ):
        """
        Initializes configuration based on the given parameters.
        Determines weight constraints.

        :param size: size of the layer (number of neurons).
        :param layer_type: name of the layer.
        :param input_layers_parameters: ordered list of input layer parameters of the layer.
        The parameters are in form of tuple with first value its name from `LayerType`
        and second value its time step name from `TimeStepVariant`.
        :param neuron_model: shared complexity model(s), if none then `None`.
        :param synaptic_activation_models: synaptic adaptation models for the layer,
        if none then `None`.
        """
        self.size: int = size
        self.layer_type: str = layer_type
        self.input_layers_parameters: List[Tuple[str, str]] = input_layers_parameters
        self.neuron_model = neuron_model
        self.synaptic_activation_models = synaptic_activation_models

        # Determine weight constraints for the layer (excitatory/inhibitory).
        self.input_constraints = (
            self._determine_input_constraints()
        )  # Constraints setup (for determining inh/excitatory in the architecture).
        self.constraint = self._determine_constraint(layer_type)

    def _determine_input_constraints(self) -> List[Dict]:
        """
        Determines input weights constraint (chooses between excitatory/inhibitory).

        :return: Returns list of dictionaries with parameters specifying
        the distribution of input weight types of each input layer.

        The format is same as the expected kwargs for `WeightConstraint` objects.
        The order of the dictionaries should be same as the order of the input layers.

        The keys in the dictionaries are:
            `part_size` (int): size of the input layer.
            `part_type` (WeightTypes value): type of the layer (exc/inh).
        """
        return [
            {
                LayerConstraintFields.SIZE.value: nn_model.globals.MODEL_SIZES[
                    layer[0]
                ],
                LayerConstraintFields.TYPE.value: self._get_constraint_type(layer[0]),
            }
            for layer in self.input_layers_parameters
        ]

    def _get_constraint_type(self, layer_type: str) -> str:
        """
        Determines type of the constraint that should be used for given layer.

        :param layer_type: name of the layer. Should be value from `LayerType`.
        :return: Returns identifier of constraint type (value from `WeightTypes`).
        """
        if layer_type in nn_model.globals.EXCITATORY_LAYERS:
            return WeightTypes.EXCITATORY.value
        if layer_type in nn_model.globals.INHIBITORY_LAYERS:
            return WeightTypes.INHIBITORY.value

        class WrongLayerException(Exception):
            """
            Exception class to be raised while wrong layer type chosen.
            """

        raise WrongLayerException(
            f"Wrong layer type. The type {layer_type} does not exist."
        )

    def _determine_constraint(self, layer_type: str) -> Optional[ConstraintRegistrar]:
        """
        Determines weight constraints of the layer.

        :param layer_type: name of the layer. Should be value from `LayerType`,
        or different if we do not want to use the weight constraints.
        :param input_constraints: list of constraint kwargs for input layer weight constraints.
        :return: Returns appropriate `WeightConstraint` object,
        or `None` if we do not want to use the weight constraint.
        """

        if layer_type in nn_model.globals.EXCITATORY_LAYERS:
            return ConstraintRegistrar(WeightTypes.EXCITATORY, WeightConstraint.SHARP)
        if layer_type in nn_model.globals.INHIBITORY_LAYERS:
            return ConstraintRegistrar(WeightTypes.INHIBITORY, WeightConstraint.SHARP)
        return None

    def apply_synaptic_adaptation(
        self,
        input_layer: str,
        input_tensor: torch.Tensor,
        hidden_states: Optional[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Applies proper synaptic adaptation model based on input layer to the input tensor.

        :param input_layer: Layer from which the input comes.
        :param input_tensor: Input values of the synaptic adaptation model.
        :param hidden_states: Hidden states of the synaptic adaptation model.
        :return: Returns output of the synaptic adaptation model.
        """
        # Reshape the layer output to [batch_size * hidden_size, neuron_model_input_size]
        # for batch processing (parallel application of the neuron module for all
        # the layer output values).
        synaptic_activation_model = self.synaptic_activation_models[input_layer]
        complexity_result = input_tensor.reshape(
            -1, synaptic_activation_model.input_size
        )

        # Apply the neuron model to all values at parallel.
        complexity_result, hidden_states = synaptic_activation_model(
            complexity_result, hidden_states
        )

        # Define the output shape.
        viewing_shape: torch.Tensor = input_tensor

        # Reshape back to [batch_size, hidden_size]
        complexity_result = complexity_result.view_as(viewing_shape)

        return complexity_result, hidden_states
