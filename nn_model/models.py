"""
This source code contains definition of all models used in our experiments.
"""

from typing import List, Dict, Tuple

import torch
import torch.nn as nn

import globals
from type_variants import LayerType, TimeStepVariant
from weights_constraints import (
    WeightTypes,
    ExcitatoryWeightConstraint,
    InhibitoryWeightConstraint,
)
from layers import (
    ConstrainedRNNCell,
    ComplexConstrainedRNNCell,
)
from neurons import SharedComplexity


class LayerConfig:
    """
    Class for storing configuration of model layer.
    """

    def __init__(
        self,
        size: int,
        layer_type: str,
        input_layers_parameters: List[Tuple[str, str]],  # List input layer names.
        shared_complexity=None,
    ):
        """
        Initializes configuration based on the given parameters.
        Determines weight constraints.

        :param size: size of the layer (number of neurons).
        :param layer_type: name of the layer.
        :param input_layers_parameters: ordered list of input layer parameters of the layer.
        The parameters are in form of tuple with first value its name from `LayerType`
        and second value its time step name from `TimeStepVariant`.
        :param shared_complexity: shared complexity model(s), if no then `None`.
        """
        self.size: int = size
        self.layer_type: str = layer_type
        self.input_layers_parameters: List[Tuple[str, str]] = input_layers_parameters
        self.shared_complexity = shared_complexity

        # Determine weight constraints for the layer (excitatory/inhibitory).
        input_constraints = self._determine_input_constraints()
        self.constraint = self._determine_constraint(layer_type, input_constraints)

    def _determine_input_constraints(self) -> List[Dict]:
        """
        Determines input weights constraint (chooses between excitatory/inhibitory).

        :return: Returns list of dictionaries with parameters specifying
        the distribution of input weight types of each input layer.

        The format is same as the expected kwargs for `WeightConstraint` objects.
        The order of the dictionaries should be same as the order of the input layers.

        The keys in the dictionaries are:
            'part_size' (int): size of the input layer.
            'part_type' (WeightTypes value): type of the layer (exc/inh).
        """
        return [
            {
                "part_size": globals.MODEL_SIZES[layer[0]],
                "part_type": self._get_constraint_type(layer[0]),
            }
            for layer in self.input_layers_parameters
        ]

    def _get_constraint_type(self, layer_type: str) -> str:
        """
        Determines type of the constraint that should be used for given layer.

        :param layer_type: name of the layer. Should be value from `LayerType`.
        :return: Returns identifier of constraint type (value from `WeightTypes`).
        """
        if layer_type in globals.EXCITATORY_LAYERS:
            return WeightTypes.EXCITATORY.value
        if layer_type in globals.INHIBITORY_LAYERS:
            return WeightTypes.INHIBITORY.value

        class WrongLayerException(Exception):
            """
            Exception class to be raised while wrong layer type chosen.
            """

        raise WrongLayerException(
            f"Wrong layer type. The type {layer_type} does not exist."
        )

    def _determine_constraint(self, layer_type: str, input_constraints: List[Dict]):
        """
        Determines weight constraints of the layer.

        :param layer_type: name of the layer. Should be value from `LayerType`,
        or different if we do not want to use the weight constraints.
        :param input_constraints: list of constraint kwargs for input layer weight constraints.
        :return: Returns appropriate `WeightConstraint` object,
        or `None` if we do not want to use the weight constraint.
        """
        if layer_type in globals.EXCITATORY_LAYERS:
            # Excitatory neurons.
            return ExcitatoryWeightConstraint(input_constraints)
        if layer_type in globals.INHIBITORY_LAYERS:
            # Inhibitory neurons.
            return InhibitoryWeightConstraint(input_constraints)

        # Apply no constraint.
        return None


class RNNCellModel(nn.Module):
    """
    Class defining models that use LGN outputs as input and predicts rest of the layers.
    """

    # Define model architecture (output layers with its inputs).
    layers_input_parameters = {
        LayerType.V1_EXC_L4.value: [
            (LayerType.X_ON.value, TimeStepVariant.CURRENT.value),
            (LayerType.X_OFF.value, TimeStepVariant.CURRENT.value),
            (LayerType.V1_INH_L4.value, TimeStepVariant.PREVIOUS.value),
            (LayerType.V1_EXC_L23.value, TimeStepVariant.PREVIOUS.value),
        ],
        LayerType.V1_INH_L4.value: [
            (LayerType.X_ON.value, TimeStepVariant.CURRENT.value),
            (LayerType.X_OFF.value, TimeStepVariant.CURRENT.value),
            (LayerType.V1_EXC_L4.value, TimeStepVariant.PREVIOUS.value),
            (LayerType.V1_EXC_L23.value, TimeStepVariant.PREVIOUS.value),
        ],
        LayerType.V1_EXC_L23.value: [
            (LayerType.V1_EXC_L4.value, TimeStepVariant.CURRENT.value),
            (LayerType.V1_INH_L23.value, TimeStepVariant.PREVIOUS.value),
        ],
        LayerType.V1_INH_L23.value: [
            (LayerType.V1_EXC_L4.value, TimeStepVariant.CURRENT.value),
            (LayerType.V1_EXC_L23.value, TimeStepVariant.PREVIOUS.value),
        ],
    }

    def __init__(
        self,
        layer_sizes: Dict[str, int],
        rnn_cell_cls=ConstrainedRNNCell,
        complexity_kwargs: Dict = {},
    ):
        """
        Initializes model parameters, sets weights constraints and creates model architecture.

        :param layer_sizes: sizes of all model layers (input included).
        :param rnn_cell_cls: type of the layer used in the model.
        :param complexity_kwargs: kwargs of the used complexities (if any).
        """
        super(RNNCellModel, self).__init__()

        self.rnn_cell_cls = rnn_cell_cls  # Store the RNN cell class
        # Kwargs to store complexity properties for various complexity types.
        self.complexity_kwargs = complexity_kwargs

        self.layer_sizes = layer_sizes  # Needed for model architecture definition

        # Layer configuration.
        self.layers_configs = self._init_layer_configs(
            layer_sizes, self._init_shared_complexities(layer_sizes)
        )

        # Init model.
        self._init_model_architecture()

    def _init_simple_complexity(self) -> Dict:
        """
        Initializes simple complexities (None complexity).

        :return: Returns dictionary of layer name (`LayerType`) and Nones.
        """
        return {layer: None for layer in RNNCellModel.layers_input_parameters}

    def _init_complex_complexities(
        self,
        layer_sizes: Dict[str, int],
        complexity_kwargs: Dict,
    ) -> Dict:
        """
        Initializes complex complexity layers.

        :param layer_sizes: sizes of all model layers (input included).
        :complexity_kwargs: kwargs of `SharedComplexity` object `__init__`.
        :return: Returns dictionary of layer name (`LayerType`) and appropriate shared
        complex complexity object.
        """
        return {
            layer[0]: SharedComplexity(layer_sizes[layer], **complexity_kwargs)
            for layer in RNNCellModel.layers_input_parameters
        }

    def _init_shared_complexities(
        self,
        layer_sizes: Dict[str, int],
    ) -> Dict:
        """
        Initializes shared complexities of the model.

        :param layer_sizes: sizes of all model layers (input included).
        :return: Returns dictionary of layer name (`LayerType`) and
        appropriate shared complexity object.
        """
        if self.rnn_cell_cls == ComplexConstrainedRNNCell:
            # Complex complexity.
            return self._init_complex_complexities(
                layer_sizes, self.complexity_kwargs["complex"]
            )

        # Simple complexity (no additional complexity).
        return self._init_simple_complexity()

    def _init_layer_configs(
        self,
        layer_sizes: Dict[str, int],
        shared_complexities: Dict,
    ) -> Dict[str, LayerConfig]:
        """
        Initializes `LayerConfig` objects for all layers of the model.

        :param layer_sizes: sizes of the layers.
        :param shared_complexities: shared complexities of the layers.
        :return: Returns dictionary of layer configurations for all model layers.
        """
        return {
            layer: LayerConfig(
                layer_sizes[layer],
                layer,
                input_parameters,
                shared_complexities[layer],
            )
            for layer, input_parameters in RNNCellModel.layers_input_parameters.items()
        }

    def _init_layer(self, layer: str, rnn_cell_cls):
        """
        Initializes one layer of the model.

        :param layer: layer name (`LayerType`).
        :param rnn_cell_cls: layer object variant.
        :return: Returns initializes layer object.
        """
        return rnn_cell_cls(
            sum(
                self.layer_sizes[layer_name]
                for layer_name, _ in RNNCellModel.layers_input_parameters[layer]
            ),
            self.layers_configs[layer].size,
            self.layers_configs[layer].constraint,
            self.layers_configs[layer].shared_complexity,
        )

    def _init_model_architecture(self):
        """
        Initializes all model layers and stored them as `nn.ModuleDict` object
        under the keys `LayerType`.
        """
        self.layers = nn.ModuleDict()

        for layer in RNNCellModel.layers_input_parameters:
            self.layers[layer] = self._init_layer(layer, self.rnn_cell_cls)

    def _init_hidden_layers(self, targets) -> Dict[str, torch.Tensor]:
        """
        Initializes hidden layers for targets based on the training mode.

        :param targets: dictionary containing the hidden layers for a neural network model.
        :return: Returns a dictionary of hidden layers moved to CUDA if in evaluation mode,
        otherwise (training mode) returns empty dictionary.
        """
        if not self.training:
            # Evaluation step. Use only time step 0 as initialization of hidden states.
            return {
                layer: hidden.to(globals.device0) for layer, hidden in targets.items()
            }
        # Training mode (initialization of the hidden layer is in each time step).
        return {}

    def _move_targets_to_cuda(
        self, targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Moves targets to CUDA if in training mode as they will be used as hidden states.
        We want to convert it once at the start of the training.

        :param targets: targets to be moved to CUDA.
        :return: Returns dictionary of targets moved to CUDA if in training mode, otherwise
        (in evaluation mode) returns empty dictionary.
        """
        if self.training:
            # In case we train, move all targets to CUDA
            # (will be used as hidden states during training)
            return {
                layer: target.clone().to(globals.device0)
                for layer, target in targets.items()
            }
        return {}

    def _assign_time_step_hidden_layers(
        self, time: int, targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Assigns hidden layers at previous time step based on the provided targets.

        NOTE: This function is used during training for assigning
        previous time steps of the model layers.

        :param time: current time steps of the model prediction.
        :param targets: model targets to assign previous time step for the hidden states.
        :return: Returns dictionary of hidden layers of previous time step.
        """
        return {layer: target[:, time - 1, :] for layer, target in targets.items()}

    def _get_layer_input_tensor(self, input_current_parts, input_previous_parts):
        """
        input_parts: are in form of tuple of import pytorch tensors.
        """
        # print(len(input_current_parts))
        # print(len(input_previous_parts))
        return torch.cat(
            input_current_parts + input_previous_parts,
            dim=1,
        ).to(globals.device0)

    def _get_list_by_time_variant(
        self, layer_type: str, time_variant: str, time_variant_layer_values
    ):
        """_summary_

        :param layer_type: type name (X_ON, V1_EXC_L4...)
        :param time_variant: time varian (previous, current)
        :param time_variant_layer_values: hidden_layer or current_time_values
        :return: _description_
        """
        time_variant_list = []
        for (
            input_part_layer_name,
            input_part_time_variant,
        ) in RNNCellModel.layers_input_parameters[layer_type]:
            if input_part_time_variant == time_variant:
                time_variant_list.append(
                    time_variant_layer_values[input_part_layer_name]
                )

        return time_variant_list

    def _get_all_input_tensors(self, inputs, hidden_layers, time):
        
        # We already have LGN inputs (assign them)
        current_time_outputs = {
            layer: layer_tensor[:, time, :] for layer, layer_tensor in inputs.items()
        }
        for layer in RNNCellModel.layers_input_parameters:
            # current_time_outputs[layer] = self.layers[LayerType.V1_EXC_L4.value](
            current_time_outputs[layer] = self.layers[layer](
                self._get_layer_input_tensor(
                    # input_t,
                    self._get_list_by_time_variant(
                        layer,
                        TimeStepVariant.CURRENT.value,
                        current_time_outputs,
                    ),  # time t
                    self._get_list_by_time_variant(
                        layer,
                        TimeStepVariant.PREVIOUS.value,
                        hidden_layers,
                    ),  # time t-1
                    # (
                    #     hidden_layers[LayerType.V1_INH_L4.value],
                    #     hidden_layers[LayerType.V1_EXC_L23.value],
                    # ),
                ),
                hidden_layers[layer],  # Recurrent itself time (t-1)
                # hidden_layers[LayerType.V1_EXC_L4.value],  # time t-1
            )

        return current_time_outputs

    def _append_outputs(self, all_outputs, time_step_outputs):
        for layer, layer_outputs in time_step_outputs.items():
            if layer in RNNCellModel.layers_input_parameters:
                all_outputs[layer].append(layer_outputs.unsqueeze(1).cpu())

    def forward(self, inputs, targets):
        """

        :param inputs: dict[layer, inputs] - size (batch, time, neurons)
        :param targets: dict[layer, inputs] - size (batch, neurons) or (batch, time, neurons)
        :return: _description_
        """
        # Init dictionary of model outputs.
        all_time_outputs = {layer: [] for layer in RNNCellModel.layers_input_parameters}

        hidden_layers = self._init_hidden_layers(targets)
        all_hidden_layers = self._move_targets_to_cuda(targets)

        # Start from the second step, because the first one is
        # the initial one (we predict all time steps but the 0-th one).
        time_length = inputs[LayerType.X_ON.value].size(1)
        for t in range(1, time_length):
            if self.training:
                # In case we train, we assign new hidden layers based on
                # the previous target in each step.
                hidden_layers = self._assign_time_step_hidden_layers(
                    t, all_hidden_layers
                )

            # Perform model step
            current_time_outputs = self._get_all_input_tensors(inputs, hidden_layers, t)

            # Append time step prediction to all predictions.
            self._append_outputs(all_time_outputs, current_time_outputs)

            if not self.training:
                hidden_layers = current_time_outputs

        # Clear caches
        del inputs
        torch.cuda.empty_cache()

        return all_time_outputs
