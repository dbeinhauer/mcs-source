"""
This source code contains definition of all models used in our experiments.
"""

from typing import List, Dict

import torch
import torch.nn as nn

import globals
from type_variants import LayerType
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
        input_layers: List[str],  # List input layer names.
        shared_complexity=None,
    ):
        """
        Initializes configuration based on the given parameters.
        Determines weight constraints.

        :param size: size of the layer (number of neurons).
        :param layer_type: name of the layer.
        :param input_layers: ordered list of input layers of the layer.
        :shared_complexity: shared complexity model(s), if no then `None`.
        """
        self.size = size
        self.layer_type = layer_type
        self.input_layers = input_layers
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
                "part_size": globals.MODEL_SIZES[layer],
                "part_type": self._get_constraint_type(layer),
            }
            for layer in self.input_layers
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

        raise Exception(f"Wrong layer type. The type {layer_type} does not exist.")

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
    layers_inputs = {
        LayerType.V1_EXC_L4.value: [
            LayerType.X_ON.value,
            LayerType.X_OFF.value,
            LayerType.V1_INH_L4.value,
            LayerType.V1_EXC_L23.value,
        ],
        LayerType.V1_INH_L4.value: [
            LayerType.X_ON.value,
            LayerType.X_OFF.value,
            LayerType.V1_EXC_L4.value,
            LayerType.V1_EXC_L23.value,
        ],
        LayerType.V1_EXC_L23.value: [
            LayerType.V1_EXC_L4.value,
            LayerType.V1_INH_L23.value,
        ],
        LayerType.V1_INH_L23.value: [
            LayerType.V1_EXC_L4.value,
            LayerType.V1_EXC_L23.value,
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
        self.compexity_kwargs = complexity_kwargs

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
        return {layer: None for layer in RNNCellModel.layers_inputs.keys()}

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
            layer: SharedComplexity(layer_sizes[layer], **complexity_kwargs)
            for layer in RNNCellModel.layers_inputs.keys()
        }

    def _init_shared_complexities(
        self,
        layer_sizes: Dict[str, int],
    ) -> Dict:
        """
        Initializes shared complexities of the model.

        :param layer_sizes: sizes of all model layers (input included).
        :return: Retruns dictionary of layer name (`LayerType`) and
        appropriate shared complexity object.
        """
        if self.rnn_cell_cls == ComplexConstrainedRNNCell:
            # Complex complexity.
            return self._init_complex_complexities(
                layer_sizes, self.compexity_kwargs["complex"]
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
                RNNCellModel.layers_inputs[layer],
                shared_complexities[layer],
            )
            for layer in RNNCellModel.layers_inputs.keys()
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
                self.layer_sizes[input_layer]
                for input_layer in RNNCellModel.layers_inputs[layer]
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

        for layer in RNNCellModel.layers_inputs.keys():
            self.layers[layer] = self._init_layer(layer, self.rnn_cell_cls)

    # def forward(self, x_on, x_off, h4_exc, h4_inh, h23_exc, h23_inh):
    def forward(self, inputs, targets):
        """

        :param inputs: dict[layer, inputs] - size (batch, time, neurons)
        :param targets: dict[layer, inputs] - size (batch, neurons) or (batch, time, neurons)
        :return: _description_
        """
        outputs = {layer: [] for layer in RNNCellModel.layers_inputs.keys()}

        time_length = inputs[LayerType.X_ON.value].size(1)

        hidden_layers = targets
        if not self.training:
            hidden_layers = {
                layer: hidden.to(globals.device0)
                for layer, hidden in hidden_layers.items()
            }

        # Start from the second step, because the first one is
        # the initial one (we predict all time steps but the )
        for t in range(1, time_length):  # x_on.size(1)):
            if self.training:
                # In case we train, we assign new hidden layers based on the target in each step.
                hidden_layers = {
                    layer: target[:, t - 1, :].clone().to(globals.device0)
                    for layer, target in targets.items()
                }

            # if t % 100 == 0:
            #     print(f"Got to iteration: {t}")
            #     torch.cuda.empty_cache()

            # LGN
            # input_t = torch.cat(
            #     (x_on[:, t, :], x_off[:, t, :]),
            #     dim=1,
            # ).to(globals.device0)
            input_t = torch.cat(
                (
                    inputs[LayerType.X_ON.value][:, t, :],
                    inputs[LayerType.X_OFF.value][:, t, :],
                ),
                dim=1,
            ).to(globals.device0)

            # L4:
            ## L4_Exc
            L4_input_exc = torch.cat(
                # (input_t, h4_inh, h23_exc),
                (
                    input_t,
                    hidden_layers[LayerType.V1_INH_L4.value],
                    hidden_layers[LayerType.V1_EXC_L23.value],
                ),
                dim=1,
            ).to(globals.device0)
            self.layers[LayerType.V1_EXC_L4.value].to(globals.device0)
            # h4_exc = self.layers[LayerType.V1_Exc_L4.value](L4_input_exc, h4_exc)
            h4_exc = self.layers[LayerType.V1_EXC_L4.value](
                L4_input_exc, hidden_layers[LayerType.V1_EXC_L4.value]
            )

            ## L4_Inh
            # L4_input_inh = torch.cat(
            #     (input_t, h4_exc, h23_exc),
            #     dim=1,
            # ).to(globals.device0)

            L4_input_inh = torch.cat(
                (
                    input_t,
                    hidden_layers[LayerType.V1_EXC_L4.value],
                    hidden_layers[LayerType.V1_EXC_L23.value],
                ),
                dim=1,
            ).to(globals.device0)
            self.layers[LayerType.V1_INH_L4.value].to(globals.device0)
            # h4_inh = self.layers[LayerType.V1_Inh_L4.value](L4_input_inh, h4_inh)
            h4_inh = self.layers[LayerType.V1_INH_L4.value](
                L4_input_inh, hidden_layers[LayerType.V1_INH_L4.value]
            )

            ## Collect L4 outputs
            outputs[LayerType.V1_EXC_L4.value].append(h4_exc.unsqueeze(1).cpu())
            outputs[LayerType.V1_INH_L4.value].append(h4_inh.unsqueeze(1).cpu())

            # L23:
            ## L23_Exc
            L23_input_exc = torch.cat(
                # (h4_exc, h23_inh),
                (
                    hidden_layers[LayerType.V1_EXC_L4.value],
                    hidden_layers[LayerType.V1_INH_L23.value],
                ),
                dim=1,
            ).to(globals.device0)
            self.layers[LayerType.V1_EXC_L23.value].to(globals.device0)
            # h23_exc = self.layers[LayerType.V1_Exc_L23.value](L23_input_exc, h23_exc)
            h23_exc = self.layers[LayerType.V1_EXC_L23.value](
                L23_input_exc, hidden_layers[LayerType.V1_EXC_L23.value]
            )
            ## L23_Inh
            L23_input_inh = torch.cat(
                # (h4_exc, h23_exc),
                (
                    hidden_layers[LayerType.V1_EXC_L4.value],
                    hidden_layers[LayerType.V1_EXC_L23.value],
                ),
                dim=1,
            ).to(globals.device0)
            self.layers[LayerType.V1_INH_L23.value].to(globals.device0)
            # h23_inh = self.layers[LayerType.V1_Inh_L23.value](L23_input_inh, h23_inh)
            h23_inh = self.layers[LayerType.V1_INH_L23.value](
                L23_input_inh, hidden_layers[LayerType.V1_INH_L23.value]
            )

            # Collect L23 outputs
            outputs[LayerType.V1_ECX_L23.value].append(h23_exc.unsqueeze(1).cpu())
            outputs[LayerType.V1_INH_L23.value].append(h23_inh.unsqueeze(1).cpu())

            if not self.training:
                hidden_layers[LayerType.V1_EXC_L4.value] = h4_exc
                hidden_layers[LayerType.V1_INH_L4.value] = h4_inh
                hidden_layers[LayerType.V1_EXC_L23.value] = h23_exc
                hidden_layers[LayerType.V1_INH_L23.value] = h23_inh

        # Clear caches
        del inputs, input_t, L4_input_inh, L23_input_exc, L23_input_inh
        torch.cuda.empty_cache()

        return outputs
