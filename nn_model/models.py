"""
This source code contains definition of all models used in our experiments.
"""

import torch
import torch.nn as nn

import globals
from globals import LayerType
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
    def __init__(
        self,
        size: int,
        layer_type,
        input_layers,  # List input layer names.
        shared_complexity=None,
    ):

        self.size = size
        self.layer_type = layer_type
        self.input_layers = input_layers
        self.shared_complexity = shared_complexity

        input_constraints = self._determine_input_constraints()
        self.constraint = self._determine_constraint(layer_type, input_constraints)

    def _determine_input_constraints(self):
        return [
            {
                "part_size": globals.MODEL_SIZES[layer],
                "part_type": self._get_constraint_type(layer),
            }
            for layer in self.input_layers
        ]

    def _get_constraint_type(self, layer_type):
        if layer_type in globals.EXCITATORY_LAYERS:
            return WeightTypes.EXCITATORY.value
        if layer_type in globals.INHIBITORY_LAYERS:
            return WeightTypes.INHIBITORY.value

        raise f"Wrong layer type. The type {layer_type} does not exist."

    def _determine_constraint(self, layer_type, input_constraints):
        if layer_type in globals.EXCITATORY_LAYERS:
            return ExcitatoryWeightConstraint(input_constraints)
        if layer_type in globals.INHIBITORY_LAYERS:
            return InhibitoryWeightConstraint(input_constraints)

        # Apply no constraint.
        return None


class RNNCellModel(nn.Module):
    """ """

    layers_inputs = {
        LayerType.V1_Exc_L4.value: [
            LayerType.X_ON.value,
            LayerType.X_OFF.value,
            LayerType.V1_Inh_L4.value,
            LayerType.V1_Exc_L23.value,
        ],
        LayerType.V1_Inh_L4.value: [
            LayerType.X_ON.value,
            LayerType.X_OFF.value,
            LayerType.V1_Exc_L4.value,
            LayerType.V1_Exc_L23.value,
        ],
        LayerType.V1_Exc_L23.value: [
            LayerType.V1_Exc_L4.value,
            LayerType.V1_Inh_L23.value,
        ],
        LayerType.V1_Inh_L23.value: [
            LayerType.V1_Exc_L4.value,
            LayerType.V1_Exc_L23.value,
        ],
    }

    def __init__(
        self,
        layer_sizes,
        rnn_cell_cls=ConstrainedRNNCell,
        complexity_kwargs={},
    ):
        super(RNNCellModel, self).__init__()

        self.rnn_cell_cls = rnn_cell_cls  # Store the RNN cell class
        # Kwargs to store complexity properties for various complexity types.
        self.compexity_kwargs = complexity_kwargs

        # Needed for model architecture definition
        self.layer_sizes = layer_sizes
        self.layers_configs = self._init_layer_configs(
            layer_sizes, self._init_shared_complexities(layer_sizes)
        )
        self.weight_constraints = self._init_weights_constraints()
        self._init_model_architecture()

    def _init_simple_complexity(self):
        return {layer: None for layer in RNNCellModel.layers_inputs.keys()}

    def _init_complex_complexities(self, layer_sizes):
        complexity_size: int = self.compexity_kwargs["complex_size"]
        return {
            layer: SharedComplexity(layer_sizes[layer], complexity_size=complexity_size)
            for layer in RNNCellModel.layers_inputs.keys()
        }

    def _init_shared_complexities(self, layer_sizes):
        if self.rnn_cell_cls == ComplexConstrainedRNNCell:
            # Complex complexity.
            return self._init_complex_complexities(layer_sizes)

        # Simple complexity (no additional complexity).
        return self._init_simple_complexity()

    def _init_layer_configs(self, layer_sizes, shared_complexities):
        return {
            layer: LayerConfig(
                layer_sizes[layer],
                layer,
                RNNCellModel.layers_inputs[layer],
                shared_complexities[layer],
            )
            for layer in RNNCellModel.layers_inputs.keys()
        }

    def _init_weights_constraints(self):
        return {
            layer: layer_config.constraint
            for layer, layer_config in self.layers_configs.items()
        }

    def _init_layer(self, layer, rnn_cell_cls):
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
        self.layers = nn.ModuleDict()

        for layer in RNNCellModel.layers_inputs.keys():
            self.layers[layer] = self._init_layer(layer, self.rnn_cell_cls)

    def forward(self, x_on, x_off, h4_exc, h4_inh, h23_exc, h23_inh):
        outputs = {layer: [] for layer in RNNCellModel.layers_inputs.keys()}

        for t in range(x_on.size(1)):
            # if t % 100 == 0:
            #     print(f"Got to iteration: {t}")
            #     torch.cuda.empty_cache()

            # LGN
            input_t = torch.cat((x_on[:, t, :], x_off[:, t, :]), dim=1).to(
                globals.device0
            )

            # L4:
            ## L4_Exc
            L4_input_exc = torch.cat((input_t, h4_inh, h23_exc), dim=1).to(
                globals.device0
            )
            self.layers[LayerType.V1_Exc_L4.value].to(globals.device0)
            h4_exc = self.layers[LayerType.V1_Exc_L4.value](L4_input_exc, h4_exc)
            ## L4_Inh
            L4_input_inh = torch.cat((input_t, h4_exc, h23_exc), dim=1).to(
                globals.device0
            )
            self.layers[LayerType.V1_Inh_L4.value].to(globals.device0)
            h4_inh = self.layers[LayerType.V1_Inh_L4.value](L4_input_inh, h4_inh)

            ## Collect L4 outputs
            outputs[LayerType.V1_Exc_L4.value].append(h4_exc.unsqueeze(1).cpu())
            outputs[LayerType.V1_Inh_L4.value].append(h4_inh.unsqueeze(1).cpu())

            # L23:
            ## L23_Exc
            L23_input_exc = torch.cat((h4_exc, h23_inh), dim=1).to(globals.device0)
            self.layers[LayerType.V1_Exc_L23.value].to(globals.device0)
            h23_exc = self.layers[LayerType.V1_Exc_L23.value](L23_input_exc, h23_exc)
            ## L23_Inh
            L23_input_inh = torch.cat((h4_exc, h23_exc), dim=1).to(globals.device0)
            self.layers[LayerType.V1_Inh_L23.value].to(globals.device0)
            h23_inh = self.layers[LayerType.V1_Inh_L23.value](L23_input_inh, h23_inh)

            # Collect L23 outputs
            outputs[LayerType.V1_Exc_L23.value].append(h23_exc.unsqueeze(1).cpu())
            outputs[LayerType.V1_Inh_L23.value].append(h23_inh.unsqueeze(1).cpu())

        # Clear caches
        del x_on, x_off, input_t, L4_input_inh, L23_input_exc, L23_input_inh
        torch.cuda.empty_cache()

        return outputs
