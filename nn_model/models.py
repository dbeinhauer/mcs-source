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


# class LayerType(Enum):
#     X_ON = auto()
#     X_OFF = auto()
#     V1_Exc_L4 = auto()
#     V1_Inh_L4 = auto()
#     V1_Exc_L23 = auto()
#     V1_Inh_L23 = auto()

#     EXCITATORY_LAYERS = {X_ON, X_OFF, V1_Exc_L4, V1_Exc_L23}
#     INHIBITORY_LAYERS = {V1_Inh_L4, V1_Inh_L23}


class LayerConfig:
    def __init__(
            self, 
            size: int, 
            layer_type, 
            input_layers, # List input layer names.
            shared_complexity=None
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
                'part_type': self._get_constraint_type(layer),
            }
            for layer in self.input_layers
        ]
    
    def _get_constraint_type(self, layer_type):
        if layer_type in LayerType.EXCITATORY_LAYERS:
            return WeightTypes.EXCITATORY
        if layer_type in LayerType.INHIBITORY_LAYERS:
            return WeightTypes.INHIBITORY
        
        raise f"Wrong layer type. The type {layer_type} does not exist."

    def _determine_constraint(self, layer_type, input_constraints):
        if layer_type in LayerType.EXCITATORY_LAYERS:
            return ExcitatoryWeightConstraint(input_constraints)
        if layer_type in LayerType.INHIBITORY_LAYERS:
            return InhibitoryWeightConstraint(input_constraints)

        # Apply no constraint.
        return None

class RNNCellModel(nn.Module):
    """
    
    """
    layers_inputs = {
        LayerType.V1_Exc_L4.value:
    }


    def __init__(
            self,
            # layer_names
            layer_sizes,
            rnn_cell_cls=ConstrainedRNNCell, 
            complexity_size: int=64,
        ):
        super(RNNCellModel, self).__init__()

        self.rnn_cell_cls = rnn_cell_cls  # Store the RNN cell class
        self.complexity_size = complexity_size  # For complex RNN cells
        
        # TODO: old variant
        self._init_layer_sizes(layer_sizes)

        self.layer_sizes = layer_sizes
        # self.layers_configs = self._init_layer_configs()
        self.weight_constraints = self._init_weights_constraints()
        self._init_model_architecture()

    def _init_layer_configs(self, layer_sizes, ):
        {
            LayerConfig()
        }
    
    def _init_layer_sizes(self, layer_sizes):
        # NOTE: probably not needed
        self.x_on_size = layer_sizes['X_ON']
        self.x_off_size = layer_sizes['X_OFF']
        self.l4_exc_size = layer_sizes['V1_Exc_L4']
        self.l4_inh_size = layer_sizes['V1_Inh_L4']
        self.l23_exc_size = layer_sizes['V1_Exc_L23']
        self.l23_inh_size = layer_sizes['V1_Inh_L23']

    def _init_weights_constraints(self):#, layer_config):
        # TODO: uncomment
        return {
            layer: layer_config.constraint
            for layer, layer_config in self.layers_config.items()
        }

        # l4_exc_args = [
        #     {'part_size': self.x_on_size, 'part_type': 'exc'},
        #     {'part_size': self.x_off_size, 'part_type': 'exc'},
        #     {'part_size': self.l4_inh_size, 'part_type': 'inh'},
        #     {'part_size': self.l23_exc_size, 'part_type': 'exc'},
        # ]
        # l4_inh_args = [
        #     {'part_size': self.x_on_size, 'part_type': 'exc'},
        #     {'part_size': self.x_off_size, 'part_type': 'exc'},
        #     {'part_size': self.l4_exc_size, 'part_type': 'exc'},
        #     {'part_size': self.l23_exc_size, 'part_type': 'exc'},
        # ]
        # l23_exc_args = [
        #     {'part_size': self.l4_exc_size, 'part_type': 'exc'},
        #     {'part_size': self.l23_inh_size, 'part_type': 'inh'},
        # ]
        # l23_inh_args = [
        #     {'part_size': self.l4_exc_size, 'part_type': 'exc'},
        #     {'part_size': self.l23_exc_size, 'part_type': 'exc'},
        # ]

        # self.l4_excitatory_constraint = ExcitatoryWeightConstraint(l4_exc_args)
        # self.l4_inhibitory_constraint = InhibitoryWeightConstraint(l4_inh_args)
        # self.l23_excitatory_constraint = ExcitatoryWeightConstraint(l23_exc_args)
        # self.l23_inhibitory_constraint = InhibitoryWeightConstraint(l23_inh_args)


    # def _init_layer(self, layer_name, input_layers_names):
    #TODO: uncomment
    #     return self.rnn_cell_cls(
                
    #             #input_size + self.l4_inh_size + self.l23_exc_size,
    #             self.l4_exc_size, 
    #             self.l4_excitatory_constraint,
    #             shared_complexity_l4_exc,
    #         )


    def _init_model_architecture(self):
        
        input_size = self.x_on_size + self.x_off_size

        # If using a complex RNN cell, provide shared complexity
        if self.rnn_cell_cls == ComplexConstrainedRNNCell:
            shared_complexity_l4_exc = SharedComplexity(self.l4_exc_size, self.complexity_size)
            shared_complexity_l4_inh = SharedComplexity(self.l4_inh_size, self.complexity_size)
            shared_complexity_l23_exc = SharedComplexity(self.l23_exc_size, self.complexity_size)
            shared_complexity_l23_inh = SharedComplexity(self.l23_inh_size, self.complexity_size)
        else:
            shared_complexity_l4_exc = None
            shared_complexity_l4_inh = None
            shared_complexity_l23_exc = None
            shared_complexity_l23_inh = None

        # Layer L4 Inh and Exc
        self.L4_Exc = self.rnn_cell_cls(
                input_size + self.l4_inh_size + self.l23_exc_size,
                self.l4_exc_size, 
                self.l4_excitatory_constraint,
                shared_complexity_l4_exc,
            )
        self.L4_Inh = self.rnn_cell_cls(
                input_size + self.l4_exc_size + self.l23_exc_size,
                self.l4_inh_size, 
                self.l4_inhibitory_constraint,
                shared_complexity_l4_inh,
            )
        
        # Layer L23 Inh and Exc
        self.L23_Exc = self.rnn_cell_cls(
                self.l4_exc_size + self.l23_inh_size, 
                self.l23_exc_size, 
                self.l23_excitatory_constraint,
                shared_complexity_l23_exc,
            )
        self.L23_Inh = self.rnn_cell_cls(
                self.l4_exc_size + self.l23_exc_size, 
                self.l23_inh_size, 
                self.l23_inhibitory_constraint,
                shared_complexity_l23_inh,
            )         

    def forward(self, x_on, x_off, h4_exc, h4_inh, h23_exc, h23_inh):
        L4_Inh_outputs = []
        L4_Exc_outputs = []
        L23_Inh_outputs = []
        L23_Exc_outputs = []

        for t in range(x_on.size(1)):
            # if t % 100 == 0:
            #     print(f"Got to iteration: {t}")
            #     torch.cuda.empty_cache()

            # LGN
            input_t = torch.cat((x_on[:, t, :], x_off[:, t, :]), dim=1).to(globals.device0)

            # L4:
            ## L4_Exc
            L4_input_exc = torch.cat((input_t, h4_inh, h23_exc), dim=1).to(globals.device0)
            self.L4_Exc.to(globals.device0)
            h4_exc = self.L4_Exc(L4_input_exc, h4_exc)
            ## L4_Inh
            L4_input_inh = torch.cat((input_t, h4_exc, h23_exc), dim=1).to(globals.device0)
            self.L4_Inh.to(globals.device0)
            h4_inh = self.L4_Inh(L4_input_inh, h4_inh)
            ## Collect L4 outputs
            L4_Exc_outputs.append(h4_exc.unsqueeze(1).cpu())
            L4_Inh_outputs.append(h4_inh.unsqueeze(1).cpu())
            
            # L23:
            ## L23_Exc
            L23_input_exc = torch.cat((h4_exc, h23_inh), dim=1).to(globals.device0)
            self.L23_Exc.to(globals.device0)
            h23_exc = self.L23_Exc(L23_input_exc, h23_exc)
            ## L23_Inh
            L23_input_inh = torch.cat((h4_exc, h23_exc), dim=1).to(globals.device0)
            self.L23_Inh.to(globals.device1)
            h23_inh = self.L23_Inh(L23_input_inh, h23_inh)
            # Collect L23 outputs
            L23_Exc_outputs.append(h23_exc.unsqueeze(1).cpu())
            L23_Inh_outputs.append(h23_inh.unsqueeze(1).cpu())
    
        # Clear caches
        del x_on, x_off, input_t, L4_input_inh, L23_input_exc, L23_input_inh
        torch.cuda.empty_cache()
    
        return {
            'V1_Exc_L4': L4_Exc_outputs,
            'V1_Inh_L4': L4_Inh_outputs,
            'V1_Exc_L23': L23_Exc_outputs, 
            'V1_Inh_L23': L23_Inh_outputs,
        }
    