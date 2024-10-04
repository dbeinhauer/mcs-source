import torch
import torch.nn as nn

import globals


class WeightConstraint:
    layer_kwargs = {
        'exc': {"min": 0},
        'inh': {'max': 0},
    }

    def __init__(self, layer_parameters):
        self.layer_parameters = layer_parameters

    def hidden_constraint(self, module, kwargs):
        if hasattr(module, 'weight_hh'):
            module.weight_hh.data = torch.clamp(module.weight_hh.data, **kwargs)

    def input_constraint(self, module):
        if hasattr(module, 'weight_ih'):
            end_index = 0
            for item in self.layer_parameters:
                part_size = item['part_size']
                part_kwargs = WeightConstraint.layer_kwargs[item['part_type']]
                start_index = end_index
                end_index += part_size
                # Apply non-negative constraint to the first part
                module.weight_ih.data[start_index:end_index] = torch.clamp(
                        module.weight_ih.data[start_index:end_index], 
                        **part_kwargs,
                    )


class ExcitatoryWeightConstraint(WeightConstraint):
    def __init__(self, layer_parameters):
        super().__init__(layer_parameters)

    def apply(self, module):
        self.hidden_constraint(module, WeightConstraint.layer_kwargs['exc'])
        self.input_constraint(module)


class InhibitoryWeightConstraint(WeightConstraint):
    def __init__(self, layer_parameters):
        super().__init__(layer_parameters)

    def apply(self, module):
        self.hidden_constraint(module, WeightConstraint.layer_kwargs['inh'])
        self.input_constraint(module)


# Shared complexity module
class SharedComplexity(nn.Module):
    def __init__(self, hidden_size, complexity_size: int=64):
        super(SharedComplexity, self).__init__()
        # Small neural network for shared complexity
        self.complex_layer = nn.Sequential(
            nn.Linear(hidden_size, complexity_size),
            nn.ReLU(),
            nn.Linear(complexity_size, hidden_size)
        )

    def forward(self, hidden):
        return self.complex_layer(hidden)

class ConstrainedRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, weight_constraint, shared_complexity=None):
        super(ConstrainedRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.constraint = weight_constraint

    def forward(self, input, hidden):
        hidden = self.rnn_cell(input, hidden)
        return hidden
    
    def apply_constraints(self):
        self.constraint.apply(self.rnn_cell)

# Child class inheriting from ConstrainedRNNCell
class ComplexConstrainedRNNCell(ConstrainedRNNCell):
    def __init__(self, input_size, hidden_size, weight_constraint, shared_complexity):
        # Inherit from ConstrainedRNNCell
        super(ComplexConstrainedRNNCell, self).__init__(input_size, hidden_size, weight_constraint)
        self.shared_complexity = shared_complexity  # Shared complexity module

    def forward(self, input, hidden):
        # Apply the RNN operation
        hidden = self.rnn_cell(input, hidden)

        # Apply the shared complexity transformation
        complex_hidden = self.shared_complexity(hidden)

        # Combine the original hidden state and the transformed one
        combined_hidden = hidden + complex_hidden

        return combined_hidden

    def apply_constraints(self):
        # Apply weight constraints inherited from ConstrainedRNNCell
        self.constraint.apply(self.rnn_cell)

    

class RNNCellModel(nn.Module):
    # def __init__(self, layer_sizes):
    def __init__(self, layer_sizes, rnn_cell_cls=ConstrainedRNNCell, complexity_size: int=64):
        super(RNNCellModel, self).__init__()

        self.rnn_cell_cls = rnn_cell_cls  # Store the RNN cell class
        self.complexity_size = complexity_size  # For complex RNN cells
        
        self._init_layer_sizes(layer_sizes)
        self._init_weights_constraints()
        self._init_model_architecture()
    
    def _init_layer_sizes(self, layer_sizes):
        self.x_on_size = layer_sizes['X_ON']
        self.x_off_size = layer_sizes['X_OFF']
        self.l4_exc_size = layer_sizes['V1_Exc_L4']
        self.l4_inh_size = layer_sizes['V1_Inh_L4']
        self.l23_exc_size = layer_sizes['V1_Exc_L23']
        self.l23_inh_size = layer_sizes['V1_Inh_L23']

    def _init_weights_constraints(self):
        l4_exc_args = [
            {'part_size': self.x_on_size, 'part_type': 'exc'},
            {'part_size': self.x_off_size, 'part_type': 'exc'},
            {'part_size': self.l4_inh_size, 'part_type': 'inh'},
            {'part_size': self.l23_exc_size, 'part_type': 'exc'},
        ]
        l4_inh_args = [
            {'part_size': self.x_on_size, 'part_type': 'exc'},
            {'part_size': self.x_off_size, 'part_type': 'exc'},
            {'part_size': self.l4_exc_size, 'part_type': 'exc'},
            {'part_size': self.l23_exc_size, 'part_type': 'exc'},
        ]
        l23_exc_args = [
            {'part_size': self.l4_exc_size, 'part_type': 'exc'},
            {'part_size': self.l23_inh_size, 'part_type': 'inh'},
        ]
        l23_inh_args = [
            {'part_size': self.l4_exc_size, 'part_type': 'exc'},
            {'part_size': self.l23_exc_size, 'part_type': 'exc'},
        ]

        self.l4_excitatory_constraint = ExcitatoryWeightConstraint(l4_exc_args)
        self.l4_inhibitory_constraint = InhibitoryWeightConstraint(l4_inh_args)
        self.l23_excitatory_constraint = ExcitatoryWeightConstraint(l23_exc_args)
        self.l23_inhibitory_constraint = InhibitoryWeightConstraint(l23_inh_args)

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
    