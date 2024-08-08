
import torch
import torch.nn as nn

import globals

# device0 = 'cuda:1'
# device1 = 'cuda:0'
# device1 = 'cuda'
# device0 = device1


# class NonPositiveWeightConstraint:
#     def apply(self, module):
#         if hasattr(module, 'weight_hh'):
#             module.weight_hh.data = torch.clamp(module.weight_hh.data, max=0)
#         if hasattr(module, 'weight_ih'):
#             module.weight_ih.data = torch.clamp(module.weight_ih.data, max=0)

# import torch


class WeightConstraint:
    def __init__(self, second_part, third_part):
        self.second_part = second_part
        self.third_part = third_part

    def hidden_constraint(self, module, kwargs):
        if hasattr(module, 'weight_hh'):
            module.weight_hh.data = torch.clamp(module.weight_hh.data, **kwargs)

    def input_constraint(self, module, first_kwargs, second_kwargs, third_kwargs):
        if hasattr(module, 'weight_ih'):
            # Apply non-negative constraint to the first part
            module.weight_ih.data[:self.second_part] = torch.clamp(
                    module.weight_ih.data[:self.second_part], 
                    **first_kwargs,
                )

            # Apply non-positive constraint to the second part
            module.weight_ih.data[self.second_part:self.third_part] = torch.clamp(
                    module.weight_ih.data[self.second_part:self.third_part], 
                    **second_kwargs,
                )

            # Apply non-negative constraint to the third part
            module.weight_ih.data[self.third_part:] = torch.clamp(
                    module.weight_ih.data[self.third_part:], 
                    **third_kwargs,
                )


class ExcitatoryWeightConstraint(WeightConstraint):
    def __init__(self, second_part, third_part):
        super().__init__(second_part, third_part)

    def apply(self, module):
        self.hidden_constraint(module, {"min": 0})
        self.input_constraint(module, {"min": 0}, {"max": 0}, {"max": 0})

class InhibitoryWeightConstraint(WeightConstraint):
    def __init__(self, second_part, third_part):
        super().__init__(second_part, third_part)

    def apply(self, module):
        self.hidden_constraint(module, {"max": 0})
        self.input_constraint(module, {"min": 0}, {"max": 0}, {"min": 0})


class ConstrainedRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, weight_constraint):
        super(ConstrainedRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.constraint = weight_constraint

    def forward(self, input, hidden):
        # self.constraint(self.rnn_cell)
        hidden = self.rnn_cell(input, hidden)
        return hidden
    
    def apply_constraints(self):
        self.constraint.apply(self.rnn_cell)
    

class RNNCellModel(nn.Module):
    def __init__(self, layer_sizes, num_layers=1):
        super(RNNCellModel, self).__init__()
                
        print("Init model")
        self.x_on_size = layer_sizes['X_ON']
        self.x_off_size = layer_sizes['X_OFF']
        self.l4_exc_size = layer_sizes['V1_Exc_L4']
        self.l4_inh_size = layer_sizes['V1_Inh_L4']
        self.l23_exc_size = layer_sizes['V1_Exc_L23']
        self.l23_inh_size = layer_sizes['V1_Inh_L23']

        input_size = self.x_on_size + self.x_off_size
        l4_size = self.l4_exc_size + self.l4_inh_size
        # l23_size = self.l23_exc_size + self.l23_inh_size
        
        # Define weight constraints
        # self.non_positive_constraint = NonPositiveWeightConstraint()
        # self.non_negative_constraint = NonNegativeWeightConstraint()

        self.l4_excitatory_constraint = ExcitatoryWeightConstraint(self.x_on_size, input_size)
        self.l4_inhibitory_constraint = InhibitoryWeightConstraint(self.x_on_size, input_size)
        self.l23_excitatory_constraint = ExcitatoryWeightConstraint(self.l4_exc_size, l4_size)
        self.l23_inhibitory_constraint = InhibitoryWeightConstraint(self.l4_exc_size, l4_size)

        
        # Layer L4 Inh and Exc
        self.L4_Exc = ConstrainedRNNCell(
                input_size + self.l4_inh_size, 
                self.l4_exc_size, 
                self.l4_excitatory_constraint
            )#.to('cuda:0')
        print("L4_Exc")
        self.L4_Inh = ConstrainedRNNCell(
                input_size + self.l4_exc_size, 
                self.l4_inh_size, 
                self.l4_inhibitory_constraint
            )#.to('cuda:0')
        print("L4_Inh")

        
        # Layer L23 Inh and Exc
        self.L23_Exc = ConstrainedRNNCell(
                l4_size + self.l23_inh_size, 
                self.l23_exc_size, 
                self.l23_excitatory_constraint
            )#.to('cuda:1')
        print("L23_Exc")

        self.L23_Inh = ConstrainedRNNCell(
                l4_size + self.l23_exc_size, 
                self.l23_inh_size, 
                self.l23_inhibitory_constraint
            )#.to('cuda:1')
        print("L23_Inh")



    def forward(self, x_on, x_off, h4_exc, h4_inh, h23_exc, h23_inh):
        L4_Inh_outputs = []
        L4_Exc_outputs = []
        L23_Inh_outputs = []
        L23_Exc_outputs = []
        print("Got to forward")

        for t in range(x_on.size(1)):
            if t % 100 == 0:
                print(f"Got to iteration: {t}")
                torch.cuda.empty_cache()

            # print("input")
            input_t = torch.cat((x_on[:, t, :], x_off[:, t, :]), dim=1).to(globals.device0)
            
            # Layer L1
            # print("L4")
            L4_input_exc = torch.cat((input_t, h4_inh), dim=1).to(globals.device0)
            self.L4_Exc.to(globals.device0)
            h4_exc = self.L4_Exc(L4_input_exc, h4_exc)

            # print("L4 - 2")
            L4_input_inh = torch.cat((input_t, h4_exc), dim=1).to(globals.device0)
            self.L4_Inh.to(globals.device0)
            h4_inh = self.L4_Inh(L4_input_inh, h4_inh)

            # Collect L1 outputs
            L4_Exc_outputs.append(h4_exc.unsqueeze(1).cpu())
            L4_Inh_outputs.append(h4_inh.unsqueeze(1).cpu())
            
            # Combine L1 outputs
            # print("L4 - 3")
            L4_combined = torch.cat((h4_exc, h4_inh), dim=1).to(globals.device1)
            
            # Layer L2
            # print("L23 - 1")
            L23_input_exc = torch.cat((L4_combined, h23_inh), dim=1).to(globals.device1)
            self.L23_Exc.to(globals.device1)
            h23_exc = self.L23_Exc(L23_input_exc, h23_exc)
            # print("L23 - 2")

            L23_input_inh = torch.cat((L4_combined, h23_exc), dim=1).to(globals.device1)
            self.L23_Inh.to(globals.device1)
            h23_inh = self.L23_Inh(L23_input_inh, h23_inh)

            # Collect L2 outputs
            L23_Exc_outputs.append(h23_exc.unsqueeze(1).cpu())
            L23_Inh_outputs.append(h23_inh.unsqueeze(1).cpu())
    
        # del h4_exc, h4_inh, h23_inh, h23_exc
        del x_on, x_off, input_t, L4_input_inh, L4_combined, L23_input_exc, L23_input_inh
        torch.cuda.empty_cache()
    
        return {
            'V1_Exc_L4': L4_Exc_outputs,
            'V1_Inh_L4': L4_Inh_outputs,
            'V1_Exc_L23': L23_Exc_outputs, 
            'V1_Inh_L23': L23_Inh_outputs,
        }
    


class RNNCellFCModel(nn.Module):
    def __init__(self, layer_sizes, hidden_sizes, num_layers=1):
        super(RNNCellFCModel, self).__init__()
                
        self.x_on_size = layer_sizes['X_ON']
        self.x_off_size = layer_sizes['X_OFF']
        self.l4_exc_size = layer_sizes['V1_Exc_L4']
        self.l4_inh_size = layer_sizes['V1_Inh_L4']
        self.l23_exc_size = layer_sizes['V1_Exc_L23']
        self.l23_inh_size = layer_sizes['V1_Inh_L23']

        self.hidden_l4_exc_size = hidden_sizes['V1_Exc_L4']
        self.hidden_l4_inh_size = hidden_sizes['V1_Inh_L4']
        self.hidden_l23_exc_size = hidden_sizes['V1_Exc_L23']
        self.hidden_l23_inh_size = hidden_sizes['V1_Inh_L23']

        input_size = self.x_on_size + self.x_off_size
        l4_hidden_size = self.hidden_l4_exc_size + self.hidden_l4_inh_size
        # l23_size = self.l23_exc_size + self.l23_inh_size
        
        # Define weight constraints
        self.non_positive_constraint = NonPositiveWeightConstraint()
        self.non_negative_constraint = NonNegativeWeightConstraint()
        
        # Layer L4 Inh and Exc
        self.L4_Exc = ConstrainedRNNCell(input_size + self.hidden_l4_inh_size, self.hidden_l4_exc_size, self.non_negative_constraint)
        self.L4_Inh = ConstrainedRNNCell(input_size + self.hidden_l4_exc_size, self.hidden_l4_inh_size, self.non_positive_constraint)
        # print("Got over L4")
        # Layer L23 Inh and Exc
        self.L23_Exc = ConstrainedRNNCell(l4_hidden_size + self.hidden_l23_inh_size, self.hidden_l23_exc_size, self.non_negative_constraint)
        self.L23_Inh = ConstrainedRNNCell(l4_hidden_size + self.hidden_l23_exc_size, self.hidden_l23_inh_size, self.non_positive_constraint)
        # print("Got over L23")
        self.fc_L4_Exc = nn.Linear(self.hidden_l4_exc_size, self.l4_exc_size)
        self.fc_L4_Inh = nn.Linear(self.hidden_l4_inh_size, self.l4_inh_size)
        self.fc_L23_Exc = nn.Linear(self.hidden_l23_exc_size, self.l23_exc_size)
        self.fc_L23_Inh = nn.Linear(self.hidden_l23_inh_size, self.l23_inh_size)
        # print("Got over FC")

    def forward(self, x_on, x_off, h4_exc, h4_inh, h23_exc, h23_inh):
        # L4_Inh_outputs = torch.zeros((50000, 9000))
        # L4_Exc_outputs = torch.zeros((50000, 37000))
        # L23_Inh_outputs = torch.zeros((50000, 9000))
        # L23_Exc_outputs = torch.zeros((50000, 37000))

        L4_Inh_outputs = []
        L4_Exc_outputs = []
        L23_Inh_outputs = []
        L23_Exc_outputs = []

        # print("I get to forward")

        for t in range(x_on.size(1)):
            # if t % 1000 == 0:
                # print(f"Got to iteration: {t}")
                # del L4_input_exc, L4_input_inh, L4_combined, L23_input_exc, L23_input_inh
                # torch.cuda.empty_cache()
            input_t = torch.cat((x_on[:, t, :], x_off[:, t, :]), dim=1)
            
            # Layer L1
            L4_input_exc = torch.cat((input_t, h4_inh), dim=1)
            h4_exc = self.L4_Exc(L4_input_exc, h4_exc)
            # h4_exc = self.L4_Exc(torch.cat((torch.cat((x_on[:, t, :], x_off[:, t, :]), dim=1), h4_inh), dim=1), h4_exc)


            L4_input_inh = torch.cat((input_t, h4_exc), dim=1)
            h4_inh = self.L4_Inh(L4_input_inh, h4_inh)
            # h4_inh = self.L4_Inh(torch.cat((torch.cat((x_on[:, t, :], x_off[:, t, :]), dim=1), h4_exc), dim=1), h4_inh)

            
            # Collect L4 outputs
            L4_Exc_outputs.append(self.fc_L4_Exc(h4_exc).unsqueeze(1).cpu())
            L4_Inh_outputs.append(self.fc_L4_Inh(h4_inh).unsqueeze(1).cpu())
            
            # Combine L4 outputs
            L4_combined = torch.cat((h4_inh, h4_exc), dim=1)
            
            # Layer L23
            L23_input_exc = torch.cat((L4_combined, h23_inh), dim=1)
            h23_exc = self.L23_Exc(L23_input_exc, h23_exc)
            # h23_exc = self.L23_Exc(torch.cat((torch.cat((h4_inh, h4_exc), dim=1), h23_inh), dim=1), h23_exc)

            L23_input_inh = torch.cat((L4_combined, h23_exc), dim=1)
            h23_inh = self.L23_Inh(L23_input_inh, h23_inh)
            # h23_inh = self.L23_Inh(torch.cat((torch.cat((h4_inh, h4_exc), dim=1), h23_exc), dim=1), h23_inh)

            # if t % 100 == 0:
            #     print(f"Got to iteration: {t}")
            #     del L4_input_exc, L4_input_inh, L4_combined, L23_input_exc, L23_input_inh
            #     torch.cuda.empty_cache()

            # Collect L2 outputs
            L23_Exc_outputs.append(self.fc_L23_Exc(h23_exc).unsqueeze(1).cpu())
            L23_Inh_outputs.append(self.fc_L23_Inh(h23_inh).unsqueeze(1).cpu())

        del x_on, x_off, input_t, L4_input_inh, L4_combined, L23_input_exc, L23_input_inh
        torch.cuda.empty_cache()

        # print("I get over the timesteps")

        # L4_Exc_outputs = torch.cat(L4_Exc_outputs, dim=1)        
        # L4_Inh_outputs = torch.cat(L4_Inh_outputs, dim=1)
        # L23_Exc_outputs = torch.cat(L23_Exc_outputs, dim=1)
        # L23_Inh_outputs = torch.cat(L23_Inh_outputs, dim=1)
        
    
        return {
            'V1_Exc_L4': L4_Exc_outputs,
            'V1_Inh_L4': L4_Inh_outputs,
            'V1_Exc_L23': L23_Exc_outputs, 
            'V1_Inh_L23': L23_Inh_outputs,
        }