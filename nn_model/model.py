
import torch
import torch.nn as nn

class OnlyRNN(nn.Module):
    def __init__(self, layer_sizes, num_layers=1):
        super(OnlyRNN, self).__init__()

        self.x_on_size = layer_sizes['X_ON']
        self.x_off_size = layer_sizes['X_OFF']
        self.l4_exc_size = layer_sizes['V1_Exc_L4']
        self.l4_inh_size = layer_sizes['V1_Inh_L4']
        self.l23_exc_size = layer_sizes['V1_Exc_L23']
        self.l23_inh_size = layer_sizes['V1_Inh_L23']

        self.L4 = nn.RNN(self.x_on_size + self.x_off_size, self.l4_exc_size + self.l4_inh_size, num_layers, batch_first=True)
        self.L23 = nn.RNN(self.l4_exc_size + self.l4_inh_size, self.l23_exc_size + self.l23_inh_size, num_layers, batch_first=True)

    def clip_weights(self):

        # Clone the weights of the L4 layer
        clipped_L4_weight_hh = self.L4.weight_hh_l0.clone()
        clipped_L4_weight_ih = self.L4.weight_ih_l0.clone()

        # Clip the weights of the L4 layer
        clipped_L4_weight_hh[:self.l4_exc_size].clamp_(min=0)  # Clip weights till self.l4_exc_size to minimum 0
        clipped_L4_weight_ih[:self.l4_exc_size].clamp_(min=0)  # Clip weights till self.l4_exc_size to minimum 0
        clipped_L4_weight_hh[self.l4_exc_size:].clamp_(max=0)  # Clip weights after self.l4_exc_size to maximum 0
        clipped_L4_weight_ih[self.l4_exc_size:].clamp_(max=0)  # Clip weights after self.l4_exc_size to maximum 0

        # Copy the clipped weights back to the L4 layer
        self.L4.weight_hh_l0.data.copy_(clipped_L4_weight_hh)
        self.L4.weight_ih_l0.data.copy_(clipped_L4_weight_ih)

        # Clone the weights of the L23 layer
        clipped_L23_weight_hh = self.L23.weight_hh_l0.clone()
        clipped_L23_weight_ih = self.L23.weight_ih_l0.clone()

        # Clip the weights of the L23 layer
        clipped_L23_weight_hh[:self.l23_exc_size].clamp_(min=0)  # Clip weights till self.l23_exc_size to minimum 0
        clipped_L23_weight_ih[:self.l23_exc_size].clamp_(min=0)  # Clip weights till self.l23_exc_size to minimum 0
        clipped_L23_weight_hh[self.l23_exc_size:].clamp_(max=0)  # Clip weights after self.l23_exc_size to maximum 0
        clipped_L23_weight_ih[self.l23_exc_size:].clamp_(max=0)  # Clip weights after self.l23_exc_size to maximum 0

        # Copy the clipped weights back to the L23 layer
        self.L23.weight_hh_l0.data.copy_(clipped_L23_weight_hh)
        self.L23.weight_ih_l0.data.copy_(clipped_L23_weight_ih)


    def forward(self, x_ON, x_OFF):

        L4_output, _ = self.L4(torch.cat((x_ON, x_OFF), dim=2))
        L23_output, _ = self.L23(L4_output)

        L4_Exc_output = L4_output[:, :, :self.l4_exc_size]
        L4_Inh_output = L4_output[:, :, self.l4_exc_size:]

        L23_Exc_output = L23_output[:, :, :self.l23_exc_size]
        L23_Inh_output = L23_output[:, :, self.l23_exc_size:]

        # Clip the weights after the connections to L23 layers
        self.clip_weights()

        return {
            'V1_Exc_L4': L4_Exc_output,
            'V1_Inh_L4': L4_Inh_output,
            'V1_Exc_L23': L23_Exc_output, 
            'V1_Inh_L23': L23_Inh_output,
        }



import torch
import torch.nn as nn

class NonPositiveWeightConstraint:
    def __call__(self, module):
        if hasattr(module, 'weight_hh'):
            module.weight_hh.data = -torch.abs(module.weight_hh.data)
        if hasattr(module, 'weight_ih'):
            module.weight_ih.data = -torch.abs(module.weight_ih.data)

class NonNegativeWeightConstraint:
    def __call__(self, module):
        if hasattr(module, 'weight_hh'):
            module.weight_hh.data = torch.abs(module.weight_hh.data)
        if hasattr(module, 'weight_ih'):
            module.weight_ih.data = torch.abs(module.weight_ih.data)

class ConstrainedRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, weight_constraint):
        super(ConstrainedRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.constraint = weight_constraint

    def forward(self, input, hidden):
        self.constraint(self.rnn_cell)
        hidden = self.rnn_cell(input, hidden)
        return hidden

class RNNCellModel(nn.Module):
    # def __init__(self, input_dim, L1_inh_units, L1_exc_units, L2_inh_units, L2_exc_units):
    def __init__(self, layer_sizes, num_layers=1):
        super(RNNCellModel, self).__init__()
                
        self.x_on_size = layer_sizes['X_ON']
        self.x_off_size = layer_sizes['X_OFF']
        self.l4_exc_size = layer_sizes['V1_Exc_L4']
        self.l4_inh_size = layer_sizes['V1_Inh_L4']
        self.l23_exc_size = layer_sizes['V1_Exc_L23']
        self.l23_inh_size = layer_sizes['V1_Inh_L23']

        input_size = self.x_on_size + self.x_off_size
        l4_size = self.l4_exc_size + self.l4_inh_size
        l23_size = self.l23_exc_size + self.l23_inh_size
        
        # Define weight constraints
        self.non_positive_constraint = NonPositiveWeightConstraint()
        self.non_negative_constraint = NonNegativeWeightConstraint()
        
        # Layer L4 Inh and Exc
        self.L4_Exc = ConstrainedRNNCell(input_size + self.l4_inh_size, self.l4_exc_size, self.non_negative_constraint)
        self.L4_Inh = ConstrainedRNNCell(input_size + self.l4_exc_size, self.l4_inh_size, self.non_positive_constraint)
        
        # Layer L23 Inh and Exc
        self.L23_Exc = ConstrainedRNNCell(l4_size + self.l23_inh_size, self.l23_exc_size, self.non_negative_constraint)
        self.L23_Inh = ConstrainedRNNCell(l4_size + self.l23_exc_size, self.l23_inh_size, self.non_positive_constraint)


    def forward(self, x_on, x_off, h4_exc, h4_inh, h23_exc, h23_inh):
        L4_Inh_outputs = []
        L4_Exc_outputs = []
        L23_Inh_outputs = []
        L23_Exc_outputs = []

        for t in range(x_on.size(1)):
            input_t = torch.cat((x_on[:, t, :], x_off[:, t, :]), dim=1)
            
            # Layer L1
            L4_input_exc = torch.cat((input_t, h4_inh), dim=1)
            h4_exc = self.L4_Exc(L4_input_exc, h4_exc)

            L4_input_inh = torch.cat((input_t, h4_exc), dim=1)
            h4_inh = self.L4_Inh(L4_input_inh, h4_inh)
            
            # Collect L1 outputs
            L4_Exc_outputs.append(h4_exc.unsqueeze(1))
            L4_Inh_outputs.append(h4_inh.unsqueeze(1))
            
            # Combine L1 outputs
            L4_combined = torch.cat((h4_inh, h4_exc), dim=1)
            
            # Layer L2
            L23_input_exc = torch.cat((L4_combined, h23_inh), dim=1)
            h23_exc = self.L23_Exc(L23_input_exc, h23_exc)

            L23_input_inh = torch.cat((L4_combined, h23_exc), dim=1)
            h23_inh = self.L23_Inh(L23_input_inh, h23_inh)
            
            # Collect L2 outputs
            L23_Exc_outputs.append(h23_exc.unsqueeze(1))
            L23_Inh_outputs.append(h23_inh.unsqueeze(1))

        L4_Exc_outputs = torch.cat(L4_Exc_outputs, dim=1)        
        L4_Inh_outputs = torch.cat(L4_Inh_outputs, dim=1)
        L23_Exc_outputs = torch.cat(L23_Exc_outputs, dim=1)
        L23_Inh_outputs = torch.cat(L23_Inh_outputs, dim=1)
        
        # return L4_Exc_outputs, L4_Inh_outputs, L23_Exc_outputs, L23_Inh_outputs
    
        return {
            'V1_Exc_L4': L4_Exc_outputs,
            'V1_Inh_L4': L4_Inh_outputs,
            'V1_Exc_L23': L23_Exc_outputs, 
            'V1_Inh_L23': L23_Inh_outputs,
        }