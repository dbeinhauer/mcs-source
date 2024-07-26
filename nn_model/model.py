import torch
import torch.nn as nn


class CustomRNN(nn.Module):
    def __init__(self, input_layers, hidden_size, output_layers, num_layers=1):
        super(CustomRNN, self).__init__()

        self.rnns = nn.ModuleDict({
            layer: nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            for layer, input_size in input_layers.items()
        })

        self.fcs = nn.ModuleDict({
            layer: nn.Linear(hidden_size * len(input_layers), output_size) 
            for layer, output_size in output_layers.items()
        })

        # self.exc_outputs = nn.ModuleDict({
        #     layer: nn.Linear(hidden_size, output_size)
        #     for layer, output_size in output_layers["exc"].items()
        # })
        
        # self.inh_outputs = nn.ModuleDict({
        #     layer: nn.Linear(hidden_size, output_size)
        #     for layer, output_size in output_layers["inh"].items()
        # })
    
    def forward(self, inputs):
    
        rnn_outputs = {}
        for layer, rnn in self.rnns.items():
            h0 = torch.zeros(rnn.num_layers, inputs[layer].size(0), rnn.hidden_size).to(inputs[layer].device)
            out, _ = rnn(inputs[layer], h0)
            rnn_outputs[layer] = out
            # print(out.shape)

        # Inhibitory/Excitatory outputs        
        outputs = {}
        for output_layer in self.fcs.keys():
            # fc_results = {}
            # for input_layer, output in rnn_outputs.items():
                # if layer in self.exc_outputs:
                #     outputs[layer + "_exc"] = self.exc_outputs[layer](output)
                # if layer in self.inh_outputs:
                #     outputs[layer + "_inh"] = self.inh_outputs[layer](output)
                # fc_results[input_layer] = self.fcs[output_layer](output)
            # print(rnn_outputs.values().)
            combined = torch.cat(list(rnn_outputs.values()), dim=2)
            outputs[output_layer] = self.fcs[output_layer](combined)#torch.cat(fc_results.values(), dim=2)
        
        return outputs

import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, layer_sizes, num_layers=1):
        super(MyRNN, self).__init__()

        self.x_on_size = layer_sizes['X_ON']
        self.x_off_size = layer_sizes['X_OFF']
        self.l4_exc_size = layer_sizes['V1_Exc_L4']
        self.l4_inh_size = layer_sizes['V1_Inh_L4']
        self.l23_exc_size = layer_sizes['V1_Exc_L23']
        self.l23_inh_size = layer_sizes['V1_Inh_L23']

        self.L4 = nn.RNN(self.x_on_size + self.x_off_size, self.l4_exc_size + self.l4_inh_size, num_layers, batch_first=True)
        self.L23 = nn.RNN(self.l4_exc_size + self.l4_inh_size, self.l23_exc_size + self.l23_inh_size, num_layers, batch_first=True)



    def clip_weights(self):
        # # Clip the weights of the inhibitory layer
        # self.L4.weight_hh_l0.clamp_(max=0)
        # self.L4.weight_ih_l0.clamp_(max=0)

        # # Clip the weights of the excitatory layer
        # self.L4.weight_hh_l0.clamp_(min=0)
        # self.L4.weight_ih_l0.clamp_(min=0)

        # # Clip the weights of the inhibitory layer
        # self.L23.weight_hh_l0.clamp_(max=0)
        # self.L23.weight_ih_l0.clamp_(max=0)

        # # Clip the weights of the excitatory layer
        # self.L23.weight_hh_l0.clamp_(min=0)
        # self.L23.weight_ih_l0.clamp_(min=0)

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
