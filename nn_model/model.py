import torch
import torch.nn as nn

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, layer_sizes, num_layers=1):
        super(CustomRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        self.fc_layers = nn.ModuleList()
        for size in layer_sizes:
            self.fc_layers.append(nn.Linear(hidden_size, size))
    
    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        # out = out[:, -1, :]  # Use the last time step output

        # out = self.fc(out)
        
        outputs = []
        for fc in self.fc_layers:
            # out_control = fc(out)
            # print(out_control.shape)
            outputs.append(fc(out))
            # outputs.append(out_control)

        
        return torch.cat(outputs, dim=2)

# input_size = num_neurons_x_on + num_neurons_x_off
# hidden_size = 128
# layer_sizes = [num_neurons_l23_inh, num_neurons_l23_exc, num_neurons_l4_inh, num_neurons_l4_exc]

# model = CustomRNN(input_size, hidden_size, layer_sizes)


import torch.nn as nn

class CustomRNN_new(nn.Module):
    def __init__(self, input_sizes, hidden_size, output_size, num_layers=1):
        super(CustomRNN_new, self).__init__()
        
        # Define separate RNNs for each input layer
        self.rnns = nn.ModuleList([nn.RNN(input_size, hidden_size, num_layers, batch_first=True) for input_size in input_sizes])
        self.fc = nn.Linear(hidden_size * len(input_sizes), output_size)
    
    def forward(self, inputs):
        # inputs is a list of tensors, one for each input layer
        rnn_outputs = []
        for i, rnn in enumerate(self.rnns):
            h0 = torch.zeros(rnn.num_layers, inputs[i].size(0), rnn.hidden_size).to(inputs[i].device)
            out, _ = rnn(inputs[i], h0)
            rnn_outputs.append(out)  # Use all time steps output
        
        combined = torch.cat(rnn_outputs, dim=2)
        out = self.fc(combined)
        return out

