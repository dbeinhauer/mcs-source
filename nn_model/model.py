import torch
import torch.nn as nn


class CustomRNN(nn.Module):
    def __init__(self, input_sizes, hidden_size, output_size, num_layers=1):
        super(CustomRNN, self).__init__()
        
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

