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
            fc_results = {}
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
