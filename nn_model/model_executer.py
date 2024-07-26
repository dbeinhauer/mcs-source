#!/usr/bin/env python3

import numpy as np

import torch.nn
import torch.optim as optim

from nn_model.dataset_loader import DataLoader, SparseSpikeDataset
from nn_model.model import CustomRNN, MyRNN


X_ON_SIZE = 7200
X_OFF_SIZE = 7200
L23_EXC_SIZE = 37500
L23_INH_SIZE = 9375
L4_EXC_SIZE = 37500
L4_INH_SIZE = 9375

# X_ON_SIZE = 72
# X_OFF_SIZE = 72
# L23_EXC_SIZE = 375
# L23_INH_SIZE = 93
# L4_EXC_SIZE = 375
# L4_INH_SIZE = 93

# torch.autograd.set_detect_anomaly(True)

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs = {layer: input_data.float() for layer, input_data in inputs.items()}
            targets = {layer: output_data.float() for layer, output_data in targets.items()}
            
            optimizer.zero_grad()
            predictions = model(inputs['X_ON'], inputs['X_OFF'])
            
            loss = 0
            for layer, target in targets.items():
                loss += criterion(predictions[layer], target)
            
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluation(model, train_loader, criterion):
    model.eval()
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = {layer: input_data.float() for layer, input_data in inputs.items()}
            targets = {layer: output_data.float() for layer, output_data in targets.items()}
            
            predictions = model(inputs['X_ON'], inputs['X_OFF'])
            loss = 0
            for layer in targets.keys():
                loss += criterion(predictions[layer], targets[layer])

            print(f'Test Loss: {loss.item():.4f}')


from torchviz import make_dot

def main():
    # Define directories and layers
    # base_dir = "dataset/dataset_raw/dataset/artificial_spikes/"
    base_dir = "testing_dataset/"

    # inhibitory_layers = ['X_OFF', "V1_Inh_L23", "V1_Inh_L4"]
    layer_sizes = {
        'X_ON': X_ON_SIZE, 
        'X_OFF': X_OFF_SIZE,
        'V1_Exc_L23': L23_EXC_SIZE, 
        'V1_Exc_L4': L4_EXC_SIZE,
        'V1_Inh_L23': L23_INH_SIZE, 
        'V1_Inh_L4': L4_INH_SIZE,
    }
    input_layers = {
        'X_ON': X_ON_SIZE, 
        'X_OFF': X_OFF_SIZE,
    }
    output_layers = {
        'V1_Exc_L23': L23_EXC_SIZE, 
        'V1_Exc_L4': L4_EXC_SIZE,
        'V1_Inh_L23': L23_INH_SIZE, 
        'V1_Inh_L4': L4_INH_SIZE,
    }

    # Create dataset and dataloader
    dataset = SparseSpikeDataset(base_dir, input_layers, output_layers)#, inhibitory_layers)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = MyRNN(layer_sizes).to(device)

    # make_dot(model)

    # inputs = {
    #         'X_ON': torch.zeros((50, 72)),
    #         'X_OFF': torch.zeros((50, 72)),
    #     }
    
    # inputs = (torch.zeros((1, 50, 72)),torch.zeros((1, 50, 72)))
    # yhat = model(inputs['X_ON'], inputs['X_OFF'])
    # yhat = model(inputs[0], inputs[1])
    # make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

    # print(model)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 3
    train(model, train_loader, criterion, optimizer, num_epochs)
    evaluation(model, train_loader, criterion)

if __name__ == "__main__":
    main()