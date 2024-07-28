#!/usr/bin/env python3

import numpy as np

import torch.nn
import torch.optim as optim

from nn_model.dataset_loader import DataLoader, SparseSpikeDataset
from nn_model.model import OnlyRNN, RNNCellModel


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

            batch_size = inputs['X_ON'].size(0)

            h4_exc = torch.zeros(batch_size, model.l4_exc_size)
            h4_inh = torch.zeros(batch_size, model.l4_inh_size)
            h23_exc = torch.zeros(batch_size, model.l23_exc_size)
            h23_inh = torch.zeros(batch_size, model.l23_inh_size)

            # Forward pass
            # L1_Inh_outputs, L1_Exc_outputs, L2_Inh_outputs, L2_Exc_outputs = model(x, h1_inh, h1_exc, h2_inh, h2_exc)

            predictions = model(inputs['X_ON'], inputs['X_OFF'], h4_exc, h4_inh, h23_exc, h23_inh)
            
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
            
            batch_size = inputs['X_ON'].size(0)

            h4_exc = torch.zeros(batch_size, model.l4_exc_size)
            h4_inh = torch.zeros(batch_size, model.l4_inh_size)
            h23_exc = torch.zeros(batch_size, model.l23_exc_size)
            h23_inh = torch.zeros(batch_size, model.l23_inh_size)

            # predictions = model(inputs['X_ON'], inputs['X_OFF'])
            predictions = model(inputs['X_ON'], inputs['X_OFF'], h4_exc, h4_inh, h23_exc, h23_inh)

            loss = 0
            for layer in targets.keys():
                loss += criterion(predictions[layer], targets[layer])

            print(f'Test Loss: {loss.item():.4f}')


# from torchviz import make_dot

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
    # model = OnlyRNN(layer_sizes).to(device)

    model = RNNCellModel(layer_sizes).to(device)


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


#     # Example usage
# input_dim = 10   # Input feature dimension
# L1_inh_units = 15  # Number of units in L1 Inh
# L1_exc_units = 20  # Number of units in L1 Exc
# L2_inh_units = 25  # Number of units in L2 Inh
# L2_exc_units = 30  # Number of units in L2 Exc

# # Initialize the model
# model = CustomRNNModel(input_dim, L1_inh_units, L1_exc_units, L2_inh_units, L2_exc_units)

# # Example input
# batch_size = 5
# seq_length = 15
# x = torch.randn(batch_size, seq_length, input_dim)

# # Initial hidden states
# h1_inh = torch.zeros(batch_size, L1_inh_units)
# h1_exc = torch.zeros(batch_size, L1_exc_units)
# h2_inh = torch.zeros(batch_size, L2_inh_units)
# h2_exc = torch.zeros(batch_size, L2_exc_units)

# # Forward pass
# L1_Inh_outputs, L1_Exc_outputs, L2_Inh_outputs, L2_Exc_outputs = model(x, h1_inh, h1_exc, h2_inh, h2_exc)

# print("L1_Inh_outputs shape:", L1_Inh_outputs.shape)  # Expected output shape: (batch_size, seq_length, L1_inh_units)
# print("L1_Exc_outputs shape:", L1_Exc_outputs.shape)  # Expected output shape: (batch_size, seq_length, L1_exc_units)
# print("L2_Inh_outputs shape:", L2_Inh_outputs.shape)  # Expected output shape: (batch_size, seq_length, L2_inh_units)
# print("L2_Exc_outputs shape:", L2_Exc_outputs.shape)  # Expected output shape: (batch_size, seq_length, L2_exc_units)
