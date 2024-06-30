#!/usr/bin/env python3

import numpy as np

import torch.nn
import torch.optim as optim

from nn_model.dataset_loader import SpikeDataset, DataLoader, SparseSpikeDataset
from nn_model.model import CustomRNN, CustomRNN_new


# def main():

#     # Simulate data
#     num_time_steps = 720
#     num_neurons_x_on = 13
#     num_neurons_x_off = 25
#     num_neurons_l23_inh = 70
#     num_neurons_l23_exc = 100
#     num_neurons_l4_inh = 30
#     num_neurons_l4_exc = 40

#     num_experiments = 120

#     X_ON = np.random.rand(num_experiments, num_time_steps, num_neurons_x_on)
#     X_OFF = np.random.rand(num_experiments, num_time_steps, num_neurons_x_off)
#     L23_Inh = np.random.rand(num_experiments, num_time_steps, num_neurons_l23_inh)
#     L23_Exc = np.random.rand(num_experiments, num_time_steps, num_neurons_l23_exc)
#     L4_Inh = np.random.rand(num_experiments, num_time_steps, num_neurons_l4_inh)
#     L4_Exc = np.random.rand(num_experiments, num_time_steps, num_neurons_l4_exc)

#     # Combine all output layers
#     Y_output = np.concatenate((L23_Inh, L23_Exc, L4_Inh, L4_Exc), axis=2)


#     # Create dataset and dataloaders
#     dataset = SpikeDataset(X_ON, X_OFF, Y_output)
#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True)



#     input_size = num_neurons_x_on + num_neurons_x_off
#     hidden_size = 128
#     layer_sizes = [num_neurons_l23_inh, num_neurons_l23_exc, num_neurons_l4_inh, num_neurons_l4_exc]

#     model = CustomRNN(input_size, hidden_size, layer_sizes)

#     # Loss and optimizer
#     criterion = torch.nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Training loop
#     num_epochs = 20

#     for epoch in range(num_epochs):
#         for X_batch, Y_batch in train_loader:
#             X_batch, Y_batch = X_batch.float(), Y_batch.float()
            
#             # Forward pass
#             outputs = model(X_batch)
#             # print(outputs.shape)
#             # print(Y_batch.shape)
#             loss = criterion(outputs, Y_batch)
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



#     # Evaluation
#     model.eval()
#     with torch.no_grad():
#         for X_batch, Y_batch in train_loader:
#             X_batch, Y_batch = X_batch.float(), Y_batch.float()
#             outputs = model(X_batch)
#             loss = criterion(outputs, Y_batch)
#             print(f'Test Loss: {loss.item():.4f}')


def main():
    # Define directories and layers
    base_dir = "dataset/dataset_raw/dataset/artificial_spikes/"
    input_layers = ['X_ON', 'X_OFF']
    output_layers = ['V1_Exc_L23', 'V1_Inh_L23', 'V1_Exc_L4', 'V1_Inh_L4']

    X_ON_SIZE = 7200
    X_OFF_SIZE = 7200
    L23_EXC_SIZE = 37500
    L23_INH_SIZE = 9375
    L4_EXC_SIZE = 37500
    L4_INH_SIZE = 9375
    

    # Create dataset and dataloader
    dataset = SparseSpikeDataset(base_dir, input_layers, output_layers)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    input_sizes = [X_ON_SIZE, X_OFF_SIZE] 
    hidden_size = 128
    output_size = L23_EXC_SIZE + L23_INH_SIZE + L4_EXC_SIZE + L4_INH_SIZE

    model = CustomRNN_new(input_sizes, hidden_size, output_size)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20

    for epoch in range(num_epochs):
        for inputs, Y_batch in train_loader:
            inputs = [input_data.float() for input_data in inputs]
            Y_batch = Y_batch.float()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, Y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        for inputs, Y_batch in train_loader:
            inputs = [input_data.float() for input_data in inputs]
            Y_batch = Y_batch.float()
            outputs = model(inputs)
            loss = criterion(outputs, Y_batch)
            print(f'Test Loss: {loss.item():.4f}')



import os
import numpy as np
from scipy.sparse import load_npz, vstack


if __name__ == "__main__":
    main()
    # directory = "dataset_creation/dataset/dataset/spikes/V1_Exc_L23"

    # # List to hold the loaded sparse arrays
    # sparse_arrays = []

    # # Load each sparse array from the directory
    # for filename in os.listdir(directory):
    #     if filename.endswith('.npz'):
    #         filepath = os.path.join(directory, filename)
    #         sparse_arrays.append(load_npz(filepath))

    # # Concatenate along a new first dimension
    # # Convert each sparse matrix to dense format and stack them
    # dense_arrays = [arr.toarray() for arr in sparse_arrays]
    # stacked_dense = np.stack(dense_arrays, axis=0)

    # print(stacked_dense.shape)
