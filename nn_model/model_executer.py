#!/usr/bin/env python3

import numpy as np

import torch.nn
import torch.optim as optim

from nn_model.dataset_loader import SpikeDataset, DataLoader, SparseSpikeDataset
from nn_model.model import CustomRNN

def main():
    # Define directories and layers
    # base_dir = "dataset/dataset_raw/dataset/artificial_spikes/"
    base_dir = "dataset/compresed_datasets/test_20/"
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
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    input_sizes = [X_ON_SIZE, X_OFF_SIZE] 
    hidden_size = 128
    output_size = L23_EXC_SIZE + L23_INH_SIZE + L4_EXC_SIZE + L4_INH_SIZE

    model = CustomRNN(input_sizes, hidden_size, output_size)

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



if __name__ == "__main__":
    main()
