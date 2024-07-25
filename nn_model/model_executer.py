#!/usr/bin/env python3

import numpy as np

import torch.nn
import torch.optim as optim

from nn_model.dataset_loader import DataLoader, SparseSpikeDataset
from nn_model.model import CustomRNN


X_ON_SIZE = 7200
X_OFF_SIZE = 7200
L23_EXC_SIZE = 37500
L23_INH_SIZE = 9375
L4_EXC_SIZE = 37500
L4_INH_SIZE = 9375


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, outputs in train_loader:
            inputs = {layer: input_data.float() for layer, input_data in inputs.items()}
            outputs = {layer: output_data.float() for layer, output_data in outputs.items()}
            
            optimizer.zero_grad()
            predictions = model(inputs)
            
            loss = 0
            for layer, output_data in outputs.items():
                loss += criterion(predictions[layer], output_data)
            
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluation(model, train_loader, criterion):
    model.eval()
    with torch.no_grad():
        for inputs, target_outputs in train_loader:
            inputs = {layer: input_data.float() for layer, input_data in inputs.items()}
            target_outputs = {layer: output_data.float() for layer, output_data in target_outputs.items()}
            
            model_outputs = model(inputs)
            loss = 0
            for layer in target_outputs.keys():
                loss += criterion(model_outputs[layer], target_outputs[layer])
                print(f"layer: {layer}, model output shape: {model_outputs[layer].shape}")
                print(f"layer: {layer}, target output shape: {target_outputs[layer].shape}")


            # loss = criterion(model_outputs, outputs)
            print(f'Test Loss: {loss.item():.4f}')


def main():
    # Define directories and layers
    # base_dir = "dataset/dataset_raw/dataset/artificial_spikes/"
    base_dir = "testing_dataset/"

    inhibitory_layers = ['X_OFF', "V1_Inh_L23", "V1_Inh_L4"]
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
    dataset = SparseSpikeDataset(base_dir, input_layers, output_layers, inhibitory_layers)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    hidden_size = 1

    # model = CustomRNN(input_sizes, hidden_size, output_size)
    model = CustomRNN(input_layers, hidden_size, output_layers)


    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 3
    train(model, train_loader, criterion, optimizer, num_epochs)
    evaluation(model, train_loader, criterion)

if __name__ == "__main__":
    main()