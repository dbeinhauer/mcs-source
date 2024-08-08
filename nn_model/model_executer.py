#!/usr/bin/env python3

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use the second GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np

import torch.nn
import torch.optim as optim

from dataset_loader import DataLoader, SparseSpikeDataset
from model import RNNCellModel, RNNCellFCModel, ConstrainedRNNCell
import globals

# from model import device0, device1

from torch.amp import autocast, GradScaler

# from dataset_loader import SIZE_MULTIPLIER, X_ON_SIZE, X_OFF_SIZE, L4_EXC_SIZE, L4_INH_SIZE, L23_EXC_SIZE, L23_INH_SIZE

# import gc

from tqdm import tqdm

# device0 = 'cuda:1'
# device1 = 'cuda:0'
# device1 = 'cuda'
# device0 = device1

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    # Initialize GradScaler for mixed precision training
    # scaler = GradScaler()
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            if i > 5:
                break
            print("I get to training")
            inputs = {layer: input_data.float().to(globals.device0) for layer, input_data in inputs.items()}
            print("I get to inputs")
            targets = {layer: output_data.float() for layer, output_data in targets.items()}
            print("I get to outputs")

            optimizer.zero_grad()

            batch_size = inputs['X_ON'].size(0)

            h4_exc = torch.zeros(batch_size, model.l4_exc_size).to(globals.device0)#.cuda()
            h4_inh = torch.zeros(batch_size, model.l4_inh_size).to(globals.device0)#.cuda()
            h23_exc = torch.zeros(batch_size, model.l23_exc_size).to(globals.device1)#.cuda()
            h23_inh = torch.zeros(batch_size, model.l23_inh_size).to(globals.device1)#.cuda()

            loss = 0

            with autocast(device_type="cuda", dtype=torch.float16):
                predictions = model(inputs['X_ON'], inputs['X_OFF'], h4_exc, h4_inh, h23_exc, h23_inh)            
                print("Predictions done")
                del inputs, h4_exc, h4_inh, h23_inh, h23_exc
                torch.cuda.empty_cache()


                # loss = 0
                for layer, target in targets.items():
                    loss += criterion(torch.cat(predictions[layer], dim=1).float().cpu(), target.float())
                print("Loss done")


            del targets, predictions
            torch.cuda.empty_cache()


            loss.float().backward()
            print("Backward done")
            optimizer.step()
            print("Optimizer step done")

            # Apply constraints to all constrained RNN cells
            for module in model.modules():
                if isinstance(module, ConstrainedRNNCell):
                    module.apply_constraints()

            torch.cuda.empty_cache()

        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluation(model, train_loader, criterion):
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(train_loader):
            inputs = {layer: input_data.float().cuda() for layer, input_data in inputs.items()}
            targets = {layer: output_data.float() for layer, output_data in targets.items()}
            
            batch_size = inputs['X_ON'].size(0)

            h4_exc = torch.zeros(batch_size, model.l4_exc_size).cuda()
            h4_inh = torch.zeros(batch_size, model.l4_inh_size).cuda()
            h23_exc = torch.zeros(batch_size, model.l23_exc_size).cuda()
            h23_inh = torch.zeros(batch_size, model.l23_inh_size).cuda()

            predictions = model(inputs['X_ON'], inputs['X_OFF'], h4_exc, h4_inh, h23_exc, h23_inh)

            loss = 0
            for layer in targets.keys():
                loss += criterion(torch.cat(predictions[layer], dim=1).cpu(), targets[layer])

            del inputs, targets, predictions, h4_exc, h4_inh, h23_inh, h23_exc


            print(f'Test Loss: {loss.item():.4f}')


# from torchviz import make_dot

def main():
    # Define directories and layers
    # base_dir = "dataset/dataset_raw/dataset/artificial_spikes/"
    base_dir = "testing_dataset/size_5"
    base_dir = "/home/beinhaud/diplomka/mcs-source/dataset/compressed_data/size_5"
    base_dir = "/home/beinhaud/diplomka/mcs-source/testing_dataset/test_out"

    # inhibitory_layers = ['X_OFF', "V1_Inh_L23", "V1_Inh_L4"]
    layer_sizes = {
        'X_ON': globals.X_ON_SIZE,
        'X_OFF': globals.X_OFF_SIZE,
        'V1_Exc_L4': globals.L4_EXC_SIZE,
        'V1_Inh_L4': globals.L4_INH_SIZE,
        'V1_Exc_L23': globals.L23_EXC_SIZE, 
        'V1_Inh_L23': globals.L23_INH_SIZE, 
    }
    input_layers = {
        'X_ON': globals.X_ON_SIZE, 
        'X_OFF': globals.X_OFF_SIZE,
    }
    output_layers = {
        'V1_Exc_L4': globals.L4_EXC_SIZE,
        'V1_Inh_L4': globals.L4_INH_SIZE,
        'V1_Exc_L23': globals.L23_EXC_SIZE, 
        'V1_Inh_L23': globals.L23_INH_SIZE, 
    }

    # Create dataset and dataloader
    dataset = SparseSpikeDataset(base_dir, input_layers, output_layers)#, inhibitory_layers)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    model = RNNCellModel(layer_sizes).to(globals.device1).half()
    # model = RNNCellFCModel(layer_sizes, hidden_sizes).to(device)       

    print("Criterion")
    criterion = torch.nn.MSELoss()
    print("optimizer")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 3
    print("training")
    train(model, train_loader, criterion, optimizer, num_epochs)
    evaluation(model, train_loader, criterion)

if __name__ == "__main__":
    main()
