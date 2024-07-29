import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.sparse import load_npz
from scipy.sparse import csc_matrix

class SparseSpikeDataset(Dataset):
    def __init__(self, base_dir, input_layers, output_layers):#, inhibitory_layers):
        self.base_dir = base_dir
        self.input_layers = input_layers
        self.output_layers = output_layers
        # self.inhibitory_layers = inhibitory_layers
        self.experiments = os.listdir(os.path.join(base_dir, 'X_ON'))
    
    def __len__(self):
        return len(self.experiments)

    
    def load_experiment(self, exp_name, layer):
        file_path = os.path.join(self.base_dir, layer, exp_name)
        data = load_npz(file_path)
        dense_data = data.toarray().transpose(1, 0)
        # print("I loaded the EXPERIMENT")
        return dense_data
    
    def __getitem__(self, idx):
        exp_name = self.experiments[idx]
        
        # inputs = {
        #     'X_ON': torch.zeros((12000, 7000)),
        #     'X_OFF': torch.zeros((12000, 7000)),
        # }

        # outputs = {
        #     'V1_Exc_L4': torch.zeros((12000, 37000)),
        #     'V1_Inh_L4': torch.zeros((12000, 9000)),
        #     'V1_Exc_L23': torch.zeros((12000, 37000)), 
        #     'V1_Inh_L23': torch.zeros((12000, 9000)), 
        # }

        inputs = {layer: self.load_experiment(exp_name, layer) for layer in self.input_layers}
        inputs = {layer: torch.tensor(input_data, dtype=torch.float32) for layer, input_data in inputs.items()}
        
        outputs = {layer: self.load_experiment(exp_name, layer) for layer in self.output_layers}
        outputs = {layer: torch.tensor(output_data, dtype=torch.float32) for layer, output_data in outputs.items()} 
        
        return inputs, outputs

