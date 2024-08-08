import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.sparse import load_npz
from scipy.sparse import csc_matrix

import globals


class SparseSpikeDataset(Dataset):
    def __init__(self, base_dir, input_layers, output_layers):
        self.base_dir = base_dir
        self.input_layers = input_layers
        self.output_layers = output_layers
        # self.inhibitory_layers = inhibitory_layers
        self.experiments = os.listdir(os.path.join(base_dir, 'X_ON'))
        
        self.experiment_batch = None
        self.experiment_offset = 0
    
    def __len__(self):
        return len(self.experiments)


    def load_experiment(self, exp_name, layer):
        print("Got to loading")
        file_path = os.path.join(self.base_dir, layer, exp_name)
        data = load_npz(file_path)
        dense_data = data.toarray().transpose(1, 0)
        print("I loaded the EXPERIMENT")
        return dense_data
    

    # def get_experiment(self, )
    
    def __getitem__(self, idx):
        exp_name = self.experiments[idx]
        
        inputs = {
            'X_ON': torch.zeros((globals.TIME_STEP, globals.X_ON_SIZE)),
            'X_OFF': torch.zeros((globals.TIME_STEP, globals.X_OFF_SIZE)),
        }

        outputs = {
            'V1_Exc_L4': torch.zeros((globals.TIME_STEP, globals.L4_EXC_SIZE)),
            'V1_Inh_L4': torch.zeros((globals.TIME_STEP, globals.L4_INH_SIZE)),
            'V1_Exc_L23': torch.zeros((globals.TIME_STEP, globals.L23_EXC_SIZE)), 
            'V1_Inh_L23': torch.zeros((globals.TIME_STEP, globals.L23_INH_SIZE)), 
        }




        # inputs = {layer: self.load_experiment(exp_name, layer) for layer in self.input_layers}
        # inputs = {layer: torch.tensor(input_data, dtype=torch.float32) for layer, input_data in inputs.items()}
        
        # outputs = {layer: self.load_experiment(exp_name, layer) for layer in self.output_layers}
        # outputs = {layer: torch.tensor(output_data, dtype=torch.float32) for layer, output_data in outputs.items()} 
        
        return inputs, outputs

