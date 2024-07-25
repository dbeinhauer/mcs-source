import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Custom dataset
class SpikeDataset(Dataset):
    def __init__(self, X_ON, X_OFF, Y):
        self.X_ON = X_ON
        self.X_OFF = X_OFF
        self.X = np.concatenate((X_ON, X_OFF), axis=2)
        self.Y = Y
    
    def __len__(self):
        # return len(self.X)
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        return x, y



import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.sparse import load_npz
from scipy.sparse import csc_matrix

class SparseSpikeDataset(Dataset):
    def __init__(self, base_dir, input_layers, output_layers):
        self.base_dir = base_dir
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.experiments = os.listdir(os.path.join(base_dir, input_layers[0]))
    
    def __len__(self):
        return len(self.experiments)

    
    def load_experiment(self, exp_name, layer):
        file_path = os.path.join(self.base_dir, layer, exp_name)
        data = load_npz(file_path)
        # dense_spikes = sparse_spikes.toarray()
        # data = np.load(file_path)['data']  # Assume data is stored under the key 'data'
        dense_data = data.toarray().transpose(1, 0)
        return dense_data
    
    def __getitem__(self, idx):
        exp_name = self.experiments[idx]
        
        # Load input layers
        inputs = [self.load_experiment(exp_name, layer) for layer in self.input_layers]
        inputs = [torch.tensor(input_data, dtype=torch.float32) for input_data in inputs]
        
        # Load output layers
        outputs = [self.load_experiment(exp_name, layer) for layer in self.output_layers]
        outputs = np.concatenate(outputs, axis=1)
        outputs = torch.tensor(outputs, dtype=torch.float32)
        
        return inputs, outputs

