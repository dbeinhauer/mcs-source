import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# # Simulate data
# num_time_steps = 720
# num_neurons_x_on = 7000
# num_neurons_x_off = 7000
# num_neurons_l23_inh = 7000
# num_neurons_l23_exc = 7000
# num_neurons_l4_inh = 7000
# num_neurons_l4_exc = 7000

# X_ON = np.random.rand(num_time_steps, num_neurons_x_on)
# X_OFF = np.random.rand(num_time_steps, num_neurons_x_off)
# L23_Inh = np.random.rand(num_time_steps, num_neurons_l23_inh)
# L23_Exc = np.random.rand(num_time_steps, num_neurons_l23_exc)
# L4_Inh = np.random.rand(num_time_steps, num_neurons_l4_inh)
# L4_Exc = np.random.rand(num_time_steps, num_neurons_l4_exc)

# # Combine all output layers
# Y_output = np.concatenate((L23_Inh, L23_Exc, L4_Inh, L4_Exc), axis=1)



# import os
# import numpy as np
# from scipy.sparse import load_npz, vstack

# class DatasetLoader():
#     def __init__(self):
#         # Directory containing the sparse array files
#         pass

#     def load_dataset():
#         directory = "dataset_creation/dataset/dataset/spikes/X_OFF"

#         # List to hold the loaded sparse arrays
#         sparse_arrays = []

#         # Load each sparse array from the directory
#         for filename in os.listdir(directory):
#             if filename.endswith('.npz'):
#                 filepath = os.path.join(directory, filename)
#                 sparse_arrays.append(load_npz(filepath))

#         # Concatenate along a new first dimension
#         # Convert each sparse matrix to dense format and stack them
#         dense_arrays = [arr.toarray() for arr in sparse_arrays]
#         stacked_dense = np.stack(dense_arrays, axis=0)

#         print(stacked_dense.shape)

#         # If needed, convert the stacked dense array back to sparse format
#         # stacked_sparse = csr_matrix(stacked_dense)

#         # Save the concatenated sparse array if necessary
#         # save_path = 'path/to/save/concatenated_array.npz'
#         # save_npz(save_path, stacked_sparse)




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
        # return x[np.newaxis, :], y  # Add a new dimension to make it [1, input_size]

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
        dense_data = data.toarray() #if isinstance(data, csc_matrix) else data
        return dense_data
    
    def __getitem__(self, idx):
        exp_name = self.experiments[idx]
        
        # Load input layers
        inputs = [self.load_experiment(exp_name, layer) for layer in self.input_layers]
        inputs = [torch.tensor(input_data, dtype=torch.float32) for input_data in inputs]
        
        # Load output layers
        outputs = [self.load_experiment(exp_name, layer) for layer in self.output_layers]
        for output in outputs:
            print(output.shape)
        outputs = np.concatenate(outputs, axis=0)
        outputs = torch.tensor(outputs, dtype=torch.float32)
        
        return inputs, outputs

# # Define directories and layers
# base_dir = '/path/to/your/data'
# input_layers = ['X_ON', 'X_OFF']
# output_layers = ['L23_Inh', 'L23_Exc', 'L4_Inh', 'L4_Exc']

# # Create dataset and dataloader
# dataset = SparseSpikeDataset(base_dir, input_layers, output_layers)
# train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
