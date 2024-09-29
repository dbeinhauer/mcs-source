import os
import pickle
import re

import numpy as np
from scipy.sparse import load_npz
import torch
from torch.utils.data import Dataset

import globals


import os
from collections import defaultdict

class SparseSpikeDataset(Dataset):
    def __init__(
            self, 
            spikes_dir, 
            input_layers, 
            output_layers,
            is_test: bool=False,
            model_subset_path=None, 
            train_test_path=None, 
            include_experiments=True,
        ):
        self.spikes_dir = spikes_dir
        self.input_layers = input_layers
        self.output_layers = output_layers

        self.is_test = is_test

        self.model_subset_indices = self._load_model_subset_indices(model_subset_path)       
        # self.experiments = self._load_experiment_paths(train_test_path, include_experiments)
        
        self.experiments = self._load_all_spikes_filenames()

        # print(self.experiments)

        # self.experiment_batch = None
        # self.experiment_offset = 0
    
    def __len__(self):
        return len(self.experiments)

    def _load_all_spikes_filenames(self, subdir="X_ON"):
        # Get all filenames in the specified subdirectory
        all_files = os.listdir(os.path.join(self.spikes_dir, subdir))

        # Dictionary to group files by their spike_id
        coupled_files = defaultdict(list)
        
        for file in all_files:
            if file.endswith(".npz") and "_summed" in file:
                # Split the filename by underscores
                file_without_ext = file.replace('.npz', '')
                parts = file_without_ext.split('_')
                
                # Ensure the filename ends with 'summed' and spike_id is the second-to-last part
                if len(parts) >= 3 and parts[-2].isdigit():
                    spike_id = parts[-2]  # Extract spike_id as the second-to-last part
                    
                    # Add the file to the list of files with the same spike_id
                    coupled_files[spike_id].append(file)

        # Convert the dictionary values (lists of files) to a list of lists
        return list(coupled_files.values())

    
    # def _load_all_spikes_filenames(self, subdir="X_ON"):
        
    #     return os.listdir(os.path.join(self.spikes_dir, subdir))

    def _load_train_test_indices(self, path):
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
            
    # def _select_experiments_subset(self, indices_path, include=True):
    #     subset = []
    #     # Iterate through each filename
    #     train_test_indices = self._load_train_test_indices(indices_path)
    #     for filename in self._load_all_spikes_filenames():
    #         # Create the regex pattern for each number in the numpy array
    #         match_found = any(re.match(rf"^spikes_{num}[0-9][0-9]_summed\.npz$", filename) for num in train_test_indices)
            
    #         # Check if the current filename should be appended based on the include flag
    #         if (include and match_found) or (not include and not match_found):
    #             subset.append(filename)

    #     return subset
    
    # def _load_experiment_paths(self, train_test_path, include_experiments):
    #     # if train_test_path is None:
    #     #     # Load all experiments.
    #     #     return self._load_all_spikes_filenames()

    #     if self.is_test:
    #         # TODO: add trials coupling

    #     # Not testing -> return all filenames
    #     return self._load_all_spikes_filenames()

        
        # Load only subset of experiments.
        # return self._select_experiments_subset(train_test_path, include_experiments)
    
    def _load_model_subset_indices(self, model_subset_path):
        if model_subset_path is not None:
            # Load indices of subset.
            with open(model_subset_path, 'rb') as pickle_file:
                return pickle.load(pickle_file)
        
        return None

    def _load_experiment(self, exp_names, layer):
        # List to hold the loaded data from each file
        all_data = []
        
        for exp_name in exp_names:
            # Construct the full file path for each file in the list
            file_path = os.path.join(self.spikes_dir, layer, exp_name)
            
            # Load the data from the file
            data = load_npz(file_path)
            
            # Convert to dense format
            dense_data = data.toarray()
            
            # # Step for training set (we want to have the trials dimension too)
            # if dense_data.ndim == 2:
            
            # Create trials dimension.
            dense_data = np.expand_dims(dense_data, axis=0)
            
            # Transpose to have the shape: (time, neurons)
            dense_data = dense_data.transpose(0, 2, 1)
            
            # If there is a model subset, apply it
            if self.model_subset_indices is not None:
                # Subset creation
                dense_data = dense_data[:, :, self.model_subset_indices[layer]]

            # Append the dense_data to the list
            all_data.append(dense_data)
        
        # Concatenate all loaded data along the trials dimension (axis 0)
        combined_data = np.concatenate(all_data, axis=0)
        
        # Return the combined data
        return combined_data



    # def _load_experiment(self, exp_name, layer):
    #     # print("Got to loading")
    #     file_path = os.path.join(self.spikes_dir, layer, exp_name)
    #     data = load_npz(file_path)

    #     dense_data = data.toarray()
    #     # Step for training set (we want to have the trials dimension too)
    #     if dense_data.ndim == 2:
    #         dense_data = np.expand_dims(dense_data, axis=0)
        
    #     # We want to have the shape: (trials, time, neurons)
    #     dense_data = dense_data.transpose(0, 2, 1)
    #     if self.model_subset_indices is not None:
    #         # Subset creation.
    #         dense_data = dense_data[:, :, self.model_subset_indices[layer]]

    #     # print("I loaded the EXPERIMENT")
    #     return dense_data

    # def get_experiment(self, )
    
    def __getitem__(self, idx):
        # print(f"IDX: {idx}")
        exp_name = self.experiments[idx]
        
        # inputs = {
        #     'X_ON': torch.zeros((1, globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.X_ON_SIZE)),
        #     'X_OFF': torch.zeros((1, globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.X_OFF_SIZE)),
        # }

        # outputs = {
        #     'V1_Exc_L4': torch.zeros((1, globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.L4_EXC_SIZE)),
        #     'V1_Inh_L4': torch.zeros((1, globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.L4_INH_SIZE)),
        #     'V1_Exc_L23': torch.zeros((1, globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.L23_EXC_SIZE)), 
        #     'V1_Inh_L23': torch.zeros((1, globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.L23_INH_SIZE)), 
        # }


        inputs = {layer: self._load_experiment(exp_name, layer) for layer in self.input_layers}
        inputs = {layer: torch.tensor(input_data, dtype=torch.half) for layer, input_data in inputs.items()}
        
        outputs = {layer: self._load_experiment(exp_name, layer) for layer in self.output_layers}
        outputs = {layer: torch.tensor(output_data, dtype=torch.half) for layer, output_data in outputs.items()} 
        
        return inputs, outputs


def custom_collate_fn(batch):
    # Find the maximum size in the second dimension
    # for item in batch:
    #     # print(next(iter(item[0].values())))
    #     # print(item)
    max_size = max([next(iter(item[0].values())).size(1) for item in batch])
    
    padded_batch_0 = {}
    padded_batch_1 = {}

    # for input, output in batch:

    num_dicts = len(batch[0])
    
    # Initialize a list to hold the result dictionaries
    result = [{} for _ in range(num_dicts)]
    
    # Loop over each index in the tuples
    for i in range(num_dicts):
        # Get all dictionaries at the current index across all tuples
        dicts_at_index = [tup[i] for tup in batch]
        
        # Get the keys (assuming all dictionaries have the same keys)
        keys = dicts_at_index[0].keys()
        
        # For each key, concatenate the tensors from all dictionaries at the current index
        for key in keys:
            # padding = (0, max_size - next(iter(item[0].values())).size(1))
            # padded_item_0 = {
            #     layer: torch.nn.functional.pad(value, padding, "constant", 0) 
            #     for layer, value in item[0].items()
            # }

            # Collect all tensors associated with the current key
            tensors_to_concat = [
                torch.nn.functional.pad(
                    d[key],#.transpose(0, 2, 1), 
                    (0, 0, 0, max_size - d[key].size(1)), 
                    "constant", 
                    0,
                )#.transpose(0, 2, 1)
                for d in dicts_at_index
            ]

            # tensors_to_concat = [d[key] for d in dicts_at_index]
            
            # Concatenate tensors along a new dimension (e.g., dimension 0)
            result[i][key] = torch.stack(tensors_to_concat, dim=0)
    
    # Convert the list of dictionaries into a tuple
    return tuple(result)



    # Pad each tensor in the batch to have the same second dimension size
    padded_batch_0 = {}
    padded_batch_1 = {}
    for item in batch:
        # print(next(iter(item[0].values())).size(1))
        # Pad tensor along the second dimension
        padding = (0, max_size - next(iter(item[0].values())).size(1))
        padded_item_0 = {
            layer: torch.nn.functional.pad(value, padding, "constant", 0) 
            for layer, value in item[0].items()
        }
        padded_item_1 = {
            layer: torch.nn.functional.pad(value, padding, "constant", 0) 
            for layer, value in item[1].items()
        }
        # padded_item = torch.nn.functional.pad(item, padding, "constant", 0)
        # padded_batch.append((padded_item_0, padded_item_1))
        padded_batch_0.append(padded_item_0)
        padded_batch_1.append(padded_item_1)


    # print(batch[0])
    # print("_-----------------------------------------")
    # print(padded_batch[0])
    # print(batch)
    
    # Stack all padded tensors to form a batch
    # return torch.stack(padded_batch)
    return padded_batch_0, padded_batch_1
