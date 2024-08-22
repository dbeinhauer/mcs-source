import os
import pickle
import re

from scipy.sparse import load_npz
import torch
from torch.utils.data import Dataset

import globals


class SparseSpikeDataset(Dataset):
    def __init__(
            self, 
            spikes_dir, 
            input_layers, 
            output_layers,
            model_subset_path=None, 
            train_test_path=None, 
            include_experiments=True,
        ):
        self.spikes_dir = spikes_dir
        self.input_layers = input_layers
        self.output_layers = output_layers

        self.model_subset_indices = self._load_model_subset_indices(model_subset_path)        
        self.experiments = self._load_experiment_paths(train_test_path, include_experiments)
        
        self.experiment_batch = None
        self.experiment_offset = 0
    
    def __len__(self):
        return len(self.experiments)
    
    def _load_all_spikes_filenames(self, subdir="X_ON"):
        return os.listdir(os.path.join(self.spikes_dir, subdir))

    def _load_train_test_indices(self, path):
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
            
    def _select_experiments_subset(self, indices_path, include=True):
        subset = []
        # Iterate through each filename
        train_test_indices = self._load_train_test_indices(indices_path)
        for filename in self._load_all_spikes_filenames():#self.train_experiments:
            # Create the regex pattern for each number in the numpy array
            testing_match = re.match(rf"^spikes_{train_test_indices[0]}[0-9][0-9]_summed\.npz$", filename)
            testing_match_full = re.fullmatch(rf"^spikes_{train_test_indices[0]}[0-9][0-9]_summed\.npz$", filename)
            match_found = any(re.match(rf"^spikes_{num}[0-9][0-9]_summed\.npz$", filename) for num in train_test_indices)
            
            # Check if the current filename should be appended based on the include flag
            if (include and match_found) or (not include and not match_found):
                subset.append(filename)

        return subset
    
    def _load_experiment_paths(self, train_test_path, include_experiments):
        if train_test_path is None:
            # Load all experiments.
            return self._load_all_spikes_filenames()
        
        # Load only subset of experiments.
        return self._select_experiments_subset(train_test_path, include_experiments)
    
    def _load_model_subset_indices(self, model_subset_path):
        if model_subset_path is not None:
            # Load indices of subset.
            with open(model_subset_path, 'rb') as pickle_file:
                return pickle.load(pickle_file)
        
        return None

    def _load_experiment(self, exp_name, layer):
        # print("Got to loading")
        file_path = os.path.join(self.spikes_dir, layer, exp_name)
        data = load_npz(file_path)
        dense_data = data.toarray().transpose(1, 0)
        if self.model_subset_indices is not None:
            # Subset creation.
            dense_data = dense_data[:, self.model_subset_indices[layer]]

        # print("I loaded the EXPERIMENT")
        return dense_data
    

    # def get_experiment(self, )
    
    def __getitem__(self, idx):
        exp_name = self.experiments[idx]
        
        # inputs = {
        #     'X_ON': torch.zeros((globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.X_ON_SIZE)),
        #     'X_OFF': torch.zeros((globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.X_OFF_SIZE)),
        # }

        # outputs = {
        #     'V1_Exc_L4': torch.zeros((globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.L4_EXC_SIZE)),
        #     'V1_Inh_L4': torch.zeros((globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.L4_INH_SIZE)),
        #     'V1_Exc_L23': torch.zeros((globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.L23_EXC_SIZE)), 
        #     'V1_Inh_L23': torch.zeros((globals.BLANK_DURATION + globals.IMAGE_DURATION, globals.L23_INH_SIZE)), 
        # }


        inputs = {layer: self._load_experiment(exp_name, layer) for layer in self.input_layers}
        inputs = {layer: torch.tensor(input_data, dtype=torch.half) for layer, input_data in inputs.items()}
        
        outputs = {layer: self._load_experiment(exp_name, layer) for layer in self.output_layers}
        outputs = {layer: torch.tensor(output_data, dtype=torch.half) for layer, output_data in outputs.items()} 
        
        return inputs, outputs

