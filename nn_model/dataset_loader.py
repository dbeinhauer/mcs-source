"""
This source code defines dataset class for storing the experiment data. 
"""

import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.sparse import load_npz
import torch
from torch.utils.data import Dataset

import globals


class SparseSpikeDataset(Dataset):
    """
    Child class of `Dataset` to store and manipulate with our dataset.
    """

    def __init__(
        self,
        spikes_dir: str,
        input_layers,  #: dict[str, int],
        output_layers,  #: dict[str, int],
        is_test: bool = False,
        model_subset_path=None,
    ):
        """
        Initializes class atributes, loads model subset indices and all
        paths where the dataset is stored.
        :param spikes_dir: root path where all extracted spikes for all
        layers are stored in sparse scipy representation.
        :param input_layers: dictionary of all input layer names and sizes.
        Typically the layers: ['X_ON', 'X_OFF']
        :param output_layers: dictionary of all output layer names and sizes.
        Typically the layers: ['V1_Exc_L4', 'V1_Ing_L4', 'V1_Exc_L23', 'V1_Inh_L23']
        :param is_test: glag whether the dataset it used for test
        (for multitrial evaluation).
        :param model_subset_path: path to file where indicies of the model subset
        are stored (size defined in `globals.py`). If `None` then no subset.
        """
        # Define basic atributes.
        self.spikes_dir = spikes_dir
        self.input_layers = input_layers
        self.output_layers = output_layers

        # Flag if we are storing multitrial dataset (for evaluation).
        self.is_test = is_test

        # Load subset indices (subset of neurons). Dictionary of np.array of indices.
        self.model_subset_indices = self._load_model_subset_indices(model_subset_path)

        # Load all filenames of experiments to use in dataset.
        self.experiments = self._load_all_spikes_filenames()

    def __len__(self):
        """
        :return: returns number of all filenames (number of data).
        """
        return len(self.experiments)

    def _load_all_spikes_filenames(self, subdir="X_ON"):
        """
        Load all spikes filenames from the provided layer subdirectory
        (the filenames should be same for all subdirectories). Additionally,
        it couples filenames for the same example but with different trial.

        It expects that the directory contains the files with filenames
        in either of the two provided formats:
            `spikes_{experiment_id}_summed.npz`
            or
            `spikes_trial_{trial_id}_{experiment_id}_summed.npz`
        The necessary is that the `experiment_id` is in second-to-last
        position when we split by `_` symbol. And that there are at least
        3 part while splitting by `_` symbol.
        :param subdir: name of the layer subdirectory containing the spikes files.
        :return: Returns list of lists that contain all filenames of all
        trials for the given experiment.
        """
        # Get all filenames in the specified subdirectory
        all_files = os.listdir(os.path.join(self.spikes_dir, subdir))

        # Dictionary to group files by their spike_id
        coupled_files = defaultdict(list)

        for file in all_files:
            if file.endswith(".npz") and "_summed" in file:
                # Split the filename by underscores
                file_without_ext = file.replace(".npz", "")
                parts = file_without_ext.split("_")

                # Ensure the filename ends with 'summed' and spike_id is the second-to-last part
                if len(parts) >= 3 and parts[-2].isdigit():
                    spike_id = parts[-2]  # Extract spike_id as the second-to-last part

                    # Add the file to the list of files with the same spike_id
                    coupled_files[spike_id].append(file)

        # Convert the dictionary values (lists of files) to a list of lists
        return list(coupled_files.values())

    def _load_train_test_indices(self, path: str):
        """
        Loads indices used for train/test split.

        NOTE: Not used now, as we do not have multiple trials in all examples.
        :param path: path to file where all ids of the given subset are stored.
        :return: returns numpy array of all experiment ids of the given subset.
        """
        with open(path, "rb") as pickle_file:
            return pickle.load(pickle_file)

    def _load_model_subset_indices(self, model_subset_path):
        """
        Loads indices of the neurons that are part of the selected subset.
        For running the training/evaluation still on the same subset of neurons.
        :param model_subset_path: path to numpy file containing ids of
        all the neurons from all the layers that are part of the selected subset
        (selected by the subset size). If `None` then do not load model subset
        (use all model).
        :return: Returns dictionary of neuron indices that are part of the
        subset for all layers (keys are layer names). In case we want to select
        all neurons it returs `None`.
        """
        if model_subset_path is not None:
            # Load indices of subset.
            with open(model_subset_path, "rb") as pickle_file:
                return pickle.load(pickle_file)

        return None

    def _prepare_experiment_data(self, file_path: str, layer: str):
        """
        Loads spikes data from spartse scipy representation for
        the given experiment. Converts to dense reprezentation
        (`np.array`), expands it with new dimension (for trials)
        and extracts only subset of neurons (if the subset provided).
        :param file_path: path where the spikes data are stored.
        :param layer: layer subdirectory name.
        :return: Returns loaded spikes data as `np.array` of shape
        (num_trials, time, num_neurons).
        Where `num_trials` is always 1 (new dimension).
        """
        # Load the data from the file and convert to dense reprezentation.
        data = load_npz(file_path)
        data = data.toarray()

        # Create trials dimension.
        data = np.expand_dims(data, axis=0)

        # Transpose to have the shape: (trials, time, neurons)
        data = data.transpose(0, 2, 1)

        # If there is a model subset, apply it.
        if self.model_subset_indices is not None:
            # Subset creation
            data = data[:, :, self.model_subset_indices[layer]]

        return data

    def _load_experiment(self, exp_names, layer: str):
        """
        Loads all spikes data for the provided experiment for all
        the trials and converts the data to proper format for futher
        work with it.
        :param exp_names: list of names of experiment files on same
        data but different trials.
        :param layer: name of the layer to extract (subdirectory name).
        :return: Returns `np.array` of prepared spikes data for the given
        experiment and layer. The shape of the array is:
            `(num_trials, time, num_neurons)`
        """
        # List to hold the loaded data from each file.
        all_data = []
        for exp_name in exp_names:
            # Construct the full file path for each file in the list.
            file_path = os.path.join(self.spikes_dir, layer, exp_name)

            # Load and prepare the data for the next operations.
            data = self._prepare_experiment_data(file_path, layer)
            all_data.append(data)

        # Concatenate all loaded data along the trials dimension (axis 0).
        combined_data = np.concatenate(all_data, axis=0)

        return combined_data

    def __getitem__(self, idx):
        """
        Loads spikes data for the given experiment splitted on input and
        output layers.
        :param idx: ID of the experiment to load.
        :return: Returns tuple of two dictionaries. First stands for input
        data (input layers). Second is stands for output data (output layers).
        The directory keys are names of the layers, the values are `np.array`s
        of the spikes for the corresponding layer.
        """
        exp_name = self.experiments[idx]

        # Load inputs and outputs for the given id.
        inputs = {
            layer: self._load_experiment(exp_name, layer) for layer in self.input_layers
        }
        inputs = {
            layer: torch.tensor(input_data, dtype=torch.half)
            for layer, input_data in inputs.items()
        }

        outputs = {
            layer: self._load_experiment(exp_name, layer)
            for layer in self.output_layers
        }
        outputs = {
            layer: torch.tensor(output_data, dtype=torch.half)
            for layer, output_data in outputs.items()
        }

        return inputs, outputs


def different_times_collate_fn(batch):
    """
    Function that deals with loading data in batch that differ in time
    duration (there are fragments of dataset that are slightly different
    in time duration (the start and end parts of the experiment run)).
    It padds the missing time with zeros (pads missing blank stage).
    :param batch: batch of data to pad (tuple of input and output).
    :return: Returns tuple of padded batch of input and output data.
    """
    # Find the maximum size in the second (time) dimension.
    max_size = max([next(iter(item[0].values())).size(1) for item in batch])

    # Initialize a list to hold the result dictionaries.
    num_dicts = len(batch[0])
    result = [{} for _ in range(num_dicts)]

    # Loop over each index in the tuples
    for i in range(num_dicts):
        # Get all dictionaries at the current index across all tuples
        dicts_at_index = [tup[i] for tup in batch]

        # Get the keys (assuming all dictionaries have the same keys)
        keys = dicts_at_index[0].keys()

        # For each key, concatenate the tensors from all dictionaries at the current index
        for key in keys:

            # Collect all tensors associated with the current key
            tensors_to_concat = [
                torch.nn.functional.pad(
                    d[key],
                    (0, 0, 0, max_size - d[key].size(1)),
                    "constant",
                    0,
                )
                for d in dicts_at_index
            ]

            # Concatenate tensors along a new dimension (e.g., dimension 0)
            result[i][key] = torch.stack(tensors_to_concat, dim=0)

    # Convert the list of dictionaries into a tuple
    return tuple(result)
