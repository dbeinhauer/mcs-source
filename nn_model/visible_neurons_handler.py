from typing import Dict

import pickle

import nn_model.globals

class VisibleNeuronsHandler:
    
    def __init__(self, arguments):
        self.visible_neurons_ratio = arguments.visible_neurons_ratio
        self.visible_neuron_indices = self._load_visible_neuron_indices()
    
    @staticmethod
    def get_visible_indices_path(visible_neurons_ratio, size_multiplier = nn_model.globals.SIZE_MULTIPLIER * 100, directory_path: str = "") -> str:
        """
        Returns the path to the visible neurons indices file based on the ratio and size multiplier.
        
        :param visible_neurons_ratio: Ratio of visible neurons.
        :param size_multiplier: Size multiplier for the model.
        :return: Path to the visible neurons indices file.
        """
        directory_path = nn_model.globals.DEFAULT_PATHS[nn_model.globals.PathDefaultFields.VISIBLE_NEURONS_DIR.value] if directory_path == "" else directory_path
        filename = f"size_{int(size_multiplier*100)}_ratio_{visible_neurons_ratio}.pkl"
        return f"{directory_path}/{filename}"
    
    def _load_visible_neuron_indices(self) -> Dict[str]:
        """
        Loads indices of the visible neurons for each layer.

        :return: 
        """
        with open(self.get_visible_indices_path(self.visible_neurons_ratio), "rb") as pickle_file:
            return pickle.load(pickle_file)