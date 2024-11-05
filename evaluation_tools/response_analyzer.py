import sys
import os
from typing import List

import pickle
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../nn_model")))

from type_variants import LayerType
from dataset_loader import SparseSpikeDataset, different_times_collate_fn


class ResponseAnalyzer:
    """
    Class used for analysis of the model responses and the dataset properties.
    """

    target_subdirectory_prefix = "V1_"

    def __init__(self, dataset_dir: str, responses_dir: str):
        """
        Initializes tool that is used for analysis of the model responses.

        :param dataset_dir: Directory containing original dataset we want to analyse.
        :param responses_dir: Directory containing averaged responses of the model and
        its targets per trial.
        """

        self.dataset_dir = dataset_dir
        self.responses_dir = responses_dir

        # Total number of responses batches to analysis
        self.num_responses = 0

        self.selected_neurons = []
        self.selected_images = []

        # Dictionary of layers and its mean neural responses through time (all examples, all neurons)
        self.mean_layer_responses = {}
        # Dictionary of `neuron ids` and its mean responses through time
        self.mean_neurons_responses = {}
        # Dictionary of `neuron ids` and its dictionary of responses on selected images (key is `image_id`)
        self.selected_neurons_responses = {}

    def create_spikes_histogram(self):
        """
        Creates histogram of number of neurons per number of spikes bins for all targets.
        """
        test_dataset = SparseSpikeDataset(
            self.dataset_dir,
            input_layers,
            output_layers,
            is_test=True,
            model_subset_path=arguments.subset_dir,
            experiment_selection_path=arguments.experiment_selection_path,
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=globals.test_batch_size,
            shuffle=False,  # Load the test dataset always in the same order
            collate_fn=different_times_collate_fn,
        )

        for i, (inputs, targets) in enumerate(tqdm(self.test_loader)):
            pass

    def load_pickle_file(self, filename: str):
        """
        Loads pickle file.

        :param filename: Name of the pickle file.
        :return: Returns content of the pickle file.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def _update_mean_layer_responses(
        self,
        layer_data: torch.Tensor,
        layer_responses_sum,
        identifier: str,
        layer: str,
    ):
        time_sum = torch.sum(torch.sum(layer_data, axis=2), axis=0)
        if layer not in layer_responses_sum[identifier]:
            layer_responses_sum[identifier][layer] = time_sum
        else:
            layer_responses_sum[identifier][layer] += time_sum

    def iterate_through_all_responses(
        self,
    ):
        """
        Iterates through all mean responses (both predictions and targets).
        While iterating performs selected task.

        """
        layer_responses_sum = {}

        all_batch_response_filenames = os.listdir(os.path.join(self.responses_dir))

        self.num_responses = len(all_batch_response_filenames)

        # counter =
        for response_filename in all_batch_response_filenames:
            all_predictions_and_targets = self.load_pickle_file(response_filename)
            counter += 1
            for identifier, data in all_predictions_and_targets.items():
                # TODO: it should be probably somehow done functionally.
                if identifier not in layer_responses_sum:
                    layer_responses_sum[identifier] = {}
                for layer, layer_data in data.items():
                    # Sum first across neurons dimension -> sum across batch dimension (images)
                    # -> I get 1D tensor of sum of time responses
                    self._update_mean_layer_responses(
                        layer_data, layer_responses_sum, identifier, layer
                    )

        counter = len(all_batch_response_filenames)
        self.mean_layer_responses = {
            identifier: {
                layer: layer_data / counter for layer, layer_data in data.items()
            }
            for identifier, data in layer_responses_sum.items()
        }  # all_predictions_and_targets / counter

    def plot_mean_neural_response_per_populations(self):
        """
        Plots mean spatio-temporal responses of all the neurons from population.
        For both averaged predictions and targets.
        """

        counter = len(all_batch_response_filenames)
        average_responses = {
            identifier: {
                layer: layer_data / counter for layer, layer_data in data.items()
            }
            for identifier, data in all_data.items()
        }  # all_predictions_and_targets / counter

        return average_responses

    def compute_mean_neuron_response_per_all_images(self, neuron_id: int, layer: str):
        """
        Computes mean spatio-temporal response of a selected neurons through all images.

        :param neuron_id: ID of the neuron to compute the mean for.
        :param layer: name of the layer where the selected neuron lies.
        """
        pass

    def plot_neuron_responses_on_multiple_images(
        self, neuron_id: int, layer: str, selected_images_ids: List[int]
    ):
        """
        Plots mean neuron responses/targets per selected images.

        :param neuron_id: ID of the neuron to plot the responses for.
        :param layer: name of the layer where the selected neuron lies.
        :param selected_images_ids: list of image ids that we are interested to plot.
        """
        pass
