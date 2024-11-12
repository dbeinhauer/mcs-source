"""
This source code defines class used for analysis and plotting the model evaluation results 
and dataset analysis.
"""

import sys
import os
from typing import List, Dict, Tuple
import random

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nn_model.globals
from nn_model.type_variants import LayerType
from nn_model.dataset_loader import SparseSpikeDataset, different_times_collate_fn
from nn_model.model_executer import ModelExecuter


class ResponseAnalyzer:
    """
    Class used for analysis of the model responses and the dataset properties.
    """

    target_subdirectory_prefix = "V1_"

    def __init__(
        self,
        train_dataset_dir: str,
        test_dataset_dir: str,
        responses_dir: str,
        neurons_path: str,
    ):
        """
        Initializes tool that is used for analysis of the model responses.

        :param dataset_dir: Directory containing original dataset we want to analyse.
        :param responses_dir: Directory containing averaged responses of the model and
        its targets per trial.
        """

        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.responses_dir = responses_dir

        # Initialize batch size (for loader) as default batch size for test dataset.
        self.batch_size = nn_model.globals.test_batch_size

        # Raw data loaders
        self.train_dataset, self.train_loader = self._init_dataloader(is_test=False)
        self.test_dataset, self.test_loader = self._init_dataloader()

        self.selected_neurons = self.load_selected_neurons(neurons_path)
        self.images_batches = self.randomly_select_batches()
        self.batch_image_indices = self.randomly_select_img_index(
            range(0, nn_model.globals.test_batch_size), len(self.images_batches)
        )

        # Total number of responses batches to analysis
        self.num_responses = 0

        # self.responses_time_evolution = {
        #     "layer_mean": {},
        #     "neuron_means": {},
        #     "image_means": {},
        # }

        # Dictionary of layers and its mean neural responses through time (all examples, all neurons)
        self.mean_layer_responses = {}
        # Dictionary of `neuron ids` and its mean responses through time
        self.mean_neurons_responses = {}
        # Dictionary of `neuron ids` and its dictionary of responses on selected images (key is `image_id`)
        self.selected_neurons_responses = {}

    @staticmethod
    def load_selected_neurons(path: str) -> Dict:
        """
        Loads selected neurons for output layers from pickle file.

        :param path: Path of the pickle file where the neuron IDs are stored.
        :return: Returns dictionary of loaded neurons IDs for each layer.
        """
        selected_neurons = {}
        with open(path, "rb") as f:
            selected_neurons = pickle.load(f)

        return {
            layer: data
            for layer, data in selected_neurons.items()
            if layer.startswith(ResponseAnalyzer.target_subdirectory_prefix)
        }

    @staticmethod
    def randomly_select_batches(
        num_batches: int = 90, subset_size: int = 10
    ) -> List[int]:
        """
        Randomly selects batches subset.

        :param num_batches: Total number of batches.
        :param selection_size: Size of the selection subset.
        :return: Returns list of randomly selected indices of the batches.
        """
        return random.sample(np.arange(num_batches).tolist(), subset_size)

    @staticmethod
    def randomly_select_img_index(selection_range, selection_size) -> List[int]:
        """
        Randomly selects indices of the images (each index per each batch).

        :param selection_range: Range of possible indices.
        :param selection_size: How many indices we want to select.
        :return: Returns list of randomly selected image indices.
        """
        return random.choices(selection_range, k=selection_size)

    def _init_dataloader(self, is_test=True) -> Tuple[SparseSpikeDataset, DataLoader]:
        """
        Initializes dataset and dataloader objects.

        :param is_test: Flag whether load test dataset (multitrial dataset).
        :return: Returns tuple of initialized dataset object and DataLoader object.
        """
        input_layers, output_layers = ModelExecuter._split_input_output_layers(
            nn_model.globals.ORIGINAL_SIZES
        )
        dataset_dir = self.test_dataset_dir
        if not is_test:
            dataset_dir = self.train_dataset_dir

        dataset = SparseSpikeDataset(
            dataset_dir,
            input_layers,
            output_layers,
            is_test=is_test,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Load the dataset always in the same order.
            collate_fn=different_times_collate_fn,
        )

        return dataset, loader

    def create_spikes_histogram(
        self,
        processed_layer: str = "",
        subset: int = -1,
        num_bins: int = 144,
        process_test: bool = True,
        include_input: bool = False,
        include_output: bool = True,
    ) -> Tuple:
        """
        Creates histogram counts and bins for the given layer data.

        The histogram should display number of neurons per total number of spikes in
        all examples from the provided dataset.

        :param processed_layer: Name of the layer to create the histogram from.
        :param subset: Create histogram from given number of batches of data.
        If -1 then use all of them.
        :param num_bins: Number of bins in the histogram. Ideally it should be maximal number
        of spikes in the dataset of individual neuron in the dataset.
        :param process_test: Flag whether we want to process test dataset (by default `True`).
        :param include_input: Flag whether we want to include also input layers
        in the histogram creation (by default `False`).
        :param include_output: Flag whether we want to include target layers in the histogram
        creation (by default `True`).
        NOTE: In case neither input nor output included use output.
        :return: Returns tuple of bin counts and bin edges of histogram.
        """

        # Initialize the histogram variables.
        bin_edges = np.arange(0, num_bins)
        hist_counts = np.zeros(len(bin_edges) - 1, dtype=np.float32)

        # Select data loader.
        loader = self.test_loader
        if not process_test:
            loader = self.train_loader

        for i, (inputs, targets) in enumerate(tqdm(loader)):
            if i == subset:
                # Create the histogram from the subset of data.
                break

            # Select layers to include in histogram (by default use only targets).
            result_dictionary = targets
            if include_input:
                # Create histogram only from inputs.
                result_dictionary = inputs
                if include_output:
                    # Include input and output layer in histogram creation.
                    result_dictionary = {**inputs, **targets}

            # Histogram creation.
            for layer, data in result_dictionary.items():
                if processed_layer and processed_layer != layer:
                    # Processing specific layer and not this -> skip layer.
                    continue
                summed_data = torch.sum(data, dim=2).view(-1).float().numpy()
                batch_counts, _ = np.histogram(summed_data, bins=bin_edges)

                hist_counts += batch_counts

        return hist_counts, bin_edges

    def load_pickle_file(self, filename: str):
        """
        Loads pickle file.

        :param filename: Name of the pickle file.
        :return: Returns content of the pickle file.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _pad_vector_to_size(vector: torch.Tensor, size: int) -> torch.Tensor:
        """
        Pads vector with zeros at the end to match the wanted size.

        :param vector: Vector (1D) to be padded.
        :param size: Target vector size (the missing size pad with zeros).
        :return: Returns padded `vector`.
        """
        return torch.nn.functional.pad(
            vector,
            (
                0,
                size - vector.shape[0],
            ),
            "constant",
            0,
        )

    @staticmethod
    def _sum_vectors(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
        """
        Takes two 1D vectors checks their sizes. If sizes are not matching then pads the smaller
        one to corresponding size with zeros at the end of it.

        :param vector1: First vector to be summed.
        :param vector2: Second vector to be summed.
        :return: Returns summed vectors.
        """
        if vector1.shape[0] < vector2.shape[0]:
            vector1 = ResponseAnalyzer._pad_vector_to_size(vector1, vector2.shape[0])
        else:
            vector2 = ResponseAnalyzer._pad_vector_to_size(vector2, vector1.shape[0])

        return vector1 + vector2

    @staticmethod
    def _sum_over_neurons(data: torch.Tensor, dim: int = 2) -> torch.Tensor:
        """
        Sums the data over neurons dimension.

        :param data: Data to be summed.
        :param dim: Axis of neuron dimension.
        :return: Returns summed data.
        """
        return torch.sum(data, dim=dim)

    @staticmethod
    def _sum_over_images(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Sums the data over images (examples) dimension.

        :param data: Data to be summed.
        :param dim: Axis of neuron dimension.
        :return: Returns summed data.
        """
        return torch.sum(data, dim=dim)

    @staticmethod
    def _time_sum_over_layer(data: torch.Tensor) -> torch.Tensor:
        """
        Sums spikes in time for all all the neurons and images in the layer.

        :param data: Tensor of shape `(num_images, time, num_neurons)` to be summed.
        :return: Returns summed data (for plotting mean responses).
        """
        return ResponseAnalyzer._sum_over_images(
            ResponseAnalyzer._sum_over_neurons(data)
        )

    @staticmethod
    def _update_time_sum(
        all_layer_data_batch: Dict[str, torch.Tensor],
        sums_dictionary: Dict[str, torch.Tensor],
    ):
        """
        Takes data batch, iterates through all layers and updates each layer
        sum of the spikes through time (used for computation of mean spike rate through time).

        :param all_layer_data_batch: One batch of data to add to sum of spikes in all data in time.
        :param sums_dictionary: Current sum of spikes in all layers in time.
        """
        for layer, layer_data in all_layer_data_batch.items():
            # Sum first across neurons dimension -> sum across batch dimension (images)
            # -> I get 1D tensor of sum of time responses
            if layer not in sums_dictionary:
                sums_dictionary[layer] = torch.zeros(0)

            sums_dictionary[layer] = ResponseAnalyzer._sum_vectors(
                sums_dictionary[layer],
                ResponseAnalyzer._time_sum_over_layer(layer_data),
            )

    @staticmethod
    def _compute_mean_responses(total_number_examples, subset=-1):
        counter = len(self.test_loader) * nn_model.globals.test_batch_size * 20
        if subset != -1:
            counter = subset * nn_model.globals.test_batch_size * 20
        return {
            layer: layer_data / (counter * nn_model.globals.MODEL_SIZES[layer])
            for layer, layer_data in layer_responses_sum.items()
        }

    def get_original_data_mean_over_time(
        self,
        subset=-1,
        process_test: bool = True,
        include_input: bool = True,
        include_output: bool = False,
    ):
        layer_responses_sum = {}

        loader = self.test_loader
        if not process_test:
            loader = self.train_loader

        for i, (inputs, targets) in enumerate(tqdm(loader)):
            if i == subset:
                # Create the mean out of the subset of batches.
                break

            batch_to_process = inputs
            if include_output:
                batch_to_process = targets
                if include_input:
                    batch_to_process = {**inputs, **targets}

            ResponseAnalyzer._update_time_sum(
                {
                    layer: torch.sum(data.float(), dim=1)
                    for layer, data in batch_to_process.items()
                },
                layer_responses_sum,
            )

        counter = len(self.test_loader) * nn_model.globals.test_batch_size * 20
        if subset != -1:
            counter = subset * nn_model.globals.test_batch_size * 20
        self.mean_input_layer_responses = {
            layer: layer_data / (counter * nn_model.globals.MODEL_SIZES[layer])
            for layer, layer_data in layer_responses_sum.items()
        }

    def get_mean_from_evaluated_data(self, subset: int = -1):
        """
        Iterates through all mean responses (both predictions and targets).
        While iterating performs selected task.

        """
        layer_responses_sum = {}
        all_batch_response_filenames = os.listdir(os.path.join(self.responses_dir))
        self.num_responses = len(all_batch_response_filenames)

        for i, response_filename in enumerate(tqdm(all_batch_response_filenames)):
            if i == subset:
                break
            all_predictions_and_targets = self.load_pickle_file(
                self.responses_dir + "/" + response_filename
            )

            for identifier, data in all_predictions_and_targets.items():
                # TODO: it should be probably somehow done functionally.
                if identifier not in layer_responses_sum:
                    layer_responses_sum[identifier] = {}

                ResponseAnalyzer._update_time_sum(data, layer_responses_sum[identifier])

        counter = len(all_batch_response_filenames) * nn_model.globals.test_batch_size
        self.mean_layer_responses = {
            identifier: {
                layer: layer_data / (counter * nn_model.globals.MODEL_SIZES[layer])
                for layer, layer_data in data.items()
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


if __name__ == "__main__":
    train_spikes_dir = f"/home/beinhaud/diplomka/mcs-source/dataset/train_dataset/compressed_spikes/trimmed/size_{nn_model.globals.TIME_STEP}"
    test_spikes_dir = f"/home/beinhaud/diplomka/mcs-source/dataset/test_dataset/compressed_spikes/trimmed/size_{nn_model.globals.TIME_STEP}"

    all_responses_dir = "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/full_evaluation_results/model-10_step-20_lr-1e-05_complex_residual-False_neuron-layers-5_neuron-size-10"
    neuron_ids_path = "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/neurons/model_size_10_subset_10.pkl"
    images_ids_path = "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/experiments/experiments_subset_10.pkl"
    response_analyzer = ResponseAnalyzer(
        train_spikes_dir,
        test_spikes_dir,
        all_responses_dir,
        neuron_ids_path,
        images_ids_path,
    )

    # response_analyzer.create_spikes_histogram()
    response_analyzer.get_all_input_sums(subset=1)
    print(response_analyzer.mean_input_layer_responses)
    # response_analyzer.iterate_through_all_responses()
    # print(response_analyzer.mean_layer_responses)
    # print(response_analyzer.mean_layer_responses["predictions"]["V1_Exc_L4"].shape)
