"""
This source code defines class used for analysis and plotting the model evaluation results
and dataset analysis.
"""

import sys
import os
from typing import List, Dict, Tuple, Optional
import random
import argparse
from enum import Enum

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nn_model.globals
from nn_model.type_variants import (
    LayerType,
    NeuronModulePredictionFields,
    PathPlotDefaults,
    PathDefaultFields,
    EvaluationMeanVariants,
)
from nn_model.dataset_loader import SparseSpikeDataset, different_times_collate_fn
from nn_model.model_executer import ModelExecuter
from nn_model.type_variants import EvaluationFields
from nn_model.dictionary_handler import DictionaryHandler

from plugins.histogram_processor import HistogramProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use the second GPU


class AnalyzerChoices(Enum):
    HISTOGRAM_TRAIN = "histogram_train"
    HISTOGRAM_TEST = "histogram_test"


class ResponseAnalyzer:
    """
    Class used for analysis of the model responses and the dataset properties.
    """

    target_subdirectory_prefix = "V1_"

    mean_responses_fields = [
        EvaluationFields.PREDICTIONS.value,
        EvaluationFields.TARGETS.value,
    ]
    rnn_to_neuron_fields = [
        EvaluationFields.PREDICTIONS.value,
        EvaluationFields.RNN_PREDICTIONS.value,
    ]

    def __init__(
        self,
        train_dataset_dir: str,
        test_dataset_dir: str,
        responses_dir: str = "",
        dnn_responses_path: str = "",
        neurons_path: str = "",
        skip_dataset_load: bool = False,
        data_workers_kwargs={},
    ):
        """
        Initializes tool that is used for analysis of the model responses.

        :param dataset_dir: Directory containing original dataset we want to analyse.
        :param responses_dir: Directory containing averaged responses of the model and
        its targets per trial.
        """
        # TODO: make loading of the files optional

        # Set the input directories
        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.responses_dir = responses_dir
        self.dnn_responses_path = dnn_responses_path

        # Initialize batch size (for loader) as default batch size for test dataset.
        self.batch_size = nn_model.globals.TEST_BATCH_SIZE

        # Raw data loaders:
        if not skip_dataset_load:
            self.train_dataset, self.train_loader = self._init_dataloader(
                is_test=False, data_workers_kwargs=data_workers_kwargs
            )
            self.test_dataset, self.test_loader = self._init_dataloader(
                data_workers_kwargs=data_workers_kwargs
            )

        # Selected neurons for the plotting:
        self.selected_neurons = None
        if neurons_path:
            self.selected_neurons = ResponseAnalyzer.load_selected_neurons(neurons_path)

        # Selected image batches for plotting:
        self.images_batches = ResponseAnalyzer.randomly_select_batches()
        # Selected image indices for each selected batch selected for plotting.
        self.batch_image_indices = ResponseAnalyzer.randomly_select_img_index(
            range(0, nn_model.globals.TEST_BATCH_SIZE), len(self.images_batches)
        )

        # Load all batch responses filenames and count its number.
        self.responses_filenames = self._load_responses_filenames(self.responses_dir)
        self.num_responses = len(self.responses_filenames)

        # Dictionary of layers and its mean neural responses through time
        # (all examples, all neurons). For both targets and predictions.
        self.mean_layer_responses: Dict[str, Dict[str, torch.Tensor]] = {}
        # Dictionary of layers and its mean neural responses through time
        # (all examples, all neurons). Only for targets (loading from Dataloader).
        self.mean_input_layer_responses: Dict[str, torch.Tensor] = {}
        # Dictionary of `neuron ids` and its mean responses through time
        self.mean_neurons_responses: Dict[str, torch.Tensor] = {}
        # Dictionary of `neuron ids` and its dictionary of responses on selected images (key is `image_id`)
        self.selected_neurons_responses: Dict[
            str, Dict[int, Dict[int, torch.Tensor]]
        ] = {}
        # Dictionary of RNN responses and its transformations from DNN neuron for each layer.
        self.rnn_to_prediction_responses: Dict[str, Dict[str, torch.Tensor]] = {}
        # Histogram bins and edges:

        self.histogram_processor = HistogramProcessor()
        # self.histogram_experiment_counts: Dict[str, torch.Tensor] = {}
        # self.bin_edges_experiment = np.zeros(0)
        # self.histogram_bin_counts: Dict[str, torch.Tensor] = {}
        # self.bin_edges_bins = np.zeros(0)

    @staticmethod
    def load_pickle_file(filename: str):
        """
        Loads pickle file.

        :param filename: Name of the pickle file.
        :return: Returns content of the pickle file.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def store_pickle_file(filename: str, data_to_store):
        """
        Stored data to pickle file.

        :param filename: Filename.
        :param data_to_store: Data to be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(data_to_store, f)
        print(f"Data saved to {filename}")

    @staticmethod
    def load_selected_neurons(path: str) -> Dict:
        """
        Loads selected neurons for output layers from pickle file.

        :param path: Path of the pickle file where the neuron IDs are stored.
        :return: Returns dictionary of loaded neurons IDs for each layer.
        """
        selected_neurons = ResponseAnalyzer.load_pickle_file(path)

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

    def _init_dataloader(
        self, is_test=True, data_workers_kwargs={}
    ) -> Tuple[SparseSpikeDataset, DataLoader]:
        """
        Initializes dataset and dataloader objects.

        :param is_test: Flag whether load test dataset (multitrial dataset).
        :return: Returns tuple of initialized dataset object and DataLoader object.
        """
        input_layers, output_layers = DictionaryHandler.split_input_output_layers(
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
            # collate_fn=different_times_collate_fn,
            **data_workers_kwargs,
        )

        return dataset, loader

    def _load_responses_filenames(self, responses_dir: str = "") -> List[str]:
        """
        Loads all filenames from the responses directory (batches of responses).

        :param responses_dir: Path to directory containing neuronal responses.
        :returns: Returns list of paths to neuronal responses (or `None` if not stated).
        """
        if responses_dir:
            return os.listdir(os.path.join(responses_dir))

        return []

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
    def pad_tensors(
        tensor1: torch.Tensor, tensor2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes two tensors determines whether their sizes differ and if so, than pads the smaller one with zeros at the end.

        :return: Returns padded tensors to same shape.
        """

        size1 = tensor1.size()
        size2 = tensor2.size()

        # Find the maximum size for each dimension
        max_size = [max(s1, s2) for s1, s2 in zip(size1, size2)]

        # Calculate the padding needed for each tensor
        padding1 = [max_dim - s for s, max_dim in zip(size1, max_size)]
        padding2 = [max_dim - s for s, max_dim in zip(size2, max_size)]

        # Create padding tuples (reverse order for F.pad, last dimension first)
        padding1 = [
            item for sublist in reversed([[0, p] for p in padding1]) for item in sublist
        ]
        padding2 = [
            item for sublist in reversed([[0, p] for p in padding2]) for item in sublist
        ]

        # Pad both tensors
        padded_tensor1 = torch.nn.functional.pad(
            tensor1,
            padding1,
            "constant",
            0,
        )
        padded_tensor2 = torch.nn.functional.pad(
            tensor2,
            padding2,
            "constant",
            0,
        )

        return padded_tensor1, padded_tensor2

    @staticmethod
    def _sum_vectors(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
        """
        Takes two 1D vectors checks their sizes. If sizes are not matching then pads the smaller
        one to corresponding size with zeros at the end of it.

        :param vector1: First vector to be summed.
        :param vector2: Second vector to be summed.
        :return: Returns summed vectors.
        """
        padded_1, padded_2 = ResponseAnalyzer.pad_tensors(vector1, vector2)

        return padded_1 + padded_2

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

    # @staticmethod
    def _update_time_sum(
        self,
        all_layer_data_batch: Dict[str, torch.Tensor],
        sums_dictionary: Dict[str, torch.Tensor],
        variant: str,
        selected_sum: bool = False,
    ):
        """
        Takes data batch, iterates through all layers and updates each layer
        sum of the spikes through time (used for computation of mean spike rate through time).

        :param all_layer_data_batch: One batch of data to add to sum of spikes in all data in time.
        :param sums_dictionary: Current sum of spikes in all layers in time.
        """
        summing_function = ResponseAnalyzer._time_sum_over_layer
        if variant == EvaluationMeanVariants.NEURON_MEAN.value:
            summing_function = ResponseAnalyzer._sum_over_images
        elif variant == EvaluationMeanVariants.IMAGE_MEAN.value:
            summing_function = ResponseAnalyzer._sum_over_neurons

        for layer, layer_data in all_layer_data_batch.items():
            # Sum first across neurons dimension -> sum across batch dimension (images)
            # -> I get 1D tensor of sum of time responses
            if layer not in sums_dictionary:
                sums_dictionary[layer] = torch.zeros(0)
                if variant == EvaluationMeanVariants.NEURON_MEAN.value:
                    sums_dictionary[layer] = torch.zeros((1, layer_data.size(2)))

            if selected_sum:
                if variant == EvaluationMeanVariants.NEURON_MEAN.value:
                    # layer_data = layer_data[:, :, self.selected_neurons[layer]]
                    layer_data = layer_data
                elif variant == EvaluationMeanVariants.IMAGE_MEAN.value:
                    # layer_data = layer_data[self.selected_images, :, :]
                    pass

            sums_dictionary[layer] = ResponseAnalyzer._sum_vectors(
                sums_dictionary[layer],
                summing_function(layer_data),
            )

    @staticmethod
    def _compute_mean_responses(
        responses_sum: Dict[str, torch.Tensor],
        total_number_examples: int,
        batch_multiplier: int,
        # neuron_multiplier: int,
        mean_over_neurons: bool = True,
        subset: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes mean response of each provided layer in time.

        :param responses_sum: Dictionary of sums of spikes in each layer in time.
        :param total_number_examples: Total number of examples in the dataset
        for which we compute the mean spike rate.
        :param batch_multiplier: How many examples are there in one batch. Typically
        `self.test_batch_size` for train dataset and
        `self.test_batch_size * num_trials` per test dataset.
        :param subset: Size of the subset to compute the mean response
        (number of batches used). If `-1` then we use all examples.
        :return: Returns dictionary of mean neuronal responses per each layer in time.
        """

        counter = total_number_examples * batch_multiplier
        if subset != -1:
            counter = subset * batch_multiplier
        return {
            # layer: layer_data / (counter * nn_model.globals.MODEL_SIZES[layer])
            layer: layer_data
            / (
                counter
                * (nn_model.globals.MODEL_SIZES[layer] if mean_over_neurons else 1)
            )
            for layer, layer_data in responses_sum.items()
        }

    def get_original_data_mean_over_time(
        self,
        subset=-1,
        process_test: bool = True,
        include_input: bool = True,
        include_output: bool = False,
    ):
        """
        Iterates through provided data

        :param subset: _description_, defaults to -1ta
        :param process_test: _description_, defaults to True
        :param include_input: _description_, defaults to True
        :param include_output: _description_, defaults to False
        """
        layer_responses_sum = {}

        trials_multiplier = 20
        loader = self.test_loader
        if not process_test:
            # Processing train set -> use train loader and number of trials is 1.
            loader = self.train_loader
            trials_multiplier = 1

        for i, (inputs, targets) in enumerate(tqdm(loader)):
            if i == subset:
                # Create the mean out of the subset of batches.
                break

            batch_to_process = inputs
            if include_output:
                batch_to_process = targets
                if include_input:
                    batch_to_process = {**inputs, **targets}

            self._update_time_sum(
                {
                    layer: torch.sum(data.float(), dim=1)
                    for layer, data in batch_to_process.items()
                },
                layer_responses_sum,
                variant=EvaluationMeanVariants.LAYER_MEAN.value,
            )

        self.mean_input_layer_responses = self._compute_mean_responses(
            layer_responses_sum,
            len(self.test_loader),
            nn_model.globals.TEST_BATCH_SIZE * trials_multiplier,
            subset=subset,
        )

        return self.mean_input_layer_responses

    def _get_data_from_responses_file(self, filename, keys_to_select):
        return {
            key: value
            for key, value in self.load_pickle_file(
                self.responses_dir + "/" + filename
            ).items()
            if key in keys_to_select
        }

    def get_mean_from_evaluated_data(self, subset: int = -1):
        """
        Iterates through all mean responses (both predictions and targets).
        While iterating performs selected task.

        :param
        """
        layer_responses_sum: Dict[str, Dict[str, torch.Tensor]] = {
            identifier: {} for identifier in ResponseAnalyzer.mean_responses_fields
        }
        neuron_responses_sum: Dict[str, Dict[str, torch.Tensor]] = {
            identifier: {} for identifier in ResponseAnalyzer.mean_responses_fields
        }

        for i, response_filename in enumerate(tqdm(self.responses_filenames)):
            if i == subset:
                break

            all_predictions_and_targets = self._get_data_from_responses_file(
                response_filename, ResponseAnalyzer.mean_responses_fields
            )

            for identifier, data in all_predictions_and_targets.items():
                # # TODO: it should be probably somehow done functionally.
                # if identifier not in layer_responses_sum:
                #     layer_responses_sum[identifier] = {}

                self._update_time_sum(
                    data,
                    layer_responses_sum[identifier],
                    variant=EvaluationMeanVariants.LAYER_MEAN.value,
                )
                self._update_time_sum(
                    data,
                    neuron_responses_sum[identifier],
                    variant=EvaluationMeanVariants.NEURON_MEAN.value,
                )

        self.mean_layer_responses = {
            identifier: self._compute_mean_responses(
                layer_sum,
                self.num_responses,
                # len(all_batch_response_filenames),
                nn_model.globals.TEST_BATCH_SIZE,
                subset=subset,
            )
            for identifier, layer_sum in layer_responses_sum.items()
        }
        self.mean_neurons_responses = {
            identifier: self._compute_mean_responses(
                neuron_sum,
                self.num_responses,
                nn_model.globals.TEST_BATCH_SIZE,
                mean_over_neurons=False,
                subset=subset,
            )
            for identifier, neuron_sum in neuron_responses_sum.items()
        }

        return self.mean_layer_responses, self.mean_neurons_responses

    def get_rnn_responses_to_neuron_responses(self, subset: int = -1):
        self.rnn_to_prediction_responses = {
            identifier: {} for identifier in ResponseAnalyzer.rnn_to_neuron_fields
        }
        for i, response_filename in enumerate(tqdm(self.responses_filenames)):
            if i == subset:
                break

            rnn_and_normal_predictions = self._get_data_from_responses_file(
                response_filename, ResponseAnalyzer.rnn_to_neuron_fields
            )

            for identifier, batch_data in rnn_and_normal_predictions.items():
                for layer, layer_values in batch_data.items():
                    # print(layer_values)
                    layer_values_1d = layer_values.ravel()
                    if layer not in self.rnn_to_prediction_responses[identifier]:
                        self.rnn_to_prediction_responses[identifier][layer] = np.zeros(
                            0
                        )

                    self.rnn_to_prediction_responses[identifier][layer] = (
                        np.concatenate(
                            (
                                self.rnn_to_prediction_responses[identifier][layer],
                                layer_values_1d,
                            ),
                            axis=0,
                        )
                    )

        return self.rnn_to_prediction_responses

    def compute_mean_neuron_response_per_all_images(self, neuron_id: int, layer: str):
        """
        Computes mean spatio-temporal response of a selected neurons through all images.

        :param neuron_id: ID of the neuron to compute the mean for.
        :param layer: name of the layer where the selected neuron lies.
        """
        pass

    def generate_histograms(
        self, process_test: bool, save_path: str = "", subset: int = -1
    ):
        """
        Generate histogram of neuronal spike rates and rates in each bin.

        :param process_test: Whether to generate histogram on test dataset.
        :param time_step: Size of time bins to process.
        :param save_path: Where to store the histogram data.
        :param subset: What part to process.
        """
        loader = self.test_loader if process_test else self.train_loader

        self.histogram_processor.all_histograms_processing(loader, subset=subset)
        histogram_data = self.histogram_processor.get_histogram_data

        if save_path:
            ResponseAnalyzer.store_pickle_file(save_path, histogram_data)


def main(arguments):

    # Set time step selected for analysis.
    nn_model.globals.reinitialize_time_step(arguments.time_step)

    train_dir = nn_model.globals.DEFAULT_PATHS[PathDefaultFields.TRAIN_DIR.value]
    test_dir = nn_model.globals.DEFAULT_PATHS[PathDefaultFields.TEST_DIR.value]

    model_name = "model-10_sub-var-9_step-20_lr-7.5e-06_simple_optim-steps-1_neuron-layers-5-size-10-activation-leakytanh-res-False_hid-time-1_grad-clip-10000.0_optim-default_weight-init-default_synaptic-False-size-10-layers-1"

    responses_dir = f"/home/david/source/diplomka/thesis_results/simple/full_evaluation_results/{model_name}/"

    dnn_responses_dir = f"/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/neuron_model_responses/{model_name}.pth"
    neurons_path = f"/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/neurons/model_size_{int(nn_model.globals.SIZE_MULTIPLIER*100)}_subset_10.pkl"

    num_data_workers = args.num_data_workers
    workers_enabled = num_data_workers > 0
    data_workers_kwargs = {
        "collate_fn": different_times_collate_fn,
        "num_workers": num_data_workers,  # number of workers which will supply data to GPU
        "pin_memory": workers_enabled,  # speed up data transfer to GPU
        "prefetch_factor": (
            num_data_workers // 2 if workers_enabled else None
        ),  # try to always have 4 samples ready for the GPU
        "persistent_workers": workers_enabled,  # keep the worker threads alive
    }

    response_analyzer = ResponseAnalyzer(
        train_dir,
        test_dir,
        # responses_dir=responses_dir,
        data_workers_kwargs=data_workers_kwargs,
    )

    if arguments.action in [
        AnalyzerChoices.HISTOGRAM_TEST.value,
        AnalyzerChoices.HISTOGRAM_TRAIN.value,
    ]:
        # Generate histograms.
        process_test = arguments.action == AnalyzerChoices.HISTOGRAM_TEST.value
        response_analyzer.generate_histograms(
            process_test,
            arguments.results_save_path,
            subset=arguments.processing_subset,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute model training or evaluation."
    )
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=[i.value for i in AnalyzerChoices._member_map_.values()],
        help="What type of action do.",
    )
    parser.add_argument(
        "--results_save_path",
        type=str,
        default="",
        help="Where to store the results (if `" "` then do not save).",
    )
    parser.add_argument(
        "--num_data_workers",
        type=int,
        default=2,
        help="Number of CPUs (cores) for data loading.",
    )
    parser.add_argument(
        "--processing_subset",
        type=int,
        default=-1,
        help="Whether to process only subset of data (if `-1` then process all data).",
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=20,
        help="Size of the time bin to analyze.",
    )

    # data = ResponseAnalyzer.load_pickle_file("results.pkl")

    args = parser.parse_args()
    main(args)
