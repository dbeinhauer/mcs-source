import sys
import os
from typing import List

import pickle
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

    def create_spikes_histogram(
        self, processed_layer: str = "", subset: int = -1
    ):  # , spikes_directory: str):
        """
        Creates histogram of number of neurons per number of spikes bins for all targets.
        """
        input_layers, output_layers = ModelExecuter._split_input_output_layers(
            nn_model.globals.ORIGINAL_SIZES
        )

        test_dataset = SparseSpikeDataset(
            self.dataset_dir,
            input_layers,
            output_layers,
            is_test=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=nn_model.globals.test_batch_size,
            shuffle=False,  # Load the test dataset always in the same order
            collate_fn=different_times_collate_fn,
        )

        # dict_flattened_data = {layer: [] for layer in output_layers}
        all_flattened_data = []

        for i, (_, targets) in enumerate(tqdm(test_loader)):
            if i == subset:
                break
            for layer, target in targets.items():
                if processed_layer and processed_layer != layer:
                    continue
                summed_data = torch.sum(target, dim=2)
                summed_data = summed_data.view(-1)
                all_flattened_data.append(summed_data)

        final_flattened_data = torch.cat(all_flattened_data).float()

        return final_flattened_data.numpy()
        # number_bins = int(final_flattened_data.max().item())
        # hist, bin_edges = torch.histogram(final_flattened_data, bins=number_bins)

        # return hist, bin_edges
        # print(bin_edges)

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

        # return time_sum

        if layer not in layer_responses_sum[identifier]:
            layer_responses_sum[identifier][layer] = time_sum
        else:
            if layer_responses_sum[identifier][layer].shape[0] < time_sum.shape[0]:
                layer_responses_sum[identifier][layer] = torch.nn.functional.pad(
                    layer_responses_sum[identifier][layer],
                    (
                        0,
                        abs(
                            layer_responses_sum[identifier][layer].shape[0]
                            - time_sum.shape[0]
                        ),
                        # 0,
                    ),
                    "constant",
                    0,
                )
            else:
                time_sum = torch.nn.functional.pad(
                    time_sum,
                    (
                        0,
                        abs(
                            layer_responses_sum[identifier][layer].shape[0]
                            - time_sum.shape[0]
                        ),
                    ),
                    "constant",
                    0,
                )

            layer_responses_sum[identifier][layer] += time_sum

    def iterate_through_all_responses(self, subset: int = -1):
        """
        Iterates through all mean responses (both predictions and targets).
        While iterating performs selected task.

        """
        layer_responses_sum = {}
        all_batch_response_filenames = os.listdir(os.path.join(self.responses_dir))
        self.num_responses = len(all_batch_response_filenames)

        # counter =

        # sizes = set()

        # dict_sizes = {}

        for i, response_filename in enumerate(tqdm(all_batch_response_filenames)):
            if i == subset:
                break
            all_predictions_and_targets = self.load_pickle_file(
                self.responses_dir + "/" + response_filename
            )
            # counter += 1
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
                    # # sizes.add(times.shape[0])
                    # num_time_steps = times.shape[0]

                    # # if num_time_steps

                    # if num_time_steps not in dict_sizes:
                    #     dict_sizes[num_time_steps] = 1
                    # else:
                    #     dict_sizes[num_time_steps] += 1

        # print(dict_sizes)

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
    dataset_dir = f"/home/beinhaud/diplomka/mcs-source/dataset/test_dataset/compressed_spikes/trimmed/size_{nn_model.globals.TIME_STEP}"
    responses_dir = "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/full_evaluation_results/model-10_step-20_lr-1e-05_complex_residual-False_neuron-layers-5_neuron-size-10"
    response_analyzer = ResponseAnalyzer(dataset_dir, responses_dir)

    # response_analyzer.create_spikes_histogram()
    response_analyzer.iterate_through_all_responses()
    print(response_analyzer.mean_layer_responses)
    print(response_analyzer.mean_layer_responses["predictions"]["V1_Exc_L4"].shape)
