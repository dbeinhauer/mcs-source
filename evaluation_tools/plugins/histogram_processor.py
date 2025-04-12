"""
Definition of a plugin that prepares histograms and does operations with them.
"""

from typing import Dict, Tuple, Any

from enum import Enum

import torch
from tqdm import tqdm

import nn_model.globals


# class HistogramFields(Enum):
#     HISTOGRAM_EXPERIMENTS_COUNTS = "histogram_experiment_counts"  # neuron firing rates
#     BIN_EDGES_EXPERIMENT = "bin_edges_experiment"  # neuron_firing rates bins
#     HISTOGRAM_BIN_COUNTS = "histogram_bin_counts"  # time bin firing rates
#     BIN_EDGES_BINS = "bin_edges_bins"  # time bin firing rates bins


class HistogramFields(Enum):
    NEURON_SPIKE_RATES = (
        "neuron_spike_rates"  # Neuron firing rates histogram values (bins, counts).
    )
    TIME_BIN_SPIKE_RATES = (
        "time_bin_spike_rates"  # Time bin firing rates histogram values (bins, counts).
    )


class SummarizationFields(Enum):
    TIME_BIN_COUNTS = "time_bin_counts"  # Total number of spikes in each time bin.


class HistogramProcessor:
    """
    Class for processing and creating histograms from the dataset.
    """

    def __init__(
        self,
    ):
        pass

    @property
    def get_histogram_data(self) -> Dict[str, Any]:
        """
        Returns all accumulated histogram data in form of a dictionary.
        """
        # Prepare data to save
        return {
            HistogramFields.HISTOGRAM_EXPERIMENTS_COUNTS.value: self.histogram_experiment_counts,
            HistogramFields.BIN_EDGES_EXPERIMENT.value: self.bin_edges_experiment,
            HistogramFields.HISTOGRAM_BIN_COUNTS.value: self.histogram_bin_counts,
            HistogramFields.BIN_EDGES_BINS.value: self.bin_edges_bins,
        }

    @staticmethod
    def _select_layer_data(
        inputs, targets, layer: str, include_input: bool, include_output: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Selects data from batch for given layers.

        :param inputs: Input layers (LGN).
        :param targets: Output layers (V1).
        :param layer: Layer identifier (if "" -> take all possible).
        :param include_input: Whether to take input layers.
        :param include_output: Whether to tak output layers.
        :return: Returns dictionary of selected layers.
        """
        result_dictionary = {**inputs, **targets}

        # Select layer for histogram generation.
        if not layer:
            # Layer not specified.
            if include_input and not include_output:
                return inputs
            elif not include_input and include_output:
                return targets

            return result_dictionary

        # Take specified layer.
        return {k: v for k, v in result_dictionary.items() if k == layer}

    @staticmethod
    def _data_histogram_update(
        data: torch.Tensor,
        layer: str,
        num_bins: int,
        histogram_counts: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Updates histogram with provided batch data.

        :param data: Data to create histogram bins from.
        :param layer: Layer of the data.
        :param num_bins: Total number of bins (+1 bin for 0 value).
        :param histogram_counts: Dictionary of temporary histogram sums.
        :return: Returns updated histogram counts for the given data.
        """
        bin_indices_experiments = torch.clamp((data).long(), min=0, max=num_bins - 1)
        bin_counts = torch.bincount(bin_indices_experiments, minlength=num_bins)
        histogram_counts[layer] += bin_counts[:num_bins]  # truncate if needed

        return histogram_counts

    @staticmethod
    def _batch_histogram_update(
        batch_data_dict: Dict[str, torch.Tensor],
        num_neuron_spikes: int,
        num_spikes_per_bin: int,
        histogram_experiment_counts: Dict[str, torch.Tensor],
        histogram_bin_counts: Dict[str, torch.Tensor],
        device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Processes batch update of all the histograms in all layers.

        :param batch_data_dict: Current batch data.
        :param num_neuron_spikes: Number of bins in experiment time scale.
        :param num_spikes_per_bin: Number of bins in time bin scale.
        :param histogram_experiment_counts: Current experiment histogram state.
        :param histogram_bin_counts: Current time bins histogram state.
        :param device: On which device we work.
        :return: Returns tuple of updated histograms of experiment and bins.
        """
        for layer, data in batch_data_dict.items():
            # Sum over time -> neuron spike rate.
            neuron_sum_flat = torch.sum(data, dim=2).view(-1).float()
            # Sum over each time trial -> distribution of spikes in time bins.
            trials_flat = data.view(-1).float()

            # Initialize bin counts (both histograms should have same keys)
            if layer not in histogram_experiment_counts:
                histogram_experiment_counts[layer] = torch.zeros(
                    num_neuron_spikes, device=device
                )
                histogram_bin_counts[layer] = torch.zeros(
                    num_spikes_per_bin, device=device
                )
            histogram_experiment_counts = HistogramProcessor._data_histogram_update(
                neuron_sum_flat,
                layer,
                num_neuron_spikes,
                histogram_experiment_counts,
            )
            histogram_bin_counts = HistogramProcessor._data_histogram_update(
                trials_flat,
                layer,
                num_spikes_per_bin,
                histogram_bin_counts,
            )

        return histogram_experiment_counts, histogram_bin_counts

    def all_histograms_processing(
        self,
        loader,
        layer: str = "",
        subset: int = -1,
        include_input: bool = True,
        include_output: bool = True,
    ) -> Tuple[Tuple[Dict[str, torch.Tensor]], Tuple[Dict[str, torch.Tensor]]]:
        """
        Generates histogram counts and bins for the given dataset across
        neuron spiking rates (neurons separately each value) and
        general time bins spiking rate (even across all neurons).

        :param loader: Loader object to load the dataset.
        :param layer: Layer to process (if `""`, then all selected)
        :param subset: Whether process only subset of data (if `-1` then all)
        :param include_input: Whether process LGN input.
        :param include_output: Whether process V1 output.
        :return: Returns tuple of two tuples of bin counts and bins of neuron
        spiking rates and spiking rates across all bins.
        """

        # Highest possible number of spikes per experiment and per time bin (highest bin is 20)
        num_neuron_spikes = (
            nn_model.globals.BLANK_DURATION + nn_model.globals.IMAGE_DURATION
        )
        num_spikes_per_bin = 20

        # Select GPU if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize bins for neuron spike sums and time bin sums.
        bin_edges_experiment = torch.arange(
            0, num_neuron_spikes + 1, device=device, dtype=torch.float32
        )
        bin_edges_bins = torch.arange(
            0, num_spikes_per_bin + 1, device=device, dtype=torch.float32
        )

        histogram_experiment_counts: Dict[str, torch.Tensor] = {}
        histogram_bin_counts: Dict[str, torch.Tensor] = {}
        # Initialize the loader.
        # loader = self.test_loader if process_test else self.train_loader

        for i, (inputs, targets) in enumerate(tqdm(loader)):
            if 0 <= subset == i:
                # Subset -> skip the rest
                break

            # Load the batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            batch_data_dict = HistogramProcessor._select_layer_data(
                inputs, targets, layer, include_input, include_output
            )

            # Update histogram counts with current batch.
            histogram_experiment_counts, histogram_bin_counts = (
                HistogramProcessor._batch_histogram_update(
                    batch_data_dict,
                    num_neuron_spikes,
                    num_spikes_per_bin,
                    histogram_experiment_counts,
                    histogram_bin_counts,
                    device,
                )
            )

        # Final histogram conversion to numpy.
        self.histogram_experiment_counts = {
            k: v.cpu().numpy() for k, v in histogram_experiment_counts.items()
        }
        self.histogram_bin_counts = {
            k: v.cpu().numpy() for k, v in histogram_bin_counts.items()
        }
        self.bin_edges_experiment = bin_edges_experiment.cpu().numpy()
        self.bin_edges_bins = bin_edges_bins.cpu().numpy()

        return (self.histogram_experiment_counts, self.bin_edges_experiment), (
            self.histogram_bin_counts,
            self.bin_edges_bins,
        )
