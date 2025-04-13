"""
Definition of a plugin that prepares histograms and does operations with them.
"""

from typing import Dict, Tuple, Any


import torch
from tqdm import tqdm

import nn_model.globals
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    HistogramFields,
    StatisticsFields,
)


class DatasetAnalyzer:
    """
    Class for processing and creating histograms from the dataset.
    """

    # Maximal number of spikes per neuron in one experiment (experiment duration).
    max_neuron_spikes = (
        nn_model.globals.BLANK_DURATION + nn_model.globals.IMAGE_DURATION
    )
    # Maximal number of spikes in one time bin (`20` because it is our largest studied time bin).
    max_time_bin_spikes = 20

    def __init__(
        self,
    ):
        # Initialize histogram values.
        self.histogram_variants = DatasetAnalyzer._init_histogram_bins(
            DatasetAnalyzer._init_histogram_variants()
        )

    @staticmethod
    def _init_histogram_variants(
        max_neuron_spikes: int = nn_model.globals.BLANK_DURATION
        + nn_model.globals.IMAGE_DURATION,
        max_time_bin_spikes: int = 20,
    ) -> Dict[AnalysisFields, Dict[HistogramFields, Any]]:
        """
        Initializes all histograms that will be processed during analysis (neurons and time bins).

        :param max_neuron_spikes: Maximal number of spike count per neuron
        (time duration of experiment).
        :param max_time_bin_spikes: Maximal length of time bin (in our case 20).
        :return: Returns dictionary prepared to process all variants of histograms
        with predefined values inside.
        """
        return {
            AnalysisFields.HISTOGRAM_NEURON_SPIKE_RATES: {
                HistogramFields.NUM_BINS: max_neuron_spikes,
                HistogramFields.SUMMING_FUNCTION: DatasetAnalyzer._get_neuron_spike_counts_across_experiments,
            },
            AnalysisFields.HISTOGRAM_TIME_BIN_SPIKE_RATES: {
                HistogramFields.NUM_BINS: max_time_bin_spikes,
                HistogramFields.SUMMING_FUNCTION: DatasetAnalyzer._get_neuron_spike_counts_across_time_bin,
            },
        }

    @staticmethod
    def _init_histogram_fields(
        histogram_variants: Dict[AnalysisFields, Dict[HistogramFields, Any]],
    ) -> Dict[AnalysisFields, Dict[HistogramFields, Any]]:
        """
        Initializes histogram bins distribution tensor for all provided
        histograms from (0 to num_bins) and counts as empty dictionary.

        :param histogram_variants: Variants to add the bin objects.
        :return: Returns updated histogram variants with the bins added.
        """

        for variant, variant_values in histogram_variants.items():
            histogram_variants[variant][HistogramFields.BINS] = torch.arange(
                0,
                variant_values[HistogramFields.NUM_BINS] + 1,
                dtype=torch.float16,
            )
            histogram_variants[variant][HistogramFields.COUNTS] = {}

        return histogram_variants

    @property
    def get_histogram_data(self) -> Dict[str, Any]:
        """
        Returns all accumulated histogram data in form of a dictionary.
        """
        # Prepare data to save
        # TODO: rewrite
        return {}

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
    def _check_layer_field_init(
        histogram_variants: Dict[AnalysisFields, Dict[HistogramFields, Any]],
        layer: str,
        device,
    ) -> Dict[AnalysisFields, Dict[HistogramFields, Any]]:
        """
        Takes all histogram variants checks whether layer counts has been already
        initialized and if not then initializes it for the proper bin size.

        :param histogram_variants: All histogram variants to check.
        :param layer: Layer to check.
        :param device: Device where to store the values.
        :return: Returns updated histogram variants with layer initialized.
        """
        for histogram_key, histogram_values in histogram_variants.items():
            if (
                layer
                not in histogram_variants[histogram_key][HistogramFields.COUNTS][layer]
            ):
                histogram_variants[histogram_key][HistogramFields.COUNTS][layer] = (
                    torch.zeros(
                        histogram_values[HistogramFields.NUM_BINS], device=device
                    )
                )
        return histogram_variants

    @staticmethod
    def _get_neuron_spike_counts_across_experiments(
        data: torch.Tensor, dim: int = 2
    ) -> torch.Tensor:
        """
        Computes neuron spike counts across all experiments and return 1D tensor
        of all counts.

        :param data: Batch layer spikes data.
        :param dim: Time step dimension.
        :return: Returns spiking data of neuron spike counts for each experiment
        separately in 1D array of counts (for histogram computation).
        """
        return torch.sum(data, dim=2).view(-1).float()

    @staticmethod
    def _get_neuron_spike_counts_across_time_bin(data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Batch layer spikes data.
        :return: Returns 1D array of spike counts for each time bin in data separatelly (for histogram compuation).
        """
        return data.view(-1).float()

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
        data: torch.Tensor,
        layer: str,
        histogram_variants: Dict[AnalysisFields, Dict[HistogramFields, Any]],
        device,
    ) -> Dict[AnalysisFields, Dict[HistogramFields, Any]]:
        """
        Processes batch update of all the histograms in the given layer.

        :param data: Layer batch data.
        :param layer: Layer name.
        :param histogram_variants: All histogram variants to update.
        :param device: On which device we work.
        :return: Returns tuple of updated histograms of experiment and bins.
        """
        # for layer, data in batch_data_dict.items():
        histogram_variants = DatasetAnalyzer._check_layer_field_init(
            histogram_variants, layer, device
        )

        for variant, variant_value in histogram_variants.items():
            # For each histogram variant update the counts in bins.
            histogram_variants[variant][HistogramFields.COUNTS][layer] = (
                DatasetAnalyzer._data_histogram_update(
                    # Apply summing function of each histogram on data to
                    # prepare the counts values for histogram creation from.
                    variant_value[HistogramFields.SUMMING_FUNCTION](data),
                    layer,
                    variant_value[HistogramFields.NUM_BINS],
                    histogram_variants[variant][HistogramFields.COUNTS][layer],
                )
            )

        return histogram_variants

    @staticmethod
    def _convert_histograms_to_numpy(
        histogram_variants: Dict[AnalysisFields, Dict[HistogramFields, Any]],
    ) -> Dict[AnalysisFields, Dict[HistogramFields, Any]]:
        """
        Converts all provided histograms to numpy and cpu representation.

        :param histogram_variants: Histogram variants to be converted.
        :return: Converted histograms to Numpy.
        """
        for variant, variant_values in histogram_variants.items():
            histogram_variants[variant][HistogramFields.COUNTS] = {
                layer: value.cpu().numpy()
                for layer, value in variant_values[HistogramFields.COUNTS].items()
            }
        return histogram_variants

    def full_analysis_run(
        self,
        loader,
        layer: str = "",
        subset: int = -1,
        include_input: bool = True,
        include_output: bool = True,
    ):
        # Select GPU if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i, (inputs, targets) in enumerate(tqdm(loader)):
            if 0 <= subset == i:
                # Subset -> skip the rest
                break

            # Load the batch of data.
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            # Take only specified layers.
            batch_data_dict = DatasetAnalyzer._select_layer_data(
                inputs, targets, layer, include_input, include_output
            )

            for layer, data in batch_data_dict.items():
                # Update histogram counts.
                self.histogram_variants = DatasetAnalyzer._batch_histogram_update(
                    data, layer, self.histogram_variants, device
                )

        # Final conversion of the histogram values to CPU and NUMPY representation.
        self.histogram_variants = DatasetAnalyzer._convert_histograms_to_numpy(
            self.histogram_variants
        )

        return self.histogram_variants
