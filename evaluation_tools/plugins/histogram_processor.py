"""
This source serves for definition of the base functionality that works
with the histogram data. It is meant to be part of the `DatasetAnalyzer` class
"""

from typing import Any, Dict, List

import torch
import pandas as pd
import numpy as np

import nn_model.globals
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    HistogramFields,
    HISTOGRAM_ANALYSES,
)


class HistogramProcessor:
    """
    Class that processes histogram values. It is meant to be
    part of `DatasetAnalyzer` class.
    """

    @staticmethod
    def init_histogram_variants(
        fields_to_analyze: List[AnalysisFields],
        max_neuron_spikes: int = nn_model.globals.BLANK_DURATION
        + nn_model.globals.IMAGE_DURATION,
        max_time_bin_spikes: int = 20,
    ) -> Dict[AnalysisFields, Dict[HistogramFields, Any]]:
        """
        Initializes all histograms that will be processed during analysis (neurons and time bins).

        :param field_to_analyze: All fields to analyze (there should be histogram field specified).
        We are interested in
        :param max_neuron_spikes: Maximal number of spike count per neuron
        (time duration of experiment).
        :param max_time_bin_spikes: Maximal length of time bin (in our case 20).
        :return: Returns dictionary prepared to process all variants of histograms
        with predefined values inside.
        """
        histogram_variants = {}

        if AnalysisFields.HISTOGRAM_NEURON_SPIKE_RATES in fields_to_analyze:
            # Include neuron spike rates histogram.
            histogram_variants[AnalysisFields.HISTOGRAM_NEURON_SPIKE_RATES] = {
                HistogramFields.NUM_BINS: max_neuron_spikes,
                HistogramFields.SUMMING_FUNCTION: HistogramProcessor._get_neuron_spike_counts_across_experiments,
            }

        if AnalysisFields.HISTOGRAM_TIME_BIN_SPIKE_RATES in fields_to_analyze:
            # Include time bin spike rates histogram.
            histogram_variants[AnalysisFields.HISTOGRAM_TIME_BIN_SPIKE_RATES] = {
                HistogramFields.NUM_BINS: max_time_bin_spikes,
                HistogramFields.SUMMING_FUNCTION: HistogramProcessor._get_neuron_spike_counts_across_time_bin,
            }

        return histogram_variants

    @staticmethod
    def init_histogram_fields(
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
            if layer not in histogram_variants[histogram_key][HistogramFields.COUNTS]:
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
        num_bins: int,
        histogram_counts: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Updates histogram with provided batch data.

        :param data: Data to create histogram bins from.
        :param num_bins: Total number of bins (+1 bin for 0 value).
        :param histogram_counts: Dictionary of temporary histogram sums.
        :return: Returns updated histogram counts for the given data.
        """
        bin_indices_experiments = torch.clamp((data).long(), min=0, max=num_bins - 1)
        bin_counts = torch.bincount(bin_indices_experiments, minlength=num_bins)
        histogram_counts += bin_counts[:num_bins]  # truncate if needed

        return histogram_counts

    @staticmethod
    def batch_histogram_update(
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
        histogram_variants = HistogramProcessor._check_layer_field_init(
            histogram_variants, layer, device
        )

        for variant, variant_value in histogram_variants.items():
            # For each histogram variant update the counts in bins.
            histogram_variants[variant][HistogramFields.COUNTS][layer] = (
                HistogramProcessor._data_histogram_update(
                    # Apply summing function of each histogram on data to
                    # prepare the counts values for histogram creation from.
                    variant_value[HistogramFields.SUMMING_FUNCTION](data),
                    variant_value[HistogramFields.NUM_BINS],
                    histogram_variants[variant][HistogramFields.COUNTS][layer],
                )
            )

        return histogram_variants

    @staticmethod
    def to_numpy(
        histogram_variants: Dict[AnalysisFields, Dict[HistogramFields, Any]],
    ) -> Dict[AnalysisFields, Dict[HistogramFields, Any]]:
        """
        Converts all provided histogram values to numpy and cpu representation.

        :param histogram_variants: Histogram variants to be converted.
        :return: Converted histograms to Numpy.
        """
        for variant, variant_values in histogram_variants.items():
            histogram_variants[variant][HistogramFields.BINS] = histogram_variants[
                variant
            ][HistogramFields.BINS].numpy()
            histogram_variants[variant][HistogramFields.COUNTS] = {
                layer: value.cpu().numpy()
                for layer, value in variant_values[HistogramFields.COUNTS].items()
            }
        return histogram_variants

    @staticmethod
    def to_pandas(
        histogram_variants: Dict[HistogramFields, Any],
    ) -> pd.DataFrame:
        """
        Converts histogram already converted to numpy to pandas representation.

        :param histogram_variants: Histograms to be converted already in numpy.
        :return: Returns histograms as pandas dataframe.
        """
        rows = []

        histogram_counts = histogram_variants[HistogramFields.COUNTS]
        for layer, layer_counts in histogram_counts.items():
            rows.append(
                {
                    "layer_name": layer,
                    "bins": histogram_variants[HistogramFields.BINS],
                    "counts": layer_counts,
                }
            )
        return pd.DataFrame(rows)
