"""
Definition of a plugin that prepares histograms and does operations with them.
"""

from typing import Dict, Tuple, Any, List


import torch
from tqdm import tqdm

import nn_model.globals
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    HistogramFields,
    StatisticsFields,
)
from evaluation_tools.plugins.histogram_processor import HistogramProcessor


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

    def __init__(self, fields_to_analyze: List[AnalysisFields] = list(AnalysisFields)):

        # What fields we want to analyze.
        self.fields_to_analyze = fields_to_analyze

        # Histogram analysis setup (we analyze all histograms at once
        # (it is enough to specify only one of it to process all of them)).
        self.histogram_variants = {}
        if any(
            field in fields_to_analyze
            for field in [
                AnalysisFields.HISTOGRAM_NEURON_SPIKE_RATES,
                AnalysisFields.HISTOGRAM_TIME_BIN_SPIKE_RATES,
            ]
        ):
            self.histogram_variants = HistogramProcessor.init_histogram_fields(
                HistogramProcessor.init_histogram_variants()
            )

        self.time_bin_spike_counts = {}

    @property
    def get_histogram_data(self) -> Dict[AnalysisFields, Dict[HistogramFields, Any]]:
        """
        Returns all accumulated histogram data in form of a dictionary.
        """
        return self.histogram_variants

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

    def count_spikes_in_time_bin(self):
        pass

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
                if self.histogram_variants:
                    # Process histograms.
                    self.histogram_variants = HistogramProcessor.batch_histogram_update(
                        data, layer, self.histogram_variants, device
                    )
                if AnalysisFields.TIME_BIN_SPIKE_COUNTS in self.fields_to_analyze:
                    self.count_spikes_in_time_bin()

        # Final conversion of the histogram values to CPU and NUMPY representation.
        self.histogram_variants = HistogramProcessor.convert_histograms_to_numpy(
            self.histogram_variants
        )

        return self.histogram_variants
