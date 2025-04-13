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
    DatasetDimensions,
    DatasetVariantField,
    # HISTOGRAM_ANALYSES,
    # SEPARATE_EXPERIMENT_ANALYSES,
)
from evaluation_tools.plugins.histogram_processor import HistogramProcessor
from evaluation_tools.plugins.time_bin_spike_counter import TimeBinSpikeCounter
from evaluation_tools.plugins.separate_experiment_processor import (
    SeparateExperimentProcessor,
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
        is_test: bool,
        fields_to_analyze: List[AnalysisFields] = list(AnalysisFields),
    ):

        # Flag whether we are processing test dataset.
        self.is_test = is_test
        # What fields we want to analyze.
        self.fields_to_analyze = fields_to_analyze

        # Histogram analysis fields initialization.
        self.histogram_variants = HistogramProcessor.init_histogram_fields(
            HistogramProcessor.init_histogram_variants(fields_to_analyze),
        )
        # Spike counts of each time bin.
        self.time_bin_spike_counts: Dict[str, torch.Tensor] = {}
        # Statistics for separate experiments.
        self.separate_experiment_statistics: Dict[
            Dict[StatisticsFields, Dict[str, List[torch.Tensor]]]
        ] = SeparateExperimentProcessor.init_analysis_fields()

    @property
    def get_histogram_data(self) -> Dict[AnalysisFields, Dict[HistogramFields, Any]]:
        """
        Returns all accumulated histogram data in form of a dictionary
        with values as numpy arrays.
        Expected shapes:
            For neuron counts: `[total_time_duration]`
            For time bin counts: `[20]`
        """
        return HistogramProcessor.to_numpy(self.histogram_variants)

    @property
    def get_time_bin_spike_counts(self) -> Dict[str, Any]:
        """
        Returns spike counts in each time bin in from of
        dictionary of numpy array values.
        Expected shape: `[time_bins]`
        """
        return TimeBinSpikeCounter.to_numpy(self.time_bin_spike_counts)

    @property
    def get_separate_experiments_analysis(
        self,
    ) -> Dict[StatisticsFields, Dict[str, Any]]:
        """
        Returns all analysis data for separate experiments.
        Expected shape: `[experiments, trials]`
        """
        return SeparateExperimentProcessor.to_numpy(self.separate_experiment_statistics)

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

    def _count_spikes_in_time_bin(self, data: torch.Tensor) -> torch.Tensor:
        """
        Sums spike counts across all experiments, trials and neurons.

        :param data: Spiking batch layer data.
        :return: Returns tensor of spike counts for each time bin (temporal resolution).
        Output shape is: `[num_time_steps]`
        """
        return data.sum(dim=(0, 1, 3))

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
                # Process histograms.
                self.histogram_variants = HistogramProcessor.batch_histogram_update(
                    data, layer, self.histogram_variants, device
                )

                if AnalysisFields.TIME_BIN_SPIKE_COUNTS in self.fields_to_analyze:
                    self.time_bin_spike_counts = (
                        TimeBinSpikeCounter.batch_time_bin_update(
                            data, layer, self.time_bin_spike_counts
                        )
                    )

                if (
                    AnalysisFields.SEPARATE_EXPERIMENT_ANALYSIS
                    in self.fields_to_analyze
                ):
                    self.separate_experiment_statistics = (
                        SeparateExperimentProcessor.batch_separate_experiments_update(
                            data,
                            layer,
                            self.separate_experiment_statistics,
                        )
                    )
