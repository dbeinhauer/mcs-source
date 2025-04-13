"""
This script serves for processing the data in single experiment
(and trials) manner.
"""

from typing import List, Dict, Any, Tuple

import torch

from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    StatisticsFields,
    DatasetVariantField,
    DatasetDimensions,
    # SEPARATE_EXPERIMENT_ANALYSES,
)
from evaluation_tools.fields.dataset_parameters import DATASET_SIZES


class SeparateExperimentProcessor:
    """
    Class to cover processing of simple experiment statistics.
    """

    # Overview of field format of all the analyses.
    analysis_fields_overview = {
        StatisticsFields.TOTAL_COUNT: None,
        StatisticsFields.MEAN: None,
        StatisticsFields.VARIANCE: None,
        StatisticsFields.DENSITY: None,
        StatisticsFields.SYNCHRONY: None,
    }

    @staticmethod
    def _init_analysis_functions():
        """
        Initializes functions that needs to be executed for each analysis.
        Stores the functions to static `analysis_fields_overview` attribute.
        """
        SeparateExperimentProcessor.analysis_fields_overview[
            StatisticsFields.TOTAL_COUNT
        ] = SeparateExperimentProcessor._total_spike_count
        SeparateExperimentProcessor.analysis_fields_overview[StatisticsFields.MEAN] = (
            SeparateExperimentProcessor._spike_mean_experiment
        )
        SeparateExperimentProcessor.analysis_fields_overview[
            StatisticsFields.VARIANCE
        ] = SeparateExperimentProcessor._spike_variance_experiment
        SeparateExperimentProcessor.analysis_fields_overview[
            StatisticsFields.DENSITY
        ] = SeparateExperimentProcessor._spike_density_experiments
        SeparateExperimentProcessor.analysis_fields_overview[
            StatisticsFields.SYNCHRONY
        ] = SeparateExperimentProcessor._compute_mean_per_bin_synchrony

    @staticmethod
    def init_analysis_fields() -> Dict[StatisticsFields, Dict[str, List[torch.Tensor]]]:
        """
        Initializes functions for each analysis and their appropriate fields.

        :return: Returns dictionary with prepared fields for analysis statistics.
        """
        SeparateExperimentProcessor._init_analysis_functions()
        return {
            field: {} for field in SeparateExperimentProcessor.analysis_fields_overview
        }

    @staticmethod
    def _total_spike_count(data: torch.Tensor) -> torch.Tensor:
        """
        Computes total number of spikes in each experiment and trial.

        :param data: Data to summarize.
        :return: Returns result counts tensor in shape `[batch_size, trials]`.
        """
        return (
            data.float()
            .sum(
                dim=(
                    DatasetDimensions.TIME_STEP.value,
                    DatasetDimensions.NEURON.value,
                )
            )
            .cpu()
        )

    @staticmethod
    def _spike_mean_experiment(data: torch.Tensor) -> torch.Tensor:
        """
        Computes mean spike rate across of spikes for each experiment and its trial.

        :param data: Batch data to compute the means from.
        :return: Returns spike means in experiments and trials.
        Expected shape `[batch_size, trials]`.
        """
        return (
            data.float()
            .mean(
                dim=(
                    DatasetDimensions.TIME_STEP.value,
                    DatasetDimensions.NEURON.value,
                )
            )
            .cpu()
        )

    @staticmethod
    def _spike_variance_experiment(data: torch.Tensor) -> torch.Tensor:
        """
        Computes variance spike rate across of spikes for each experiment and its trial.

        :param data: Batch data to compute the variance from.
        :return: Returns spike variances in experiments and trials.
        Expected shape `[batch_size, trials]`.
        """
        return (
            data.float()
            .var(
                dim=(
                    DatasetDimensions.TIME_STEP.value,
                    DatasetDimensions.NEURON.value,
                ),
                unbiased=False,
            )
            .cpu()
        )

    @staticmethod
    def _spike_density_experiments(data: torch.Tensor) -> torch.Tensor:
        """
        Computes how many spikes happen on average in each time bin
        of the experiment (trial).

        :param data: Batch data to compute the variance from.
        :return: Returns density of spikes in experiments and trials.
        Expected shape `[batch_size, trials]`.
        """
        spikes_per_bin = data.float().mean(
            dim=DatasetDimensions.NEURON.value
        )  # shape: [experiments, trials, time_steps]
        return spikes_per_bin.mean(dim=DatasetDimensions.TIME_STEP.value).cpu()

    @staticmethod
    def _compute_mean_per_bin_synchrony(data: torch.Tensor) -> torch.Tensor:
        """
        Computes mean synchrony of neuronal spikes per all time steps.
        First how many neurons spiked in the interval, then mean
        these values across all time steps.

        :param data: Batch data to compute the variance from.
        :return: Returns average per-bin synchrony of neuron across time.
        """
        # Binary spike presence
        binary_data = (data > 0).float()
        # Fraction of neurons spiking in each time bin
        fraction_neurons_per_bin = binary_data.mean(dim=DatasetDimensions.NEURON.value)
        # Average across time bins
        return fraction_neurons_per_bin.mean(dim=DatasetDimensions.TIME_STEP.value)

    @staticmethod
    def batch_separate_experiments_update(
        data: torch.Tensor,
        layer: str,
        all_analysis_data: Dict[StatisticsFields, Dict[str, List[torch.Tensor]]],
    ) -> Dict[StatisticsFields, Dict[str, List[torch.Tensor]]]:
        for statistics_type in all_analysis_data:
            if layer not in all_analysis_data[statistics_type]:
                all_analysis_data[statistics_type][layer] = []

            # Call analysis function for the given statistic type.
            statistics_result = SeparateExperimentProcessor.analysis_fields_overview[
                statistics_type
            ](data)
            # Assert it returns expected shape
            assert statistics_result.shape == (
                data.shape[DatasetDimensions.EXPERIMENT.value],
                data.shape[DatasetDimensions.TRIAL.value],
            ), "Unexpected tensor shape"
            # Assign statistics to appropriate place in tensor for all experiments.
            all_analysis_data[statistics_type][layer].append(statistics_result)

        return all_analysis_data

    @staticmethod
    def to_numpy(
        data: Dict[StatisticsFields, Dict[str, List[torch.Tensor]]],
    ) -> Dict[StatisticsFields, Dict[str, Any]]:
        """
        Converts the analysis results to numpy representation.

        :param data: Data to be converted.
        :return: Returns provided data converted to numpy array representation.
        """

        return {
            statistics_type: {
                layer: torch.cat(value, dim=DatasetDimensions.EXPERIMENT.value).numpy()
                for layer, value in statistics_values.items()
            }
            for statistics_type, statistics_values in data.items()
        }
