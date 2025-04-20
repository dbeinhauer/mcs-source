"""
This script serves as main part that loads and processes the analysis results.
"""

from typing import Dict, Any, List, Tuple, Iterable, Union
import pickle
import re
import os
from enum import Enum

from itertools import product
import pandas as pd
import numpy as np

from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    HistogramFields,
    StatisticsFields,
    DatasetVariantField,
    DatasetDimensions,
)

from evaluation_tools.fields.dataset_parameters import (
    ALL_TIME_STEP_VARIANTS,
    ALL_SUBSET_IDS,
)
from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.fields.experiment_parameters_fields import (
    ModelEvaluationRunVariant,
)
from evaluation_tools.scripts.pickle_manipulation import load_pickle_file

import nn_model.globals

from results_analysis_tools.plugins.results_loader import ResultsLoader
from results_analysis_tools.plugins.histogram_processor import DatasetHistogramProcessor
from results_analysis_tools.fields.experiment_analyses import (
    PlottingVariants,
    PluginVariants,
)
from results_analysis_tools.plugins.temporal_evolution_processor import (
    TemporalEvolutionProcessor,
)


class ResultAnalyzer:
    """
    This class serves for statistical analysis of all processed results.
    """

    def __init__(self, results_to_load: Dict[EvaluationProcessorChoices, str]):

        # Loads all analysis results.
        self.results_loader = ResultsLoader(results_to_load)
        self.all_results = self.results_loader.load_all_results()

        # Plugins
        self.all_plugins = {
            PluginVariants.DATASET_HISTOGRAM_PROCESSOR: DatasetHistogramProcessor(
                self.all_results
            ),
            PluginVariants.TEMPORAL_EVOLUTION_PROCESSOR: TemporalEvolutionProcessor(
                self.all_results
            ),
        }

    @property
    def get_all_results(self) -> Dict[EvaluationProcessorChoices, pd.DataFrame]:
        """
        :return: Returns all analyses results.
        """
        return self.all_results

    def prepare_dataframe_for_plot(
        self, variant: PlottingVariants, is_test: bool = False
    ) -> pd.DataFrame:

        if variant == PlottingVariants.TIME_BIN_COUNT_RATIO:
            return self.all_plugins[
                PluginVariants.DATASET_HISTOGRAM_PROCESSOR
            ].prepare_for_plotting(is_test=is_test)
        if variant == PlottingVariants.TEMPORAL_SPIKE_DISTRIBUTION:
            return self.all_plugins[
                PluginVariants.TEMPORAL_EVOLUTION_PROCESSOR
            ].prepare_for_plotting(is_test=is_test)

        # Plotting variant not implement yet.
        return None

    def get_mean_spike_counts(
        self,
        is_test: bool,
        format_to_latex: bool = False,
    ) -> Union[Dict[int, List[Tuple[int, float]]] | str]:
        """
        Retrieve ratios of each spike count across all time bins.

        :param is_test: Flag whether to process test dataset, else train.
        :param format_to_latex: Flag whether to return the results as LaTeX table.
        :return: Returns ratios of each spike count for each time binning as either
        dictionary of LaTeX table representation.
        """
        distribution = self.all_plugins[
            PluginVariants.DATASET_HISTOGRAM_PROCESSOR
        ].compute_spike_count_distribution(is_test=is_test)

        if format_to_latex:
            # Format distribution to LaTeX
            return DatasetHistogramProcessor.format_distribution_as_latex_table(
                distribution
            )

        return distribution

    def dataset_time_binning_temporal_resolution_correlation_matrix(
        self, is_test: bool = False
    ) -> pd.DataFrame:
        """
        Computes correlation matrix of the time binning and temporal resolution
        across all layers.

        :param is_test: Flag whether to process test dataset, else train.
        :return: Returns correlation matrix of the time binning and temporal
        resolution across all layers.
        """
        return self.all_plugins[
            PluginVariants.TEMPORAL_EVOLUTION_PROCESSOR
        ].compute_correlation_matrix(is_test=is_test)


if __name__ == "__main__":
    EVALUATION_RESULTS_BASE = "/analysis_results"
    analysis_paths = {
        EvaluationProcessorChoices.FULL_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.FULL_DATASET_ANALYSIS.value}/",
        EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS.value}/",
        # EvaluationProcessorChoices.WANDB_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.WANDB_ANALYSIS.value}/",
        EvaluationProcessorChoices.PREDICTION_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.PREDICTION_ANALYSIS.value}/",
    }
    result_analyzer = ResultAnalyzer(analysis_paths)

    result_analyzer.prepare_dataframe_for_plot(
        PlottingVariants.TEMPORAL_SPIKE_DISTRIBUTION
    )
