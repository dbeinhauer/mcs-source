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
from results_analysis_tools.plugins.synchrony_experiments_processor import (
    SynchronyExperimentsProcessor,
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
            PluginVariants.SYNCHRONY_TIME_BINS_PROCESSOR: SynchronyExperimentsProcessor(
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

        if variant == PlottingVariants.FULL_TIME_BIN_COUNT_RATIO:
            return self.all_plugins[
                PluginVariants.DATASET_HISTOGRAM_PROCESSOR
            ].prepare_for_plotting(is_test=is_test, process_subset=False)
        if variant == PlottingVariants.FULL_TEMPORAL_SPIKE_DISTRIBUTION:
            return self.all_plugins[
                PluginVariants.TEMPORAL_EVOLUTION_PROCESSOR
            ].prepare_for_plotting(is_test=is_test, process_subset=False)
        if (
            variant
            == PlottingVariants.FULL_CORRELATION_MATRIX_BIN_SIZE_TEMPORAL_DATASET
        ):
            return self.all_plugins[
                PluginVariants.TEMPORAL_EVOLUTION_PROCESSOR
            ].compute_correlation_matrix_full(is_test=is_test)
        if variant == PlottingVariants.FULL_SYNCHRONY_TIME_BINS:
            return self.all_plugins[
                PluginVariants.SYNCHRONY_TIME_BINS_PROCESSOR
            ].prepare_for_plotting(is_test=is_test, process_subset=False)
        if variant == PlottingVariants.SUBSET_TEMPORAL_SPIKE_DISTRIBUTION:
            # Plotting variant not implement yet.
            return self.all_plugins[
                PluginVariants.TEMPORAL_EVOLUTION_PROCESSOR
            ].prepare_for_plotting(is_test=is_test, process_subset=True)
        # Plotting variant not implement yet.
        return None

    def get_mean_spike_counts(
        self,
        is_test: bool,
        process_subset: bool = False,
        format_to_latex: bool = False,
    ) -> Union[pd.DataFrame | str]:
        """
        Retrieve ratios of each spike count across all time bins.

        :param is_test: Flag whether to process test dataset, else train.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :param format_to_latex: Flag whether to return the results as LaTeX table.
        :return: Returns ratios of each spike count for each time binning as either
        dictionary of LaTeX table representation.
        """

        if process_subset:
            # If process subset dataset -> compare mean subset vs full dataset.
            full_df = self.all_plugins[
                PluginVariants.DATASET_HISTOGRAM_PROCESSOR
            ].compute_spike_count_distribution(
                is_test=is_test,
                process_subset=False,
                format_to_latex=False,
            )
            subset_df = self.all_plugins[
                PluginVariants.DATASET_HISTOGRAM_PROCESSOR
            ].compute_spike_count_distribution(
                is_test=is_test,
                process_subset=True,
                format_to_latex=False,
            )

            full_df = full_df[full_df["time_step"] == 20]

            return self.all_plugins[
                PluginVariants.DATASET_HISTOGRAM_PROCESSOR
            ].summarize_global_density_vs_full(
                subset_df,
                full_df,
                format_to_latex=format_to_latex,
            )

        return self.all_plugins[
            PluginVariants.DATASET_HISTOGRAM_PROCESSOR
        ].compute_spike_count_distribution(
            is_test=is_test,
            process_subset=process_subset,
            format_to_latex=format_to_latex,
        )

    def get_synchrony_spearman_correlation(
        self,
        is_test: bool = False,
        process_subset: bool = False,
        return_latex: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """
        Computes Spearman rank correlation of each layer across time bins
        to determine the correlation of synchrony rise with time bin size.

        :param is_test: Whether compute for test dataset, else train.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :param return_latex: Whether return LaTeX table representation, else pd.Dataframe.
        :return: Returns correlations and p-values for each layer and overall across all layers.
        """
        return self.all_plugins[
            PluginVariants.SYNCHRONY_TIME_BINS_PROCESSOR
        ].compute_synchrony_spearman_correlation(
            is_test=is_test, process_subset=process_subset, return_latex=return_latex
        )

    def get_synchrony_summary(
        self,
        is_test: bool = False,
        process_subset: bool = False,
        return_latex: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """
        Computes mean and variance of synchrony of each layer across time bins.

        :param is_test: Whether compute for test dataset, else train.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :param return_latex: Whether return LaTeX table representation, else pd.Dataframe.
        :return: Returns mean and variance for each layer and overall across all layers.
        """
        return self.all_plugins[
            PluginVariants.SYNCHRONY_TIME_BINS_PROCESSOR
        ].compute_synchrony_summary(
            is_test=is_test, process_subset=process_subset, return_latex=return_latex
        )


if __name__ == "__main__":
    EVALUATION_RESULTS_BASE = "/analysis_results"
    analysis_paths = {
        EvaluationProcessorChoices.FULL_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.FULL_DATASET_ANALYSIS.value}/",
        EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS.value}/",
        # EvaluationProcessorChoices.WANDB_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.WANDB_ANALYSIS.value}/",
        EvaluationProcessorChoices.PREDICTION_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.PREDICTION_ANALYSIS.value}/",
    }
    result_analyzer = ResultAnalyzer(analysis_paths)

    subset_df = result_analyzer.prepare_dataframe_for_plot(
        variant=PlottingVariants.SUBSET_TEMPORAL_SPIKE_DISTRIBUTION
    )
