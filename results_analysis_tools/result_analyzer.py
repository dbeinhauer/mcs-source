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
from results_analysis_tools.plugins.wandb_summary_processor import WandbSummaryProcessor
from results_analysis_tools.plugins.batch_prediction_analysis_processor import (
    BatchPredictionAnalysisProcessor,
)
from evaluation_tools.fields.prediction_analysis_fields import (
    BatchSummaryFields,
    EvaluationPairsVariants,
)
from nn_model.type_variants import EvaluationFields


class ResultAnalyzer:
    """
    This class serves for statistical analysis of all processed results.
    """

    def __init__(self, results_to_load: Dict[EvaluationProcessorChoices, str]):

        # Loads all analysis results.
        self.results_loader = ResultsLoader(results_to_load)
        self.all_results = self.results_loader.load_all_results()

        # All plugins for results analysis:
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
            PluginVariants.WANDB_SUMMARY_PROCESSOR: WandbSummaryProcessor(
                self.all_results
            ),
            PluginVariants.BATCH_PREDICTION_PROCESSOR: BatchPredictionAnalysisProcessor(
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
        self,
        variant: PlottingVariants,
        is_test: bool = False,
        synchrony_curve_kwargs: Dict = {},
    ) -> pd.DataFrame:
        """
        Prepares pandas dataframe in format necessary for specified plot.

        :param variant: Plot variant that we want to prepare data for.
        :param is_test: Flag whether we want to process test dataset.
        :param synchrony_curve_kwargs: Kwargs used for synchrony curve data preparation.
        :return: Returns dataframe with prepared data that are ready
        to be plotted for specified variant.
        """

        # Dataset plotting time bins.
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

        # Dataset plotting model subsets.
        if variant == PlottingVariants.SUBSET_TEMPORAL_SPIKE_DISTRIBUTION:
            return self.all_plugins[
                PluginVariants.TEMPORAL_EVOLUTION_PROCESSOR
            ].prepare_for_plotting_subset_full_comparison(is_test=is_test)
        if variant == PlottingVariants.SUBSET_SYNCHRONY_TIME_BINS:
            return self.all_plugins[
                PluginVariants.SYNCHRONY_TIME_BINS_PROCESSOR
            ].prepare_for_plotting_subset_full_comparison(is_test=is_test)

        # Overall model evaluation.
        if variant == PlottingVariants.MODEL_TYPES_CORRELATION_COMPARISON:
            return self.all_plugins[
                PluginVariants.WANDB_SUMMARY_PROCESSOR
            ].prepare_model_comparison_summary_for_plotting()
        if variant == PlottingVariants.MODEL_TYPES_P_VALUES_HEATMAP:
            return self.all_plugins[
                PluginVariants.WANDB_SUMMARY_PROCESSOR
            ].mann_whitney_paired_evaluation_models_test_cc_norm()
        if variant == PlottingVariants.MODEL_TYPES_SYNCHRONY_PEARSON_OVERALL_PEARSON:
            return self.all_plugins[
                PluginVariants.BATCH_PREDICTION_PROCESSOR
            ].combine_overall_and_synchrony_pearson_for_plotting()
        if variant == PlottingVariants.MODEL_TYPES_SYNCHRONY_PEARSON_LAYERS:
            return self.all_plugins[
                PluginVariants.BATCH_PREDICTION_PROCESSOR
            ].prepare_pearson_cc_synchrony()

        # Separate model evaluation.
        if variant == PlottingVariants.SEPARATE_TEMPORAL_BEHAVIOR_TARGET_PREDICTION:
            return self.all_plugins[
                PluginVariants.BATCH_PREDICTION_PROCESSOR
            ].prepare_for_plotting_synchrony_curves(
                variants_to_plot=[
                    EvaluationFields.PREDICTIONS,
                    EvaluationFields.TARGETS,
                ],
                **synchrony_curve_kwargs,  # Kwargs specifying model and optionally layers.
            )
        if variant == PlottingVariants.DRIFT_TEACHER_FORCED_FREE_TEMPORAL:
            return self.all_plugins[
                PluginVariants.BATCH_PREDICTION_PROCESSOR
            ].prepare_for_free_forced_drift_plot()

        # Additional model analysis.
        if (
            variant
            == PlottingVariants.TBPTT_MODELS_TEMPORAL_BEHAVIOR_TEACHER_FORCED_INCLUDED
        ):
            return self.all_plugins[
                PluginVariants.BATCH_PREDICTION_PROCESSOR
            ].prepare_for_plotting_synchrony_curves(
                **synchrony_curve_kwargs,  # Kwargs specifying model and optionally layers.
            )
        if variant == PlottingVariants.TRAIN_SUBSET_SIZE_ON_NORM_CC:
            return self.all_plugins[
                PluginVariants.WANDB_SUMMARY_PROCESSOR
            ].prepare_for_plotting_dataset_size_dependency_on_cc_norm()

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

    def get_latex_evaluation_setup(self) -> str:
        """
        :return: Returns prepared LaTeX table representation of the model evaluation setup.
        """
        return self.all_plugins[
            PluginVariants.WANDB_SUMMARY_PROCESSOR
        ].prepare_latex_evaluation_setup_table()

    def get_evaluation_results_summary(
        self, return_latex: bool = False
    ) -> Union[pd.DataFrame, str]:
        """
        :param return_latex: Whether to return summary as LaTeX table.
        :return: Return model variant evaluation summary as table.
        """
        return self.all_plugins[
            PluginVariants.WANDB_SUMMARY_PROCESSOR
        ].prepare_evaluation_results_summary(return_latex=return_latex)

    def get_grid_search_summary_table(self) -> Dict[str, str]:
        """
        Prepares LaTeX table of grid search summary.

        :return: Grid search summary per model variant.
        """
        return self.all_plugins[
            PluginVariants.WANDB_SUMMARY_PROCESSOR
        ].generate_latex_gridsearch_tables_per_model()


if __name__ == "__main__":
    EVALUATION_RESULTS_BASE = "/analysis_results"
    analysis_paths = {
        EvaluationProcessorChoices.FULL_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.FULL_DATASET_ANALYSIS.value}/",
        EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS.value}/",
        EvaluationProcessorChoices.WANDB_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.WANDB_ANALYSIS.value}/results.pkl",
        EvaluationProcessorChoices.PREDICTION_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.PREDICTION_ANALYSIS.value}/",
    }
    result_analyzer = ResultAnalyzer(analysis_paths)
