"""
This script defines class that encapsulates all the logic for the plotting.
"""

from typing import Dict

import pandas as pd

from results_analysis_tools.plotting.time_bin_count_distribution import (
    plot_dataset_variant_all_time_bins_full,
)
from results_analysis_tools.plotting.temporal_spike_distribution import (
    plot_temporal_spike_distribution_for_dataset_data_full,
    plot_temporal_spiking_correlation_heatmap_for_each_bin_size_full,
    plot_temporal_spike_distribution_comparison_full_subset,
)
from results_analysis_tools.plotting.synchrony_comparison import (
    plot_synchrony_boxplot_across_layers_full,
    plot_synchrony_boxplot_jitter_subset_full_comparison,
)
from results_analysis_tools.plotting.model_variants_comparison import (
    plot_model_types_pearson_normalized_box_plot,
    plot_model_variant_p_value_heatmap_cc_norm,
)
from results_analysis_tools.plotting.synchrony_curve_plotting import (
    plot_single_model_synchrony_curves_across_layers,
    plot_multiple_models_teacher_forced,
    plot_pearson_boxplot_synchrony_overall,
    plot_pearson_synchrony_boxplot_layers,
)
from results_analysis_tools.plotting.drift_teacher_forced_free import (
    plot_teacher_forced_free_drift,
)
from results_analysis_tools.plotting.train_size_dependency import (
    plot_dataset_size_dependency_on_norm_cc,
)
from results_analysis_tools.fields.experiment_analyses import PlottingVariants


class ResultsPlotter:
    """
    This class encapsulates all plotting logic.
    """

    # Map of different plot variants and their appropriate plotting functions.
    plotting_map = {
        # Full Dataset on different time bins:
        PlottingVariants.FULL_TIME_BIN_COUNT_RATIO: plot_dataset_variant_all_time_bins_full,
        PlottingVariants.FULL_TEMPORAL_SPIKE_DISTRIBUTION: plot_temporal_spike_distribution_for_dataset_data_full,
        PlottingVariants.FULL_CORRELATION_MATRIX_BIN_SIZE_TEMPORAL_DATASET: plot_temporal_spiking_correlation_heatmap_for_each_bin_size_full,
        PlottingVariants.FULL_SYNCHRONY_TIME_BINS: plot_synchrony_boxplot_across_layers_full,
        # Subset datasets:
        PlottingVariants.SUBSET_TEMPORAL_SPIKE_DISTRIBUTION: plot_temporal_spike_distribution_comparison_full_subset,
        PlottingVariants.SUBSET_SYNCHRONY_TIME_BINS: plot_synchrony_boxplot_jitter_subset_full_comparison,
        # Overall models analysis:
        PlottingVariants.MODEL_TYPES_CORRELATION_COMPARISON: plot_model_types_pearson_normalized_box_plot,
        PlottingVariants.MODEL_TYPES_P_VALUES_HEATMAP: plot_model_variant_p_value_heatmap_cc_norm,
        PlottingVariants.MODEL_TYPES_SYNCHRONY_PEARSON_OVERALL_PEARSON: plot_pearson_boxplot_synchrony_overall,
        PlottingVariants.MODEL_TYPES_SYNCHRONY_PEARSON_LAYERS: plot_pearson_synchrony_boxplot_layers,
        # Separate model analysis:
        PlottingVariants.SEPARATE_TEMPORAL_BEHAVIOR_TARGET_PREDICTION: plot_single_model_synchrony_curves_across_layers,
        # Additional analyses:
        PlottingVariants.TBPTT_MODELS_TEMPORAL_BEHAVIOR_TEACHER_FORCED_INCLUDED: plot_multiple_models_teacher_forced,
        PlottingVariants.DRIFT_TEACHER_FORCED_FREE_TEMPORAL: plot_teacher_forced_free_drift,
        PlottingVariants.TRAIN_SUBSET_SIZE_ON_NORM_CC: plot_dataset_size_dependency_on_norm_cc,
    }

    @staticmethod
    def plot(
        data: pd.DataFrame,
        plot_variant: PlottingVariants,
        save_fig: str = "",
        kwargs: Dict = {},
    ):
        """
        Plots provided data in selected plot.

        :param data: Data to plot.
        :param plot_variant: What plot we want to use.
        :param save_fig: Whether we want to save the figure, if `""` then not save.
        :param kwargs: Additional plotting kwargs.
        """
        ResultsPlotter.plotting_map[plot_variant](data, save_fig=save_fig, **kwargs)
