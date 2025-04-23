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
)
from results_analysis_tools.fields.experiment_analyses import PlottingVariants


class ResultsPlotter:
    """
    This class encapsulates all plotting logic.
    """

    plotting_map = {
        PlottingVariants.FULL_TIME_BIN_COUNT_RATIO: plot_dataset_variant_all_time_bins_full,
        PlottingVariants.FULL_TEMPORAL_SPIKE_DISTRIBUTION: plot_temporal_spike_distribution_for_dataset_data_full,
        PlottingVariants.FULL_CORRELATION_MATRIX_BIN_SIZE_TEMPORAL_DATASET: plot_temporal_spiking_correlation_heatmap_for_each_bin_size_full,
        PlottingVariants.FULL_SYNCHRONY_TIME_BINS: plot_synchrony_boxplot_across_layers_full,
        PlottingVariants.SUBSET_TEMPORAL_SPIKE_DISTRIBUTION: plot_temporal_spike_distribution_comparison_full_subset,
        PlottingVariants.SUBSET_SYNCHRONY_TIME_BINS: plot_synchrony_boxplot_jitter_subset_full_comparison,
        PlottingVariants.MODEL_TYPES_CORRELATION_COMPARISON: plot_model_types_pearson_normalized_box_plot,
        PlottingVariants.MODEL_TYPES_P_VALUES_HEATMAP: plot_model_variant_p_value_heatmap_cc_norm,
        PlottingVariants.SEPARATE_TEMPORAL_BEHAVIOR_TARGET_PREDICTION: plot_single_model_synchrony_curves_across_layers,
    }

    @staticmethod
    def plot(
        data: pd.DataFrame,
        plot_variant: PlottingVariants,
        save_fig: str = "",
        kwargs: Dict = {},
        # is_test: bool = False,
    ):
        ResultsPlotter.plotting_map[plot_variant](data, save_fig=save_fig, **kwargs)
