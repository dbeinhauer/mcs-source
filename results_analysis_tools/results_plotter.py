"""
This script defines class that encapsulates all the logic for the plotting.
"""

import pandas as pd

from results_analysis_tools.plotting.time_bin_count_distribution import (
    plot_dataset_variant_all_time_bins,
)
from results_analysis_tools.plotting.temporal_spike_distribution import (
    plot_temporal_spike_distribution_for_dataset_data,
    plot_temporal_spiking_correlation_heatmap_for_each_bin_size,
)
from results_analysis_tools.plotting.synchrony_comparison import (
    plot_synchrony_boxplot_across_layers,
)
from results_analysis_tools.fields.experiment_analyses import PlottingVariants


class ResultsPlotter:
    """
    This class encapsulates all plotting logic.
    """

    plotting_map = {
        PlottingVariants.TIME_BIN_COUNT_RATIO: plot_dataset_variant_all_time_bins,
        PlottingVariants.TEMPORAL_SPIKE_DISTRIBUTION: plot_temporal_spike_distribution_for_dataset_data,
        PlottingVariants.CORRELATION_MATRIX_BIN_SIZE_TEMPORAL_DATASET: plot_temporal_spiking_correlation_heatmap_for_each_bin_size,
        PlottingVariants.SYNCHRONY_TIME_BINS: plot_synchrony_boxplot_across_layers,
    }

    @staticmethod
    def plot(
        data: pd.DataFrame,
        plot_variant: PlottingVariants,
        save_fig: str = "",
        is_test: bool = False,
    ):
        ResultsPlotter.plotting_map[plot_variant](
            data, is_test=is_test, save_fig=save_fig
        )
