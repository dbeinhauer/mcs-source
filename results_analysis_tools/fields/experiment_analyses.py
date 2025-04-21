"""
This script defines all experiment analyses for plotting and other manipulation.
"""

from enum import Enum


class PlottingVariants(Enum):
    """
    All variants of plots.
    """

    # Full dataset analyses (for all time bins).
    FULL_TIME_BIN_COUNT_RATIO = "full_time_bin_count_ratio"
    FULL_TEMPORAL_SPIKE_DISTRIBUTION = "full_temporal_spike_distribution"
    FULL_CORRELATION_MATRIX_BIN_SIZE_TEMPORAL_DATASET = (
        "full_correlation_bin_size_temporal"
    )
    FULL_SYNCHRONY_TIME_BINS = "full_synchrony_time_bins"

    # Subset dataset analyses (for all model subset variants).
    SUBSET_TEMPORAL_SPIKE_DISTRIBUTION = "subset_temporal_spike_distribution"


class PluginVariants(Enum):
    """
    All plugins for analyses results processing.
    """

    DATASET_HISTOGRAM_PROCESSOR = "dataset_histogram_processor"
    TEMPORAL_EVOLUTION_PROCESSOR = "temporal_evolution_processor"
    SYNCHRONY_TIME_BINS_PROCESSOR = "synchrony_time_bins_processor"
