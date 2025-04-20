"""
This script defines all experiment analyses for plotting and other manipulation.
"""

from enum import Enum


class PlottingVariants(Enum):
    """
    All variants of plots.
    """

    TIME_BIN_COUNT_RATIO = "time_bin_count_ratio"
    TEMPORAL_SPIKE_DISTRIBUTION = "temporal_spike_distribution"
    CORRELATION_MATRIX_BIN_SIZE_TEMPORAL_DATASET = "correlation_bin_size_temporal"


class PluginVariants(Enum):
    """
    All plugins for analyses results processing.
    """

    DATASET_HISTOGRAM_PROCESSOR = "dataset_histogram_processor"
    TEMPORAL_EVOLUTION_PROCESSOR = "temporal_evolution_processor"
