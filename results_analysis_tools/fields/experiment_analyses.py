"""
This script defines all experiment analyses for plotting and other manipulation.
"""

from enum import Enum


class PlottingVariants(Enum):
    """
    All variants of plots.
    """

    TIME_BIN_COUNT_RATIO = "time_bin_count_ratio"


class PluginVariants(Enum):
    """
    All plugins for analyses results processing.
    """

    DATASET_HISTOGRAM_PROCESSOR = "dataset_histogram_processor"
