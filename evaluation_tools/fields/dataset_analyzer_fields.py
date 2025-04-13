"""
This script defines fields definitions used in the
DatasetAnalyzer plugin.
"""

from enum import Enum


class AnalysisFields(Enum):
    """
    Keys of the all analysis fields.
    """

    # Histogram of neuron firing rates.
    HISTOGRAM_NEURON_SPIKE_RATES = "histogram_neuron_spike_rates"
    # Histogram of number of spikes across all time bin.
    HISTOGRAM_TIME_BIN_SPIKE_RATES = "histogram_time_bin_spike_rates"
    # Counts of total spikes in each time bin (size as num_time_bins).
    TIME_BIN_SPIKE_COUNTS = "time_bin_spike_counts"
    # Counts of spikes in each trial and experiment.
    EXPERIMENT_SPIKE_COUNTS = "experiment_spike_counts"
    # Neuron spike rate mean and variance across experiment.
    NEURON_MEAN_VARIANCE_EXPERIMENT = "neuron_mean_variance_experiment"
    # Total number of spikes for neuron.
    NEURON_SPIKE_COUNT = "neuron_spike_count"
    # Fano factor across trials (only for test).
    FANO_TRIALS = "fano_trials"
    # Fano factor across the time bins.
    FANO_TIME_BINS = "fano_time_bins"
    # Temporal synchrony sum for each time bin.
    TEMPORAL_SYNCHRONY = "temporal_synchrony"
    # Per trial synchrony - without mean over trials.
    PER_TRIAL_SYNCHRONY = "per_trial_synchrony"


HISTOGRAM_ANALYSES = [
    AnalysisFields.HISTOGRAM_NEURON_SPIKE_RATES,
    AnalysisFields.HISTOGRAM_TIME_BIN_SPIKE_RATES,
]


class HistogramFields(Enum):
    """
    Fields for the histogram objects (just counts and bins).
    """

    NUM_BINS = "num_bins"
    COUNTS = "counts"
    BINS = "bins"
    SUMMING_FUNCTION = (
        "summing_function"  # Function to apply to prepare the counts data.
    )


class StatisticsFields(Enum):
    """
    Fields defining all statistics metrics used in the objects.
    """

    TOTAL_COUNT = "total_count"
    MEAN = "mean"
    VARIANCE = "variance"
    FANO_FACTOR = "fano_factor"
    SYNCHRONY = "synchrony"
