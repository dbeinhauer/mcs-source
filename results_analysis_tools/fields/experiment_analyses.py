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
    SUBSET_SYNCHRONY_TIME_BINS = "subset_synchrony_time_bins"

    # Evaluation results overview.
    MODEL_TYPES_CORRELATION_COMPARISON = "model_types_correlation_comparison"
    MODEL_TYPES_P_VALUES_HEATMAP = "model_types_p_values_heatmap"

    # Evaluation results separate model plotting:
    SEPARATE_TEMPORAL_BEHAVIOR_TARGET_PREDICTION = (
        "separate_temporal_behavior_target_prediction"
    )
    TBPTT_MODELS_TEMPORAL_BEHAVIOR_TEACHER_FORCED_INCLUDED = (
        "tbptt_models_temporal_behavior_teacher_forced_included"
    )


class PluginVariants(Enum):
    """
    All plugins for analyses results processing.
    """

    DATASET_HISTOGRAM_PROCESSOR = "dataset_histogram_processor"
    TEMPORAL_EVOLUTION_PROCESSOR = "temporal_evolution_processor"
    SYNCHRONY_TIME_BINS_PROCESSOR = "synchrony_time_bins_processor"
    WANDB_SUMMARY_PROCESSOR = "wandb_summary_processor"
    BATCH_PREDICTION_PROCESSOR = "batch_prediction_processor"
