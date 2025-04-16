"""
This script defines all variants for the analysis of the model predictions.
"""

from enum import Enum
from nn_model.type_variants import EvaluationFields


class BatchSummaryFields(Enum):
    """
    All statistical summary for batched evaluation results.
    """

    MSE = "mse"
    NEURON_PEARSON = "neuron_pearson"
    # Synchrony curve in time.
    SYNCHRONY = "synchrony"
    # MSE of the synchrony curves.
    MSE_SYNCHRONY = "mse_synchrony"
    # Drift in time between classical predictions and teacher-forced ones.
    DRIFT_FREE_FORCED = "drift_free_forced"

    # Counts of the predictions, targets and teacher-forced predictions
    # across the time bins for plotting the model dynamics.
    PREDICTION_TIME_COUNT = "prediction_time_count"
    TARGET_TIME_COUNT = "target_time_count"
    TRAIN_LIKE_TIME_COUNT = "train_like_time_count"


class PredictionDimensions(Enum):
    """
    Dimensions of the model predictions (differ from the dataset dimensions).
    """

    EXPERIMENT = 0
    TIME_STEP = 1
    NEURON = 2


class EvaluationPairsVariants(Enum):
    """
    All paired variants of evaluation pairs Variants.
    """

    PREDICTION_TARGET_PAIR = (EvaluationFields.PREDICTIONS, EvaluationFields.TARGETS)
    PREDICTION_TRAIN_LIKE_PAIR = (
        EvaluationFields.PREDICTIONS,
        EvaluationFields.TRAIN_LIKE_PREDICTION,
    )
    TRAIN_LIKE_TARGET_PAIR = (
        EvaluationFields.TRAIN_LIKE_PREDICTION,
        EvaluationFields.TARGETS,
    )
