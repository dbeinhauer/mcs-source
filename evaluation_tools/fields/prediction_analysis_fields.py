"""
This script defines all variants for the analysis of the model predictions.
"""

from typing import Union
from enum import Enum
from nn_model.type_variants import EvaluationFields


class BatchSummaryFields(Enum):
    """
    All statistical summary for batched evaluation results.
    """

    MSE = "mse"
    PEARSON = "pearson"
    # Synchrony curve in time.
    SYNCHRONY = "synchrony"
    # MSE of the synchrony curves.
    MSE_SYNCHRONY = "mse_synchrony"
    # Pearson's CC of synchrony curves.
    PEARSON_SYNCHRONY = "person_synchrony"
    # Drift in time between classical predictions and teacher-forced ones.
    DRIFT_FREE_FORCED = "drift_free_forced"
    # Drift delta - how much does the drift change in each time step.
    DRIFT_DELTA = "drift_delta"


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


# All field defining different evaluation results in either pair of separately
# that have been used to compute specific metric.
PredictionAnalysisVariants = Union[EvaluationFields, EvaluationPairsVariants]


class BatchJobParameters(Enum):
    """
    All parameters needed for the batch job.
    """

    # While single evaluation -> the fields to be taken.
    EVALUATION_FIELDS_TO_COMPUTE = "evaluation_fields_to_compute"
    # While paired-wise evaluation metric -> all pairs to be take.
    PAIRS_TO_COMPUTE = "pairs_to_compute"
    # Summary function to be applied on the selected batch.
    FUNCTION_TO_APPLY = "function_to_apply"
