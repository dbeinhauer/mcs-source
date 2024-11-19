"""
This script defines all enums of variants (layers, weights, models, etc.) used in the model.
"""

from enum import Enum


class LayerType(Enum):
    """
    All possible layers.
    """

    X_ON = "X_ON"
    X_OFF = "X_OFF"
    V1_EXC_L4 = "V1_Exc_L4"
    V1_INH_L4 = "V1_Inh_L4"
    V1_EXC_L23 = "V1_Exc_L23"
    V1_INH_L23 = "V1_Inh_L23"


class TimeStepVariant(Enum):
    """
    All possible time steps used in the model.
    """

    PREVIOUS = "previous"
    CURRENT = "current"


class WeightTypes(Enum):
    """
    All possible weight types.
    """

    EXCITATORY = "exc"
    INHIBITORY = "inh"


class ModelTypes(Enum):
    """
    All possible model types.
    """

    SIMPLE = "simple"
    COMPLEX = "complex"


class MetricTypes(Enum):
    """
    All possible metric types.
    """

    CC_NORM = "cc_norm"


class EvaluationFields(Enum):
    """
    All evaluation fields names.
    """

    PREDICTIONS = "predictions"
    TARGETS = "targets"
    RNN_PREDICTIONS = "rnn_predictions"
