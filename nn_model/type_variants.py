"""
This script defines all enums of variants (layers, weights, models, etc.) used in the model.
"""

from enum import Enum


class LayerParent(Enum):
    """
    All possible layers.
    """

    LGN = "LGN"
    L4 = "L4"
    L23 = "L23"


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


class ModelModulesFields(Enum):
    """
    All fields of the model modules.
    """

    NEURON_MODULE = "neuron_module"
    SYNAPTIC_ADAPTION_MODULE = "synaptic_adaption_module"


class ModelTypes(Enum):
    """
    All possible model types.
    """

    SIMPLE = "simple"
    DNN_JOINT = "dnn_joint"
    DNN_SEPARATE = "dnn_separate"
    RNN_JOINT = "rnn_joint"
    RNN_SEPARATE = "rnn_separate"


class LossTypes(Enum):
    """
    All possible loss types.
    """

    MSE = "mse"
    POISSON = "poisson"


class RNNTypes(Enum):
    """
    All possible RNN types.
    """

    LSTM = "lstm"
    GRU = "gru"


class NeuronActivationTypes(Enum):
    """
    All possible neuron activation types.
    """

    LEAKYTANH = "leakytanh"
    SIGMOIDTANH = "sigmoidtanh"


class MetricTypes(Enum):
    """
    All possible metric types.
    """

    CC_NORM = "cc_norm"


class OptimizerTypes(Enum):
    """
    All possible optimizer types.
    """

    DEFAULT = "default"
    EXC_INH_SPECIFIC = "exc_inh_specific"


class WeightsInitializationTypes(Enum):
    """
    All possible weights initialization variants.
    """

    DEFAULT = "default"
    NORMAL = "normal"


class LayerConstraintFields(Enum):
    """
    All fields of the layer constraints (to determine excitatory/inhibitory layers).
    """

    SIZE = "size"
    TYPE = "type"
    NAME = "name"
    TIMESTEP = "timestep"


class PredictionTypes(Enum):
    """
    All variants of predictions that are provided by the model
    (to distinguish between the predictions).
    """

    # General model prediction -> full sequence prediction (without resetting).
    FULL_PREDICTION = "full_prediction"
    # Predictions where we reset the hidden states based on the target
    # (same approach as in train).
    TRAIN_LIKE_PREDICTION = "train_like_prediction"
    # Predictions of the outer RNN (not working currently).
    RNN_PREDICTION = "rnn_prediction"


class NeuronModulePredictionFields(Enum):
    """
    All keys of the predictions of the DNN neuron module.
    """

    INPUT = "input"
    OUTPUT = "output"


class EvaluationFields(Enum):
    """
    All evaluation fields names for storing the results.
    """

    PREDICTIONS = "predictions"
    TRAIN_LIKE_PREDICTION = "train_like_predictions"
    TARGETS = "targets"
    RNN_PREDICTIONS = "rnn_predictions"


class PathDefaultFields(Enum):
    """
    All input paths fields used in the model
    """

    TRAIN_DIR = "train_dir"
    TEST_DIR = "test_dir"
    SUBSET_DIR = "subset_dir"
    MODEL_DIR = "model_dir"
    EXPERIMENT_SELECTION_PATH = "experiment_selection_path"
    NEURON_SELECTION_PATH = "neuron_selection_path"
    SELECTION_RESULTS_DIR = "selection_results_dir"
    FULL_EVALUATION_DIR = "full_evaluation_dir"
    NEURON_MODEL_RESPONSES_DIR = "neuron_model_responses_dir"
    VISIBLE_NEURONS_DIR = "visible_neurons_dir"


class PathPlotDefaults(Enum):
    """
    All default paths fields where to store the plots.
    """

    NEURON_MODULE_SEPARATE = "neuron_module_separate"
    NEURON_MODULE_TOGETHER = "neuron_module_together"
    MEAN_LAYER_RESPONSES = "mean_layer_responses"


class EvaluationMeanVariants(Enum):
    """
    All variants of evaluation mean.
    """

    LAYER_MEAN = "layer_mean"
    NEURON_MEAN = "neuron_mean"
    IMAGE_MEAN = "image_mean"


class EvaluationMetricVariants(Enum):
    """
    All variants of evaluated subsets.
    """

    FULL_METRIC = "full_metric"
    VISIBLE_METRIC = "visible_metric"
    INVISIBLE_METRIC = "invisible_metric"
    LAYER_SPECIFIC = "layer_specific_metric"
