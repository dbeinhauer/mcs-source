"""
This script defines variants of experiments, setup and other.
"""

from typing import Union
from enum import Enum

# Entity from which take the weights and biases results.
WANDB_ENTITY = "david-beinhauer-charles-university"


class WandbExperimentVariants(Enum):
    """
    High-level variants of Weights and Biases experiments.
    """

    GRID_SEARCH = "grid_search"
    EVALUATION = "evaluation"
    ADDITIONAL = "additional"


class GridSearchRunVariants(Enum):
    """
    All variants of grid search.
    """

    # Simple mode variants with either tanh or leakytanh activation functions.
    SIMPLE_TANH = "simple_grid"
    SIMPLE_LEAKYTANH = "simple_grid_new"
    DNN = "dnn_grid"
    RNN = "rnn_grid"
    SYNAPTIC_ADAPTATION = "syn_adaptation_grid"


class ModelEvaluationRunVariant(Enum):
    """
    All evaluation variants of experiments with 20 subsets and same parameters.
    """

    # Simple mode variants with either tanh or leakytanh activation functions.
    SIMPLE_TANH = "simple_evaluation"
    SIMPLE_LEAKYTANH = "simple_evaluation_new"
    DNN_JOINT = "dnn_joint_evaluation"
    DNN_SEPARATE = "dnn_separate_evaluation"
    # RNN evaluation on different number of truncated backpropagation through time steps.
    RNN_BACKPROPAGATION_5 = "rnn_separate_5_evaluation"
    RNN_BACKPROPAGATION_10 = "rnn_separate_10_evaluation"
    # Synaptic adaptation on only LGN connections.
    SYN_ADAPT_LGN_BACKPROPAGATION_5 = "syn_only_lgn_5_evaluation"
    SYN_ADAPT_LGN_BACKPROPAGATION_10 = "syn_only_lgn_10_evaluation"


class AdditionalExperiments(Enum):
    """
    All additional experiments that are not grid search and evaluation runs.
    """

    # Experiment evaluating influence of dataset size on model performance.
    DATASET_SUBSET_SIZE = "subset_size"
    # Experiment evaluating influence of model subset size on model performance.
    MODEL_SIZES = "model_sizes"


AllWandbVariants = Union[
    GridSearchRunVariants, ModelEvaluationRunVariant, AdditionalExperiments
]

NUM_EVALUATION_SUBSETS = 20
NUM_EVALUATION_BATCHES = 90
