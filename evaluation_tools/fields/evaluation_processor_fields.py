"""
This script defines all fields used for the evaluation processing.
"""

from enum import Enum


class EvaluationProcessorChoices(Enum):
    """
    All variants of actions for `EvaluationProcessor` class.
    """

    # Evaluate full dataset (mainly time binning focused).
    FULL_DATASET_ANALYSIS = "full_dataset"
    # Evaluate provided neuron subsets (typically of time bins 20).
    SUBSET_DATASET_ANALYSIS = "subset_dataset"
