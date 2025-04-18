"""
This script defines structure of datasets used in the analysis.
"""

from evaluation_tools.fields.dataset_analyzer_fields import (
    DatasetVariantField,
    DatasetDimensions,
)

ALL_TIME_STEP_VARIANTS = [1, 5, 10, 15, 20]
ALL_SUBSET_IDS = range(0, 20)

DATASET_SIZES = {
    DatasetVariantField.TRAIN: {
        DatasetDimensions.EXPERIMENT: 50000,
        DatasetDimensions.TRIAL: 1,
    },
    DatasetVariantField.TEST: {
        DatasetDimensions.EXPERIMENT: 900,
        DatasetDimensions.TRIAL: 20,
    },
}
