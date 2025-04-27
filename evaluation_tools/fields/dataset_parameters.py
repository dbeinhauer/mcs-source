"""
This script defines structure of datasets used in the analysis.
"""

from evaluation_tools.fields.dataset_analyzer_fields import (
    DatasetVariantField,
    DatasetDimensions,
)

# All time binning variants that we want to analyze.
ALL_TIME_STEP_VARIANTS = [1, 5, 10, 15, 20]

# All subset ids that we want to analyze.
ALL_SUBSET_IDS = range(0, 20)

# Size of the dataset that we want to analyze:
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
