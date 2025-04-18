"""
This script serves as main part that loads and processes the analysis results.
"""

from typing import Dict, Any, List
import pickle
import re
import os
from enum import Enum

from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    HistogramFields,
    StatisticsFields,
    DatasetVariantField,
    DatasetDimensions,
)

from evaluation_tools.fields.dataset_parameters import ALL_TIME_STEP_VARIANTS
from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.scripts.pickle_manipulation import load_pickle_file

import nn_model.globals


class ResultAnalyzer:
    """
    This class serves for statistical analysis of the processed results.
    """

    def __init__(self, results_to_load: List[EvaluationProcessorChoices]):

        # What processed evaluation results should we load.
        self.results_to_load = results_to_load

        self.all_results: Dict[EvaluationProcessorChoices, Any] = {}

    @property
    def get_all_results(self) -> Dict[EvaluationProcessorChoices, Any]:
        return self.all_results

    def load_analysis_results(
        self,
        base_dir: str,
        results_variant: EvaluationProcessorChoices,
    ) -> Dict[DatasetVariantField, Dict[int, Dict[AnalysisFields, Any]]]:
        """
        Loads either all dataset analysis results on:
            1. The full dataset with multiple time bins variants.
            2. The subset dataset with multiple subset variants.

        Stores the results to dictionary that captures all loaded results.

        :param base_dir: Base directory with all target results.
        :param results_variant: Variant of the results to load.
        :return: Returns loaded results.
        """
        results: Dict[DatasetVariantField, Dict[int, object]] = {}
        pattern = re.compile(rf"{results_variant.value}-(.+)-(\d+)\.pkl")

        for filename in os.listdir(base_dir):
            match = pattern.match(filename)
            if match:
                # The file is one of the analysis results files.
                variant_str, time_or_subset_str = match.groups()
                time_or_subset = int(time_or_subset_str)

                # Convert string to enum
                try:
                    variant = DatasetVariantField(variant_str)
                except ValueError:
                    print(f"Skipping unknown variant: {variant_str}")
                    continue

                if variant not in results:
                    # Initialize the variant.
                    results[variant] = {}

                # Load the data.
                file_path = os.path.join(base_dir, filename)
                value = load_pickle_file(file_path)

                results[variant][time_or_subset] = value

        # Save the results to all results.
        self.all_results[results_variant] = results

        return results


if __name__ == "__main__":
    EVALUATION_RESULTS_BASE = "/evaluation_tools/evaluation_results"
    analysis_paths = {
        EvaluationProcessorChoices.FULL_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/full_dataset_analysis/",
        EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/subset_dataset_analysis/",
    }
    result_analyzer = ResultAnalyzer([variant for variant in analysis_paths])
    results_full_analysis = result_analyzer.load_analysis_results(
        analysis_paths[EvaluationProcessorChoices.FULL_DATASET_ANALYSIS],
        EvaluationProcessorChoices.FULL_DATASET_ANALYSIS,
    )
    results_subset_analysis = result_analyzer.load_analysis_results(
        analysis_paths[EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS],
        EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS,
    )

    all_results = result_analyzer.get_all_results
    print(all_results)

    # print(results.keys())
