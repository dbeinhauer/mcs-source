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

import nn_model.globals


class ResultAnalyzer:
    """
    This class serves for statistical analysis of the processed results.
    """

    def __init__(self, results_to_load: List[EvaluationProcessorChoices]):

        # What processed evaluation results should we load.
        self.results_to_load = results_to_load

        self.all_results: Dict[EvaluationProcessorChoices, Any] = {}

    @staticmethod
    def load_pickle_file(filename: str):
        """
        Loads pickle file.

        :param filename: Name of the pickle file.
        :return: Returns content of the pickle file.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def store_pickle_file(filename: str, data_to_store):
        """
        Stored data to pickle file.

        :param filename: Filename.
        :param data_to_store: Data to be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(data_to_store, f)
        print(f"Data saved to {filename}")

    def load_analysis_results(
        self,
        directory: str,
    ) -> Dict[DatasetVariantField, Dict[int, Dict[AnalysisFields, Any]]]:

        results: Dict[DatasetVariantField, Dict[int, object]] = {}
        pattern = re.compile(
            rf"{ResultVariants.FULL_DATASET_ANALYSIS.value}-(.+)-(\d+)\.pkl"
        )

        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                variant_str, timestep_str = match.groups()
                timestep = int(timestep_str)

                # Convert string to enum
                try:
                    variant = DatasetVariantField(variant_str)
                except ValueError:
                    print(f"Skipping unknown variant: {variant_str}")
                    continue

                # Insert into nested dictionary
                if variant not in results:
                    results[variant] = {}

                file_path = os.path.join(directory, filename)
                value = ResultAnalyzer.load_pickle_file(file_path)

                results[variant][timestep] = value

        # Save the results to all results.
        self.all_results[ResultVariants.FULL_DATASET_ANALYSIS] = results

        return results


if __name__ == "__main__":
    EVALUATION_RESULTS_BASE = "/evaluation_tools/evaluation_results"
    analysis_paths = {
        ResultVariants.FULL_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/full_dataset_analysis/"
    }
    result_analyzer = ResultAnalyzer([variant for variant in analysis_paths])
    results = result_analyzer.load_analysis_results(
        analysis_paths[ResultVariants.FULL_DATASET_ANALYSIS]
    )

    print(results.keys())
