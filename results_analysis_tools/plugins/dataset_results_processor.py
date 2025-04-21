"""
This module encapsulates the logic shared for majority of the dataset processing plugins.
"""

from typing import Dict

import numpy as np
import pandas as pd

from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    DatasetVariantField,
)


class DatasetResultsProcessor:

    @staticmethod
    def get_full_dataset_variant(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        :param all_results: All analyses results dictionary.
        :return: Returns full dataset analyses.
        """
        return all_results[EvaluationProcessorChoices.FULL_DATASET_ANALYSIS]

    @staticmethod
    def get_subset_dataset_variant(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        :param all_results: All analyses results dictionary.
        :return: Returns subset dataset analyses.
        """
        return all_results[EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS]

    @staticmethod
    def get_analysis_type(
        df: pd.DataFrame, analysis_type: AnalysisFields
    ) -> pd.DataFrame:
        """
        :param df: Dataframe to take the analysis.
        :param analysis_type: Type of the analysis on the dataset.
        :return: Returns dataframe rows that match the selected analysis.
        """
        return df[df["analysis_type"] == analysis_type]

    @staticmethod
    def get_dataset_type(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """
        :param df: Dataframe to take the analysis.
        :param is_test: Whether to take test dataset.
        :return: Returns dataframe rows that match the selected dataset variant (either train or test).
        """
        variant_to_take = (
            DatasetVariantField.TEST if is_test else DatasetVariantField.TRAIN
        )
        return df[df["dataset_variant"] == variant_to_take]

    @staticmethod
    def reformat_dataset_dataframe(
        original_df: pd.DataFrame,
        function_to_process_results,
        process_subset: bool = False,
    ) -> pd.DataFrame:
        """
        Takes original dataframe of analysis results on the dataset and converts it to the format
        with columns `[{"time_step"|"subset_id"}, "layer", "analysis_results"]`. Where `"analysis_results"`
        are the specific results for each different analysis type from `AnalysisFields`.

        :param original_df: Original analysis results from evaluation tools.
        :param function_to_process_results: Function to be called on each of the row of the
        `"analysis_results"` column that should retrieve the information of our interest as dictionary
        pairs for the new dataframe.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :return: Returns reformated dataframe with columns (`"subset_id"` or `"time_step"` depends on `process_subset`),
        `[{"time_step"|"subset_id"}, "layer", {columns_from_provided_function}]`.
        """

        flat_data = []
        time_step_or_subset_id_column_key = (
            "time_step" if not process_subset else "subset_id"
        )

        for _, row in original_df.iterrows():
            time_step_or_subset_id = row[time_step_or_subset_id_column_key]
            analysis_results = row["analysis_results"]

            if isinstance(analysis_results, pd.DataFrame):
                for _, subrow in analysis_results.iterrows():
                    layer = subrow["layer_name"]
                    # Ensure it's a string
                    if isinstance(layer, (list, np.ndarray)):
                        layer = str(layer[0])  # or use .item() if it's 0-dim
                    else:
                        layer = str(layer)

                    flat_data.append(
                        {
                            time_step_or_subset_id_column_key: time_step_or_subset_id,
                            "layer": layer,
                            **function_to_process_results(subrow),
                        }
                    )

        return pd.DataFrame(flat_data)
