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
    def get_time_step_20_full(df: pd.DataFrame, time_step: int = 20) -> pd.DataFrame:
        """
        :param df: Dataframe to take the analysis.
        :param tame_step: Time step to take the analysis (optionally other than 20 ms).
        :return: Returns dataframe rows that match the selected time step (typically 20 ms).
        """
        return df[df["time_step"] == time_step]

    @staticmethod
    def replace_time_step_with_subset_full(
        df: pd.DataFrame, subset_id: str = "-1"
    ) -> pd.DataFrame:
        """
        :param df: Full dataset Dataframe to take the analysis.
        :param subset_id: Subset id to assign to the full dataset (typically `"-1"`).
        :return: Returns dataframe with "time_step" column replaced with "subset_id" column.
        """
        df = df.rename(columns={"time_step": "subset_id"})
        df["subset_id"] = "-1"
        return df[df["subset_id"] == subset_id]

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

    @staticmethod
    def merge_full_and_subset_dataframes(
        full_df: pd.DataFrame, subset_df: pd.DataFrame, full_id: str = "-1"
    ) -> pd.DataFrame:
        """
        Merges the full and subset dataframes into a single dataframe. Differentiates
        between full and subset data by adding a new column `model_type` that indicates
        whether the data is from the full model or the subset.

        :param full_df: Full dataset preprocessed dataframe.
        :param subset_df: Subset dataset preprocessed dataframe.
        :return: Returns the combined dataframe with an additional column `model_type`
        indicating the type of model (full or subset).
        """
        combined_df = pd.concat([subset_df, full_df], ignore_index=True)
        # Ensure subset_id is string
        combined_df["subset_id"] = combined_df["subset_id"].astype(str)

        # Create a new column to distinguish full vs subset for plotting
        combined_df["model_type"] = combined_df["subset_id"].apply(
            lambda x: "Full model" if x == "-1" else "Subset"
        )

        return combined_df

    @staticmethod
    def ensure_layer_order(df: pd.DataFrame) -> pd.DataFrame:
        """
        :param df: Dataframe to modify.
        :return: Returns dataframe with the layer column ordered in a specific way for plotting.
        """
        desired_order = [
            "X_ON",
            "X_OFF",
            "V1_Exc_L4",
            "V1_Inh_L4",
            "V1_Exc_L23",
            "V1_Inh_L23",
        ]

        # Ensure the order of the layers.
        df["layer"] = pd.Categorical(
            df["layer"], categories=desired_order, ordered=True
        )
        return df
