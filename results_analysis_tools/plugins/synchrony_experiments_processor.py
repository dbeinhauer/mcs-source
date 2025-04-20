"""
This module encapsulates the logic for the processing of the results of
the dataset analyses on the separate experiments.
"""

from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
)
from results_analysis_tools.plugins.dataset_results_processor import (
    DatasetResultsProcessor,
)
from evaluation_tools.fields.dataset_analyzer_fields import StatisticsFields


class SynchronyExperimentsProcessor:
    """
    This class encapsulates the logic for the processing of the results of
    the dataset analyses on the separate experiments.
    """

    def __init__(self, all_results: Dict[EvaluationProcessorChoices, pd.DataFrame]):
        self.synchrony_full_dataset_df = SynchronyExperimentsProcessor._get_all_full_dataset_separate_experiment_results(
            all_results
        )

    @staticmethod
    def _get_all_full_dataset_separate_experiment_results(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Retrieve all separate experiment analyses from the full dataset results.

        :param all_results:
        :return: Dataframe of all separate experiment analyses from full dataset.
        """
        return DatasetResultsProcessor.get_analysis_type(
            DatasetResultsProcessor.get_full_dataset_variant(all_results),
            AnalysisFields.SEPARATE_EXPERIMENT_ANALYSIS,
        )

    @staticmethod
    def _get_synchrony(df: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieves synchrony from the dataframe.

        :param df: Dataframe of separate experiment results.
        :return: Dataframe with only synchrony results.
        """

        print(df.columns)

        return df[df["statistics_type"] == StatisticsFields.SYNCHRONY]

    @staticmethod
    def _separate_analysis_reformatting_row(row: pd.Series) -> Dict[str, Any]:
        """
        Function that selects only selected columns from original row for further analysis.

        :param row: Original dataframe row.
        :return: Returns "statistics_type" and "values" columns.
        """

        return {
            "statistics_type": row["statistics_type"],
            "values": row["values"],
        }

    @staticmethod
    def _reformat_original_separate_dataframe(
        original_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        :param original_df: Original dataframe as a results from analysis tools.
        :return: Dataframe reformated for easier manipulation while plotting.
        """
        return DatasetResultsProcessor.reformat_dataset_dataframe(
            original_df,
            SynchronyExperimentsProcessor._separate_analysis_reformatting_row,
        )

    @staticmethod
    def _create_long_format_for_seaborn(original_df: pd.DataFrame) -> pd.DataFrame:
        """
        :param original_df: Dataframe to convert.
        :return: Converts dataframe to long format (each value on its row) for plotting in seaborn.
        """
        long_rows = []

        df = original_df.copy()

        for _, row in df.iterrows():
            ts = row["time_step"]
            layer = row["layer"]
            sync_values = row["values"].flatten()  # flatten 2D array to 1D

            for val in sync_values:
                long_rows.append({"time_step": ts, "layer": layer, "synchrony": val})

        return pd.DataFrame(long_rows)

    def prepare_for_plotting(
        self, original_df: pd.DataFrame = None, is_test: bool = False
    ) -> pd.DataFrame:
        """
        Prepares time bin histogram data for plotting.

        :param original_df: Original histogram data, if `None` then default.
        :param is_test: Flag whether process train or test dataset.
        :return: Returns dataset prepared for plotting.
        The dataframe has columns: `["time", "density", "time_step": str, "layer"]`
        """

        if not original_df:
            # Dataframe not specified -> use default one.
            original_df = self.synchrony_full_dataset_df

        selected_df = DatasetResultsProcessor.get_dataset_type(
            original_df, is_test=is_test
        )

        return SynchronyExperimentsProcessor._create_long_format_for_seaborn(
            SynchronyExperimentsProcessor._get_synchrony(
                SynchronyExperimentsProcessor._reformat_original_separate_dataframe(
                    selected_df
                )
            )
        )

    def compute_synchrony_spearman_correlation(
        self, df: pd.DataFrame = None, is_test: bool = False, return_latex=False
    ) -> Union[pd.DataFrame, str]:
        """
        Computes Spearman correlation coefficient for between the
        time bins synchronies to figure out whether there is a correlation
        of change of synchrony with rising time bin size.

        :param df: Original dataframe to compute correlation from.
        :param is_test: Flag whether the dataset is test, else train.
        :param return_latex: Flag whether to return LaTeX table representation
        of the results.
        :return: Returns either dataframe of synchrony correlations for each layer
        and generally across all dataset or as LaTeX table representation.
        """
        if df is None:
            # DataFrame not specified -> use default one.
            df = self.prepare_for_plotting(is_test=is_test)
        results = []

        # Overall correlation
        grouped = df.groupby("time_step")["synchrony"].mean().reset_index()
        r, p = spearmanr(grouped["time_step"], grouped["synchrony"])
        results.append({"layer": "All", "r": round(r, 4), "p": round(p, 4)})

        # Per-layer correlations
        for layer in df["layer"].unique():
            sub = df[df["layer"] == layer]
            r, p = spearmanr(sub["time_step"], sub["synchrony"])
            results.append({"layer": layer, "r": round(r, 4), "p": round(p, 4)})

        result_df = pd.DataFrame(results)

        if return_latex:
            return result_df.to_latex(index=False, float_format="%.4f")

        return result_df
