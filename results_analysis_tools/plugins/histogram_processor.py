"""
This script defines class for processing the dataset histograms.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    DatasetVariantField,
)


class DatasetHistogramProcessor:
    """
    This class serves for processing the histograms and preparing them for plotting.
    """

    def __init__(self, all_results: Dict[EvaluationProcessorChoices, pd.DataFrame]):
        self.time_bin_histograms = (
            DatasetHistogramProcessor._get_all_time_bin_histograms(all_results)
        )

    @staticmethod
    def _get_all_time_bin_histograms(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        :param all_results: All analyses results.
        :return: Returns time bins counts histograms for all variants.
        """
        df = all_results[EvaluationProcessorChoices.FULL_DATASET_ANALYSIS]
        df = df[df["analysis_type"] == AnalysisFields.HISTOGRAM_TIME_BIN_SPIKE_RATES]

        return df

    @staticmethod
    def _reformat_histogram_dataframe(original_df: pd.DataFrame) -> pd.DataFrame:
        """
        :param original_df: Original dataframe as a results from analysis tools.
        :return: Histogram dataframe reformated for easier manipulation while plotting.
        """
        flat_data = []

        for _, row in original_df.iterrows():
            time_step = row["time_step"]
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
                            "time_step": time_step,
                            "layer": layer,
                            "counts": subrow["counts"],
                            "bins": subrow["bins"],
                        }
                    )

        return pd.DataFrame(flat_data)

    @staticmethod
    def _normalize_histogram_for_plotting(
        reformated_df: pd.DataFrame, bins_to_plot: int = 6
    ) -> pd.DataFrame:
        """
        Normalizes the count data to ratios, selects only subset of counts
        and returns results as dataframe.

        :param reformated_df: Already reformatted dataframe from function
        `cls._reformat_histogram_dataframe` for easier manipulation.
        :param bins_to_plot: How many count bins to take (typically for trimming the always 0 counts).
        :return: Returns trimmed and normalized counts to ratios.
        The dataframe has columns: `["layer", "time_step", "bin", "density"]`
        """
        normalized_data = []

        for idx, row in reformated_df.iterrows():
            layer = row["layer"]
            time_step = row["time_step"]
            counts = np.array(row["counts"])[:bins_to_plot]
            bins = np.array(row["bins"])[:bins_to_plot]

            total = counts.sum()
            if total == 0:
                continue  # skip empty histograms

            norm_counts = counts / total

            for bin, norm_count, count in zip(bins, norm_counts, counts):
                normalized_data.append(
                    {
                        "layer": layer,
                        "time_step": time_step,
                        "bin": float(bin),
                        "density": float(norm_count),
                        "count": int(count),
                    }
                )

        return pd.DataFrame(normalized_data)

    def prepare_for_plotting(
        self, original_df: pd.DataFrame = None, is_test: bool = False
    ) -> pd.DataFrame:
        """
        Prepares time bin histogram data for plotting.

        :param original_df: Original histogram data, if `None` then default.
        :param is_test: Flag whether process train or test dataset.
        :return: Returns dataset prepared for plotting.
        The dataframe has columns: `["layer", "time_step", "bin", "density"]`
        """
        if not original_df:
            # Dataframe not specified -> use default one.
            original_df = self.time_bin_histograms

        dataset_variant = (
            DatasetVariantField.TEST if is_test else DatasetVariantField.TRAIN
        )
        selected_df = original_df[original_df["dataset_variant"] == dataset_variant]
        return DatasetHistogramProcessor._normalize_histogram_for_plotting(
            DatasetHistogramProcessor._reformat_histogram_dataframe(selected_df)
        )

    def compute_spike_count_distribution(
        self, df: pd.DataFrame = None, is_test: bool = False
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Computes spike count distribution across all time bins.

        :param df: Data to compute spike distribution from.
        :param is_test: Flag whether we want to count distribution from test dataset.
        :return: Returns dictionary of bin size and their distributions of counts.
        """

        if df is None:
            # DataFrame not specified -> use default one.
            df = self.prepare_for_plotting(is_test=is_test)

        # Group by time_step and bin, summing the counts in each bin
        grouped = df.groupby(["time_step", "bin"])["count"].sum().reset_index()

        # Normalize within each time_step
        distributions = {}
        for ts in grouped["time_step"].unique():
            ts_data = grouped[grouped["time_step"] == ts]

            total_count = ts_data["count"].sum()
            if total_count == 0:
                continue  # avoid division by zero

            # Compute normalized density (relative frequency)
            dist = [
                (int(round(row["bin"])), round(row["count"] / total_count, 4))
                for _, row in ts_data.iterrows()
            ]
            distributions[ts] = dist

        return distributions
