"""
This script defines class for processing the dataset histograms.
"""

from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd

from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
)

from results_analysis_tools.plugins.dataset_results_processor import (
    DatasetResultsProcessor,
)


class DatasetHistogramProcessor:
    """
    This class serves for processing the histograms of dataset processing and
    preparing them for plotting.
    """

    def __init__(self, all_results: Dict[EvaluationProcessorChoices, pd.DataFrame]):
        self.time_bin_histograms_full_dataset = (
            DatasetHistogramProcessor._get_all_full_dataset_time_bin_histograms(
                all_results
            )
        )
        self.time_bin_histograms_subset_dataset = (
            DatasetHistogramProcessor._get_all_subset_dataset_time_bin_histograms(
                all_results
            )
        )

    @staticmethod
    def _get_all_full_dataset_time_bin_histograms(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        :param all_results: All analyses results.
        :return: Returns time bins counts histograms for all variants in full dataset.
        """

        return DatasetResultsProcessor.get_analysis_type(
            DatasetResultsProcessor.get_full_dataset_variant(all_results),
            AnalysisFields.HISTOGRAM_TIME_BIN_SPIKE_RATES,
        )

    @staticmethod
    def _get_all_subset_dataset_time_bin_histograms(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        :param all_results: All analyses results.
        :return: Returns time bins counts histograms for all variants in subset dataset.
        """

        return DatasetResultsProcessor.get_analysis_type(
            DatasetResultsProcessor.get_subset_dataset_variant(all_results),
            AnalysisFields.HISTOGRAM_TIME_BIN_SPIKE_RATES,
        )

    @staticmethod
    def _histogram_reformating_row(row: pd.Series) -> Dict[str, Any]:
        """
        :return: Row as dictionary for creating pandas dataframe.
        """
        return {
            "counts": row["counts"],
            "bins": row["bins"],
        }

    @staticmethod
    def _reformat_histogram_dataframe(
        original_df: pd.DataFrame,
        process_subset: bool = False,
    ) -> pd.DataFrame:
        """
        :param original_df: Original dataframe as a results from analysis tools.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :return: Histogram dataframe reformated for easier manipulation while plotting.
        """
        return DatasetResultsProcessor.reformat_dataset_dataframe(
            original_df,
            DatasetHistogramProcessor._histogram_reformating_row,
            process_subset=process_subset,
        )

    @staticmethod
    def _normalize_histogram_for_plotting(
        reformated_df: pd.DataFrame,
        bins_to_plot: int = 6,
        process_subset: bool = False,
    ) -> pd.DataFrame:
        """
        Normalizes the count data to ratios, selects only subset of counts
        and returns results as dataframe.

        :param reformated_df: Already reformatted dataframe from function
        `cls._reformat_histogram_dataframe` for easier manipulation.
        :param bins_to_plot: How many count bins to take (typically for trimming the always 0 counts).
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :return: Returns trimmed and normalized counts to ratios.
        The dataframe has columns: `["layer", "time_step", "bin", "density"]`
        or "subset_id" instead of "time_step" in case `process_subset` is `True`.
        """
        normalized_data = []

        time_step_or_subset_id_column_key = (
            "time_step" if not process_subset else "subset_id"
        )
        for idx, row in reformated_df.iterrows():
            layer = row["layer"]
            # Determine the time step or subset ID based on the process_subset flag.
            time_step_or_subset_id = row[time_step_or_subset_id_column_key]
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
                        time_step_or_subset_id_column_key: time_step_or_subset_id,
                        "bin": float(bin),
                        "density": float(norm_count),
                        "count": int(count),
                    }
                )

        return pd.DataFrame(normalized_data)

    def prepare_for_plotting(
        self,
        original_df: pd.DataFrame = None,
        is_test: bool = False,
        process_subset: bool = False,
    ) -> pd.DataFrame:
        """
        Prepares full dataset time bin histogram data for plotting.

        :param original_df: Original histogram data, if `None` then default.
        :param is_test: Flag whether process train or test dataset.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :return: Returns dataset prepared for plotting.
        The dataframe has columns: `["layer", "time_step", "bin", "density"]`
        """
        if original_df is None:
            # Dataframe not specified -> use default one.
            if process_subset:
                # Processing dataset subset variants.
                original_df = self.time_bin_histograms_subset_dataset
            else:
                # Processing dataset full variant for multiple time bin sizes.
                original_df = self.time_bin_histograms_full_dataset

        selected_df = DatasetResultsProcessor.get_dataset_type(
            original_df, is_test=is_test
        )
        return DatasetResultsProcessor.ensure_layer_order(
            DatasetHistogramProcessor._normalize_histogram_for_plotting(
                DatasetHistogramProcessor._reformat_histogram_dataframe(
                    selected_df, process_subset=process_subset
                ),
                process_subset=process_subset,
            )
        )

    def compute_spike_count_distribution(
        self,
        df: pd.DataFrame = None,
        is_test: bool = False,
        process_subset: bool = False,
        format_to_latex: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """
        Computes spike count distribution across all time bins on full dataset.

        :param df: Data to compute spike distribution from.
        :param is_test: Flag whether we want to count distribution from test dataset.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :param format_to_latex: Whether convert to latex table.
        :return: Returns pandas or LaTeX table representation of the bin count distribution.
        """

        if df is None:
            # DataFrame not specified -> use default one.
            df = self.prepare_for_plotting(
                is_test=is_test, process_subset=process_subset
            )

        time_step_or_subset_id_column_key = (
            "time_step" if not process_subset else "subset_id"
        )

        # Group by time_step and bin, summing the counts in each bin
        grouped = (
            df.groupby([time_step_or_subset_id_column_key, "bin"])["count"]
            .sum()
            .reset_index()
        )

        # Normalize within each time_step
        distributions = {}
        for ts in grouped[time_step_or_subset_id_column_key].unique():
            ts_data = grouped[grouped[time_step_or_subset_id_column_key] == ts]

            total_count = ts_data["count"].sum()
            if total_count == 0:
                continue  # avoid division by zero

            # Compute normalized density (relative frequency)
            dist = [
                (int(round(row["bin"])), round(row["count"] / total_count, 4))
                for _, row in ts_data.iterrows()
            ]
            distributions[ts] = dist

        return DatasetHistogramProcessor.convert_mean_counts_to_pandas_or_latex(
            distributions,
            format_to_latex=format_to_latex,
            process_subset=process_subset,
        )

    @staticmethod
    def convert_mean_counts_to_pandas_or_latex(
        distributions: Dict[int, List[Tuple[int, float]]],
        format_to_latex: bool = False,
        process_subset: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """
        Converts mean count distribution to either pandas or latex table.

        :param distribution: Distribution to be converted.
        :param format_to_latex: Whether convert to latex table.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :return: Returns distribution representation in either pandas or latex table.
        """
        time_step_or_subset_id_column_key = (
            "time_step" if not process_subset else "subset_id"
        )
        # Convert to DataFrame
        rows = []
        for time_step_or_subset, bin_density_list in distributions.items():
            for spike_count_bin, density in bin_density_list:
                rows.append(
                    {
                        time_step_or_subset_id_column_key: int(time_step_or_subset),
                        "spike_count_bin": int(spike_count_bin),
                        "normalized_density": round(density, 6),  # round for clarity
                    }
                )

        df = pd.DataFrame(rows)

        if format_to_latex:
            # Covert to LaTeX table
            return df.pivot(
                index="spike_count_bin",
                columns=time_step_or_subset_id_column_key,
                values="normalized_density",
            ).to_latex(index=True, float_format="%.4f")

        return df

    def summarize_global_density_vs_full(
        self,
        subset_df: pd.DataFrame,
        full_df: pd.DataFrame,
        format_to_latex: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """
        Computes the average density of spike counts across all layers of all subsets
        and compares it to the full dataset density of time step 20.

        :param subset_df: Dataframe of subset dataset.
        :param full_df: Dataframe of full dataset.
        :param format_to_latex: Flag whether to return LaTeX table.
        :return: Returns summary of spike count density across all layers
        and subsets as dataframe of LaTeX table.
        """

        # Average full model densities across layers
        full_avg = (
            full_df.groupby("spike_count_bin")["normalized_density"]
            .mean()
            .reset_index()
            .rename(columns={"normalized_density": "full_density"})
        )

        # Average subset densities across subset_id AND layer
        subset_avg = (
            subset_df.groupby(["subset_id", "spike_count_bin"])["normalized_density"]
            .mean()  # average across layers for each subset
            .reset_index()
            .groupby("spike_count_bin")["normalized_density"]
            .agg(subset_mean="mean", subset_std="std")
            .reset_index()
        )

        # Merge both summaries
        summary = pd.merge(full_avg, subset_avg, on="spike_count_bin")
        summary = summary[
            ["spike_count_bin", "full_density", "subset_mean", "subset_std"]
        ]
        summary = summary.round(4)

        if format_to_latex:
            # Return LaTeX table.
            return summary.to_latex(
                index=False,
                float_format="%.4f",
                caption="Spike Count Density Summary Averaged Across All Layers",
                label="tab:spike_density_global",
            )

        return summary
