"""
This script defines class for processing the temporal evolution of the neuronal responses for different time bins.
"""

from typing import Dict, Any

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from results_analysis_tools.plugins.dataset_results_processor import (
    DatasetResultsProcessor,
)
from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
)


class TemporalEvolutionProcessor:

    def __init__(self, all_results: Dict[EvaluationProcessorChoices, pd.DataFrame]):
        self.time_evolution_df = TemporalEvolutionProcessor._get_all_time_evolution(
            all_results
        )

    @staticmethod
    def _get_all_time_evolution(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ):
        return DatasetResultsProcessor.get_analysis_type(
            DatasetResultsProcessor.get_full_dataset_variant(all_results),
            AnalysisFields.TIME_BIN_SPIKE_COUNTS,
        )

    @staticmethod
    def _time_count_reformating_row(row: pd.Series) -> Dict[str, Any]:
        """
        Function for reformating the original data that takes care of the time counts.

        :param row: Original dataframe row.
        :return: Returns counts in each time.
        """
        return {
            "counts": row["counts"],
        }

    @staticmethod
    def _reformat_time_evolution_dataframe(original_df: pd.DataFrame) -> pd.DataFrame:
        """
        :param original_df: Original dataframe as a results from analysis tools.
        :return: Histogram dataframe reformated for easier manipulation while plotting.
        """
        return DatasetResultsProcessor.reformat_dataset_dataframe(
            original_df, TemporalEvolutionProcessor._time_count_reformating_row
        )

    @staticmethod
    def _normalize_counts_in_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes counts in each time step to the ratio of all spikes across all time steps.

        :param df: Dataframe to add normalized counts.
        :return: Returns dataframe with new column `"normalized_counts"` containing normalized
        spike counts across the time interval.
        """

        df["normalized_counts"] = df["counts"].apply(
            lambda x: x / x.sum() if x.sum() != 0 else np.zeros_like(x)
        )
        return df

    @staticmethod
    def _interpolate_normalized_counts(
        row: pd.Series, common_time: np.ndarray
    ) -> pd.Series:
        """
        Interpolates the spike count sequences to the original time sequence (duration of 1 experiment).

        :param row: Row of the dataframe to be interpolated (uniform time bin size).
        :param common_time: Array of the all integer time steps (in 1 ms resolution)
        for the experiment duration.
        :return: Returns row with interpolated count values (density) in the time in 1 ms resolution.
        """
        time_step_size = row["time_step"]
        counts = row["counts"]
        duration = len(counts) * time_step_size

        if counts.sum() == 0:
            normalized = np.zeros_like(counts)
        else:
            normalized = counts / counts.sum()

        time_points = np.linspace(0, duration, len(counts), endpoint=False)

        # Interpolate to common time (1ms resolution).
        f_interp = interp1d(
            time_points,
            normalized,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        padded = f_interp(common_time)

        # padded = gaussian_filter1d(padded, sigma=2)  # try sigma=2 to 5

        return pd.Series([common_time, padded])

    @staticmethod
    def _map_counts_to_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolates all count sequences for the time resolution 1ms. The aim of this
        function is to spread the time interval spiking information to 1ms bin resolution
        that make it possible to plot and compare the spiking distribution across different
        time bin sizes.

        :param df: Dataframe to be interpolated.
        :return: Returns the dataframe with new columns `"padded_time"` symbolizing
        the time at which the count from the `"padded_counts"` belong.
        """
        # Find experiment duration in the maximal resolution.
        df["duration"] = df.apply(
            lambda row: len(row["counts"]) * row["time_step"], axis=1
        )
        max_time = df["duration"].max()

        # Find the maximal resolution (typically 1ms).
        common_resolution = df["time_step"].min()
        common_time = np.arange(0, max_time, common_resolution)

        # Interpolate each resolution data to the maximal resolution.
        df[["padded_time", "padded_counts"]] = df.apply(
            lambda row: TemporalEvolutionProcessor._interpolate_normalized_counts(
                row, common_time
            ),
            axis=1,
        )

        return df

    @staticmethod
    def _reformat_normalize_and_map_time_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """
        From dataset in original format for given type (train, test) it first reformats
        the dataframe, then normalizes the counts to be ratios across the time interval and
        finally it interpolates the counts in each time bin to the original time interval
        duration for comparison of the temporal behavior in the plot.

        :param df: Original dataset to be formatted.
        :return: Returns dataframe with columns
        `['time_step', 'layer', 'counts', 'normalized_counts', 'duration', 'padded_time', 'padded_counts']`
        """
        return TemporalEvolutionProcessor._map_counts_to_time(
            TemporalEvolutionProcessor._normalize_counts_in_time(
                TemporalEvolutionProcessor._reformat_time_evolution_dataframe(df)
            )
        )

    @staticmethod
    def _create_long_format_for_seaborn(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts and filters the dataframe to long format (each values of the list in new row)
        for plotting in seaborn.

        :param df: Dataframe to be converted.
        :return: Returns long format of the provided dataframe with columns customized for
        plotting in format: `["time", "density", "time_step": str, "layer"]`
        """
        long_df = []

        for _, row in df.iterrows():
            time_step = row["time_step"]
            layer = row["layer"]
            times = row["padded_time"]
            counts = row["padded_counts"] / time_step
            # counts[-1] = counts[-2]

            for t, c in zip(times, counts):
                long_df.append(
                    {
                        "time": t,
                        "density": c,
                        "time_step": f"{time_step} ms",  # string for hue
                        "layer": layer,
                    }
                )

        return pd.DataFrame(long_df)

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
            original_df = self.time_evolution_df

        selected_df = DatasetResultsProcessor.get_dataset_type(
            original_df, is_test=is_test
        )

        return TemporalEvolutionProcessor._create_long_format_for_seaborn(
            TemporalEvolutionProcessor._map_counts_to_time(
                TemporalEvolutionProcessor._normalize_counts_in_time(
                    TemporalEvolutionProcessor._reformat_time_evolution_dataframe(
                        selected_df
                    )
                )
            )
        )
