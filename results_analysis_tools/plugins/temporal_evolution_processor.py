"""
This script defines class for processing the temporal evolution of the
neuronal responses for different time bins.
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
        self.time_evolution_full_dataset_df = (
            TemporalEvolutionProcessor._get_all_full_dataset_time_evolution(all_results)
        )
        self.time_evolution_subset_dataset_df = (
            TemporalEvolutionProcessor._get_all_subset_dataset_time_evolution(
                all_results
            )
        )

    @staticmethod
    def _get_all_full_dataset_time_evolution(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ):
        """
        :param all_results: All results.
        :return: Returns all results for full datasets.
        """

        return DatasetResultsProcessor.get_analysis_type(
            DatasetResultsProcessor.get_full_dataset_variant(all_results),
            AnalysisFields.TIME_BIN_SPIKE_COUNTS,
        )

    @staticmethod
    def _get_all_subset_dataset_time_evolution(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ):
        """
        :param all_results: All results.
        :return: Returns all results for subset datasets.
        """

        return DatasetResultsProcessor.get_analysis_type(
            DatasetResultsProcessor.get_subset_dataset_variant(all_results),
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
    def _reformat_time_evolution_dataframe(
        original_df: pd.DataFrame,
        process_subset: bool = False,
    ) -> pd.DataFrame:
        """
        :param original_df: Original dataframe as a results from analysis tools.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :return: Dataframe reformated for easier manipulation while plotting.
        """
        return DatasetResultsProcessor.reformat_dataset_dataframe(
            original_df,
            TemporalEvolutionProcessor._time_count_reformating_row,
            process_subset=process_subset,
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
    def _map_counts_to_time(
        df: pd.DataFrame,
        process_subset: bool = False,
    ) -> pd.DataFrame:
        """
        Interpolates all count sequences for the time resolution 1ms. The aim of this
        function is to spread the time interval spiking information to 1ms bin resolution
        that make it possible to plot and compare the spiking distribution across different
        time bin sizes.

        :param df: Dataframe to be interpolated.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :return: Returns the dataframe with new columns `"padded_time"` symbolizing
        the time at which the count from the `"padded_counts"` belong.
        """
        if process_subset:
            # In case of processing subset dataset, just add the columns to maintain
            # the format but then skip every other steps.
            df[["padded_time", "padded_counts"]] = None
            return df

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
    def _reformat_normalize_and_map_time_distribution(
        df: pd.DataFrame,
        process_subset: bool = False,
    ) -> pd.DataFrame:
        """
        From dataset in original format for given type (train, test) it first reformats
        the dataframe, then normalizes the counts to be ratios across the time interval and
        finally it interpolates the counts in each time bin to the original time interval
        duration for comparison of the temporal behavior in the plot.

        :param df: Original dataset to be formatted.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")

        :return: Returns dataframe with columns
        `[{'time_step' | "subset_id"}, 'layer', 'counts', 'normalized_counts', 'duration', 'padded_time', 'padded_counts']`
        """
        return TemporalEvolutionProcessor._map_counts_to_time(
            TemporalEvolutionProcessor._normalize_counts_in_time(
                TemporalEvolutionProcessor._reformat_time_evolution_dataframe(
                    df, process_subset=process_subset
                )
            ),
            process_subset=process_subset,
        )

    @staticmethod
    def _create_long_format_for_seaborn(
        df: pd.DataFrame,
        process_subset: bool = False,
    ) -> pd.DataFrame:
        """
        Converts and filters the dataframe to long format (each values of the list in new row)
        for plotting in seaborn.

        :param df: Dataframe to be converted.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :return: Returns long format of the provided dataframe with columns customized for
        plotting in format: `["time", "density", "time_step": str, "layer"]`
        """
        long_df = []

        time_step_or_subset_id_column_key = (
            "time_step" if not process_subset else "subset_id"
        )

        for _, row in df.iterrows():
            time_step_or_subset_id = row[time_step_or_subset_id_column_key]
            layer = row["layer"]

            # Process subset of models -> it is not necessary to take interpolated counts \
            # (all results have same time binning)
            counts = row["normalized_counts"]
            times = np.arange(0, len(counts))
            if not process_subset:
                # Processing full dataset -> select correct time binning.
                times = row["padded_time"]
                counts = row["padded_counts"] / time_step_or_subset_id

            time_or_subset_string = (
                f"{time_step_or_subset_id} ms"
                if not process_subset
                else time_step_or_subset_id
            )

            for t, c in zip(times, counts):
                long_df.append(
                    {
                        "time": t,
                        "density": c,
                        time_step_or_subset_id_column_key: time_or_subset_string,  # string for hue
                        "layer": layer,
                    }
                )

        return pd.DataFrame(long_df)

    def prepare_for_plotting(
        self,
        original_df: pd.DataFrame = None,
        is_test: bool = False,
        process_subset: bool = False,
    ) -> pd.DataFrame:
        """
        Prepares time bin histogram data for plotting on full dataset.

        :param original_df: Original histogram data, if `None` then default.
        :param is_test: Flag whether process train or test dataset.
        :param process_subset: Flag whether to process subset dataset or not ("subset_id" instead of "time_step")
        :return: Returns dataset prepared for plotting.
        The dataframe has columns: `["time", "density", {"time_step" | "subset_id"}: str, "layer"]`
        """

        if original_df is None:
            # Dataframe not specified -> use default one.
            if process_subset:
                # Processing dataset subset variants.
                original_df = self.time_evolution_subset_dataset_df
            else:
                # Processing dataset full variant for multiple time bin sizes.
                original_df = self.time_evolution_full_dataset_df

        selected_df = DatasetResultsProcessor.get_dataset_type(
            original_df, is_test=is_test
        )

        return DatasetResultsProcessor.ensure_layer_order(
            TemporalEvolutionProcessor._create_long_format_for_seaborn(
                TemporalEvolutionProcessor._reformat_normalize_and_map_time_distribution(
                    selected_df, process_subset=process_subset
                ),
                process_subset=process_subset,
            )
        )

    def _prepare_full_dataset_for_subset_comparison(
        self,
        original_df: pd.DataFrame = None,
        is_test: bool = False,
    ):
        """
        Prepares full dataset for plotting temporal dynamics comparison with the subset dataset.
        Renames "time_step" to "subset_id" and sets it to `"-1"`. Takes only binning of 20ms.

        :param original_df: Full dataset dataframe to be processed.
        :param is_test: Flag whether to process test dataset.
        :return: Returns dataframe with columns
        `["time", "density", "subset_id": str = "-1": str, "layer"]`
        """
        if original_df is None:
            # Dataframe not specified -> use default one.
            original_df = self.time_evolution_full_dataset_df

        selected_df = DatasetResultsProcessor.get_dataset_type(
            original_df, is_test=is_test
        )

        # Select only 20ms time binning, get rid of "time_step" column and assign "-1" to "subset_id".
        selected_df = DatasetResultsProcessor.replace_time_step_with_subset_full(
            DatasetResultsProcessor.get_time_step_20_full(selected_df)
        )
        # Process full dataset as the subset dataset.
        selected_df = self.prepare_for_plotting(
            selected_df,
            process_subset=True,
            is_test=False,
        )

        return selected_df

    def prepare_for_plotting_subset_full_comparison(
        self, is_test: bool = False
    ) -> pd.DataFrame:
        """
        Prepares data of both full and subset dataset for plotting comparison of the temporal
        dynamics across all layers. The full dataset is processed as subset dataset with
        `"subset_id"` set to `"-1"`.

        :param is_test: Flag whether to process test dataset.
        :return: Returns dataframe with columns of both full and subset dataset in format
        `["time", "density", "subset_id": str, "layer"]`
        """

        full_df = self._prepare_full_dataset_for_subset_comparison(is_test=is_test)
        subset_df = self.prepare_for_plotting(is_test=is_test, process_subset=True)

        # Merge full and subset dataset.
        combined_df = DatasetResultsProcessor.merge_full_and_subset_dataframes(
            full_df, subset_df
        )

        # Mapping of the 20 ms time steps to original 1ms resolution.
        combined_df["time"] *= 20
        # Pad the rest of the plot with the artificial last entry to 711 ms with the last value.
        last_entries = (
            combined_df.groupby(["layer", "subset_id"])
            .apply(lambda df: df[df["time"] == df["time"].max()])
            .reset_index(drop=True)
        )
        last_entries["time"] = 711
        combined_df = pd.concat([combined_df, last_entries], ignore_index=True)

        # Ensure the order of the layers.
        combined_df = DatasetResultsProcessor.ensure_layer_order(combined_df)

        return combined_df

    def compute_correlation_matrix_full(
        self, original_df: pd.DataFrame = None, is_test: bool = False
    ) -> pd.DataFrame:
        """
        Creates correlation matrix of the density values across all time steps
        across all layers for all time bin sizes.

        :param original_df: Original dataframe analysis data.
        :param is_test: Whether to take test dataset.
        :return: Returns Pearson correlation matrix of the count densities across all
        time steps across all layers. The matrix is in the format for `seaborn.heatmap`.
        """
        if original_df is None:
            # Dataframe not specified -> use default one.
            original_df = self.time_evolution_full_dataset_df

        selected_df = DatasetResultsProcessor.get_dataset_type(
            original_df, is_test=is_test
        )
        # Pivot: rows = time, columns = time_step, values = density
        long_df = self.prepare_for_plotting(
            original_df=selected_df, is_test=is_test, process_subset=False
        )

        # Aggregate across layers: average density at each (time, time_step)
        aggregated = (
            long_df.groupby(["time", "time_step"])["density"].mean().reset_index()
        )
        # Replace string time_step with int.
        aggregated["time_step"] = (
            aggregated["time_step"].str.replace(" ms", "", regex=False).astype(int)
        )
        pivoted = aggregated.pivot(index="time", columns="time_step", values="density")
        pivoted = pivoted.reindex(sorted(pivoted.columns), axis=1)
        pivoted = pivoted[sorted(pivoted.columns)]

        # Optional: fill or drop NaNs
        pivoted = pivoted.fillna(0)

        # Correlation matrix
        corr_matrix = pivoted.corr(method="pearson")

        corr_matrix.columns = [f"{col} ms" for col in corr_matrix.columns]
        corr_matrix.index = [f"{row} ms" for row in corr_matrix.index]

        return corr_matrix
