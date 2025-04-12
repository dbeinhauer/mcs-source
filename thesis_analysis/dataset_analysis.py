"""
This source contains scripts used for dataset analysis.
"""

from typing import Dict, Any, Tuple

from enum import Enum

import numpy as np

import nn_model.globals
from nn_model.type_variants import PathDefaultFields
from nn_model.dataset_loader import different_times_collate_fn
from evaluation_tools.response_analyzer import ResponseAnalyzer, AnalyzerChoices
from evaluation_tools.plugins.histogram_processor import HistogramFields
from evaluation_tools.results_plotter import ResultsPlotter


class SpikeCountDifferenceFields(Enum):
    TOTAL_COUNTS = "total_counts"
    COUNT_DIFFERENCE = "count_difference"
    POPULATION_NORMED_DIFFERENCE = "population_normed_difference"


def load_all_histograms(base_path: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Loads all histograms from the given base path with results.

    :param base_path: Path where all histograms are stored.
    :return: Returns all loaded histogram in a dictionary format.
    In format `Dict[variant, Dict[time, Dict[result_type, value]]]]`
    """
    all_histograms: Dict[str, Dict[int, Dict[str, Any]]] = {}
    time_steps = [1, 5, 10, 15, 20]
    variants = [
        AnalyzerChoices.HISTOGRAM_TRAIN.value,
        AnalyzerChoices.HISTOGRAM_TEST.value,
    ]
    for variant in variants:
        all_histograms[variant] = {}
        for time in time_steps:
            path = base_path + f"{variant}-{time}.pkl"
            all_histograms[variant][time] = ResponseAnalyzer.load_pickle_file(path)

    return all_histograms


def check_spike_count_differences(
    all_histograms: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]],
    field_for_inspection: str = HistogramFields.HISTOGRAM_BIN_COUNTS.value,
    base_time: int = 1,
) -> Tuple:
    """
    Checks whether total number of spikes in the merged and original time bin dataset
    is the same. Additionally also checks whether number of neurons with the given spike
    rate is the same. At all it checks that our binning procedure works as expected without
    unexpected behavior.

    :param all_histograms: All histograms to check.
    :param field_for_inspection: Field name with histograms to check.
    :param base_time: Base time to check difference.
    :return: Returns tuple of dictionary of total spike count differences in the layers.
    """
    layer_offsets = {}
    sum_times_differences = {}
    difference = 0

    counts = {}
    differences = {}
    differences_normed = {}

    comparison_results = {}
    # bins_counts = {}
    # neurons_counts = {}
    # bins_differences = {}
    # neurons_differences = {}
    for variant, variant_values in all_histograms.items():
        base_time_all_fields = variant_values[base_time]
        base_time_time_bin_bins = base_time_all_fields[
            HistogramFields.BIN_EDGES_BINS.value
        ]
        base_time_experiment_bins = base_time_all_fields[
            HistogramFields.BIN_EDGES_EXPERIMENT.value
        ]
        sum_times_differences[variant] = {}
        layer_offsets[variant] = {}

        comparison_results[variant] = {}

        counts[variant] = {}
        differences[variant] = {}
        differences_normed[variant] = {}

        for time, time_values in variant_values.items():
            # if time == base_time:
            #     continue
            sum_times_differences[variant][time] = {}
            layer_offsets[variant][time] = {}
            counts[variant][time] = {}
            differences[variant][time] = {}
            differences_normed[variant][time] = {}

            comparison_results[variant][time] = {}

            for field_for_inspection, field_values in time_values.items():
                if field_for_inspection not in [
                    HistogramFields.HISTOGRAM_BIN_COUNTS.value,
                    HistogramFields.HISTOGRAM_EXPERIMENTS_COUNTS.value,
                ]:
                    continue
                counts[variant][time][field_for_inspection] = {}
                differences[variant][time][field_for_inspection] = {}
                differences_normed[variant][time][field_for_inspection] = {}

                comparison_results[variant][time][field_for_inspection] = {}

                for layer, layer_values in field_values.items():
                    base_counts = base_time_all_fields[field_for_inspection][layer]
                    bins_object = base_time_experiment_bins
                    if (
                        field_for_inspection
                        == HistogramFields.HISTOGRAM_BIN_COUNTS.value
                    ):
                        bins_object = base_time_time_bin_bins

                    comparison_results[variant][time][field_for_inspection][layer] = {}

                    total_count_current = np.sum(bins_object[:-1] * layer_values)
                    total_count_base = np.sum(bins_object[:-1] * base_counts)

                    comparison_results[variant][time][field_for_inspection][layer][
                        SpikeCountDifferenceFields.TOTAL_COUNTS.value
                    ] = total_count_current
                    comparison_results[variant][time][field_for_inspection][layer][
                        SpikeCountDifferenceFields.COUNT_DIFFERENCE.value
                    ] = (total_count_current - total_count_base)
                    comparison_results[variant][time][field_for_inspection][layer][
                        SpikeCountDifferenceFields.POPULATION_NORMED_DIFFERENCE.value
                    ] = (
                        comparison_results[variant][time][field_for_inspection][layer][
                            SpikeCountDifferenceFields.COUNT_DIFFERENCE.value
                        ]
                        / nn_model.globals.ORIGINAL_SIZES[layer]
                    )

                    counts[variant][time][field_for_inspection][
                        layer
                    ] = total_count_current
                    differences[variant][time][field_for_inspection][layer] = (
                        total_count_current - total_count_base
                    )
                    differences_normed[variant][time][field_for_inspection][layer] = (
                        differences[variant][time][field_for_inspection][layer]
                        / nn_model.globals.ORIGINAL_SIZES[layer]
                    )

    return counts, differences, differences_normed


if __name__ == "__main__":
    histogram_path = f"{nn_model.globals.PROJECT_ROOT}/evaluation_tools/evaluation_results/histograms/"
    all_histograms = load_all_histograms(histogram_path)
    check_spike_count_differences(all_histograms)

    # print(check_spike_count_differences(all_histograms))
