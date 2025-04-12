"""
This source contains scripts used for dataset analysis.
"""

from typing import Dict, Any

import numpy as np

import nn_model.globals
from nn_model.type_variants import PathDefaultFields
from nn_model.dataset_loader import different_times_collate_fn
from evaluation_tools.response_analyzer import ResponseAnalyzer, AnalyzerChoices
from evaluation_tools.plugins.histogram_processor import HistogramFields
from evaluation_tools.results_plotter import ResultsPlotter


def load_all_histograms(base_path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Loads all histograms from the given base path with results.
    
    :param base_path: Path where all histograms are stored.
    :return: Returns all loaded histogram in a dictionary format. 
    In format `Dict[variant, Dict[time, Dict[result_type, value]]]]`
    """
    all_histograms = {}
    time_steps = [1, 5, 10, 15, 20]
    variants = [AnalyzerChoices.HISTOGRAM_TRAIN.value, AnalyzerChoices.HISTOGRAM_TEST.value]
    for variant in variants:
        all_histograms[variant] = {}
        for time in time_steps:
            path = base_path + f"{variant}-{time}.pkl"
            all_histograms[variant][time] = ResponseAnalyzer.load_pickle_file(path)
            
    return all_histograms

def check_spike_count_differences(
        all_histograms: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
        field_for_inspection: str = HistogramFields.HISTOGRAM_EXPERIMENTS_COUNTS.value,
        base_time: int = 1,
    ) -> int:
    """
    Checks whether total number of spikes in the merged and original time bin dataset 
    is the same. Additionally also checks whether number of neurons with the given spike
    rate is the same. At all it checks that our binning procedure works as expected without
    loss of information.

    :param all_histograms: All histograms to check.
    :param field_for_inspection: Field name with histograms to check.
    :param base_time: Base time to check difference.
    :return: Returns total difference in number of
    """
    difference = 0
    for variant, variant_values in all_histograms.items():
        base_time_histograms = variant_values[base_time][field_for_inspection]
        for time, time_values in variant_values.items():
            if time == base_time:
                continue
            for layer, layer_values in time_values[field_for_inspection].items():
                difference += np.sum(
                    (base_time_histograms[layer - layer_values]).abs().reshape(-1), 
                    axis=0,
                )
                
    return difference
                
    
    