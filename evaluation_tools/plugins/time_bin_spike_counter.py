"""
This script defines the functionality that counts spikes
in each time bin.
"""

from typing import Dict

import pandas as pd
import numpy as np
import torch

from evaluation_tools.fields.dataset_analyzer_fields import DatasetDimensions


class TimeBinSpikeCounter:
    """
    Class for manipulation with time bin spike counts.
    """

    @staticmethod
    def _count_spikes_in_time_bin(data: torch.Tensor) -> torch.Tensor:
        """
        Sums spike counts across all experiments, trials and neurons.

        :param data: Spiking batch layer data.
        :return: Returns tensor of spike counts for each time bin (temporal resolution).
        Output shape is: `[num_time_steps]`
        """
        return data.float().sum(
            dim=(
                DatasetDimensions.EXPERIMENT.value,
                DatasetDimensions.TRIAL.value,
                DatasetDimensions.NEURON.value,
            )
        )

    @staticmethod
    def batch_time_bin_update(
        data: torch.Tensor, layer: str, time_bin_spike_counts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Summarizes spikes in each time bin for given layer for the batch and
        updates total counter.

        :param data: Current batch data.
        :param layer: Layer to be updated.
        :param time_bin_spike_counts: Total current spike counts.
        :return: Returns updated total spikes counter for each time bin.
        """
        if layer not in time_bin_spike_counts:
            time_bin_spike_counts[layer] = torch.zeros(
                data.shape[DatasetDimensions.TIME_STEP.value], dtype=torch.float32
            )

        time_bin_spike_counts[layer] += TimeBinSpikeCounter._count_spikes_in_time_bin(
            data
        )

        return time_bin_spike_counts

    @staticmethod
    def to_numpy(
        time_bin_spike_counts: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """
        Converts the time bin counts to numpy array values.

        :param time_bin_spike_counts: Bin counts values to be converted.
        :return: Returns converted values to numpy array.
        """
        return {layer: value.numpy() for layer, value in time_bin_spike_counts.items()}

    @staticmethod
    def to_pandas(time_bin_spike_counts: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Converts time bin counts to pandas.

        :param time_bin_spike_counts: Time bin counts already in numpy.
        :return: Returns pandas dataframe of the time bin counts.
        """
        rows = []

        for layer, layer_counts in time_bin_spike_counts.items():
            rows.append(
                {
                    "layer_name": layer,
                    "counts": layer_counts,
                }
            )
        return pd.DataFrame(rows)
