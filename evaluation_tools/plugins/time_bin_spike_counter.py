"""
This script defines the functionality that counts spikes
in each time bin.
"""

from typing import Dict

import torch


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
        return data.float().sum(dim=(0, 1, 3))

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
                data.shape[2], dtype=torch.float32
            )

        time_bin_spike_counts[layer] += TimeBinSpikeCounter._count_spikes_in_time_bin(
            data
        )

        return time_bin_spike_counts

    @staticmethod
    def to_numpy(time_bin_spike_counts: Dict[str, torch.Tensor]) -> Dict:
        """
        Converts the time bin counts to numpy array values.

        :param time_bin_spike_counts: Bin counts values to be converted.
        :return: Returns converted values to numpy array.
        """
        return {layer: value.numpy() for layer, value in time_bin_spike_counts.items()}
