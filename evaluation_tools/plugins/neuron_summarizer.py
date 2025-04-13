"""
This script serves for summarization of data for each neuron separately.
"""

from typing import Dict, Any

import torch

from evaluation_tools.fields.dataset_analyzer_fields import DatasetDimensions


class NeuronSummarizer:

    @staticmethod
    def _count_neuron_spikes(data: torch.Tensor) -> torch.Tensor:
        """
        Counts neuron spikes for each neuron in batch.

        :param data: Batch data.
        :return: Tensor of shape `[neurons]` with spike counts per neuron.
        """
        return data.float().sum(
            dim=(
                DatasetDimensions.EXPERIMENT.value,
                DatasetDimensions.TRIAL.value,
                DatasetDimensions.TIME_STEP.value,
            )
        )

    @staticmethod
    def batch_neuron_sum(
        data: torch.Tensor, layer: str, all_sum_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Update spike counts for each neuron in batch.

        :param data: Batch data.
        :param layer: Layer name.
        :param all_sum_data: Current total spike counts.
        :return: Returns updated total spike counts.
        """
        if layer not in all_sum_data:
            all_sum_data[layer] = torch.zeros(
                data.shape[DatasetDimensions.NEURON.value], dtype=torch.float32
            )

        all_sum_data[layer] += NeuronSummarizer._count_neuron_spikes(data)

        return all_sum_data

    @staticmethod
    def to_numpy(all_sum_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Converts all data to numpy.

        :param all_sum_data: Data for conversion.
        :return: Returns converted data in numpy arrays.
        """
        return {layer: data.numpy() for layer, data in all_sum_data.items()}
