"""
This script servers for computing Fano Factor for the provided data.
Note: It does not really make sense to compute it for train dataset
as it contains only 1 trial.
"""

from typing import Dict, Any

import torch

from evaluation_tools.fields.dataset_analyzer_fields import (
    DatasetDimensions,
    DatasetVariantField,
)
from evaluation_tools.fields.dataset_parameters import DATASET_SIZES


class FanoFactorProcessor:

    @staticmethod
    def _init_fano_tensor(is_test: bool) -> torch.Tensor:
        """
        Initializes tensor for all experiments.

        :param is_test: Flag whether processing test dataset.
        :return: Returns pre-initialized torch tensor of `[num_experiments]`.
        """
        return torch.zeros(
            (
                DATASET_SIZES[DatasetVariantField.TEST][DatasetDimensions.EXPERIMENT]
                if is_test
                else DATASET_SIZES[DatasetVariantField.TRAIN][
                    DatasetDimensions.EXPERIMENT
                ]
            ),
            dtype=torch.float32,
        )

    @staticmethod
    def _compute_fano_factor(data: torch.Tensor) -> torch.Tensor:
        """
        Computes fano factor for the experiments.

        :param data: Data to compute fano factor from.
        :return: Returns Fano factor for the batch of data in shape `[batch_size]`.
        """

        spike_counts_per_trial = data.float().sum(
            dim=(DatasetDimensions.TIME_STEP.value, DatasetDimensions.NEURON.value)
        )

        # Compute mean and variance over trials (dim=1)
        mean_spikes = spike_counts_per_trial.mean(dim=DatasetDimensions.TRIAL.value)
        var_spikes = spike_counts_per_trial.var(
            dim=DatasetDimensions.TRIAL.value, unbiased=False
        )

        # Fano factor
        return var_spikes / (mean_spikes + 1e-6)

    @staticmethod
    def batch_fano_computation(
        data: torch.Tensor,
        layer: str,
        is_test: bool,
        all_fano_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes fano factor for batch of data (only for test dataset).

        :param data: Batch of data.
        :param layer: Layer name.
        :param is_test: Flag whether test dataset.
        :param all_fano_data: All current fano factors for experiments/
        :return: Returns updated all fano factors tensor.
        """
        if layer not in all_fano_data:
            all_fano_data[layer] = FanoFactorProcessor._init_fano_tensor(is_test)

        if not is_test:
            # It does not really make sense to compute the Fano factor for 1 trial.
            return all_fano_data

        all_fano_data[layer] = FanoFactorProcessor._compute_fano_factor(data)

        return all_fano_data

    @staticmethod
    def to_numpy(all_fano_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Convert data to numpy.
        """
        return {layer: data.numpy() for layer, data in all_fano_data.items()}
