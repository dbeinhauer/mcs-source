"""
This source code defines the functionality that processes the evaluation results from a batch.
"""

from typing import Dict, List, Tuple

import torch

from nn_model.type_variants import EvaluationFields
from evaluation_tools.fields.prediction_analysis_fields import (
    BatchSummaryFields,
    PredictionDimensions,
    EvaluationPairsVariants,
)


class BatchPredictionProcessor:

    @staticmethod
    def mse_per_example(
        first_tensor: torch.Tensor, second_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes MSE across each experiment between the given two tensors.

        :param first_tensor: First tensor for MSE.
        :param second_tensor: Second tensor for MSE.
        :return: Returns tensor of MSE per each experiment (in shape `[batch_size]`).
        """
        return ((first_tensor - second_tensor) ** 2).mean(
            dim=(
                PredictionDimensions.TIME_STEP.value,
                PredictionDimensions.NEURON.value,
            )
        )

    @staticmethod
    def apply_function_across_layers(
        first_all_layer_values: Dict[str, torch.Tensor],
        second_all_layer_values: Dict[str, torch.Tensor],
        function_to_apply,
    ) -> Dict[str, torch.Tensor]:
        """
        Applies arbitrary function between the two given dictionaries of all layers values.

        :param first_all_layer_values: First all layer dictionary.
        :param second_all_layer_values: Second all layer dictionary.
        :param function_to_apply: Function to apply between each pair of tensors in
        each layer.
        :return: Returns results of function application for each layer.
        """
        return {
            layer: function_to_apply(
                first_all_layer_values[layer], second_all_layer_values[layer]
            )
            for layer in first_all_layer_values
            # We expect the first and second values have the same keys.
        }

    @staticmethod
    def apply_function_between_selected_pairs_of_results(
        batch_data: Dict[EvaluationFields, Dict[str, torch.Tensor]],
        pairs_to_compute: List[EvaluationPairsVariants],
        function_to_apply,
    ) -> Dict[EvaluationPairsVariants, Dict[str, torch.Tensor]]:
        """
        Applies the provided function to all provided pairs from the batch data.

        :param batch_data: Batched data containing all evaluation fields.
        :param pairs_to_compute: Pairs of evaluation fields that we want
        to compute the function on.
        :param function_to_apply: Function to apply on between the given pairs.
        :return: Returns the results of function application between each
        selected evaluation fields.
        """
        results = {}

        for pair in pairs_to_compute:
            field1, field2 = pair.value
            layer_data1 = batch_data[field1]
            layer_data2 = batch_data[field2]

            result = BatchPredictionProcessor.apply_function_across_layers(
                layer_data1, layer_data2, function_to_apply
            )

            results[pair] = result

        return results

    @staticmethod
    def compute_batch_per_example_mse(
        batch_data: Dict[EvaluationFields, Dict[str, torch.Tensor]],
    ) -> Dict[EvaluationPairsVariants, Dict[str, torch.Tensor]]:
        return (
            BatchPredictionProcessor.apply_function_between_selected_pairs_of_results(
                batch_data,
                list(EvaluationPairsVariants),
                BatchPredictionProcessor.mse_per_example,
            )
        )

    @staticmethod
    def process_batch_results(
        batch_data: Dict[EvaluationFields, Dict[str, torch.Tensor]],
    ):
        return BatchPredictionProcessor.compute_batch_per_example_mse(batch_data)
