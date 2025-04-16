"""
This source code defines the functionality that processes the evaluation results from a batch.
"""

from typing import Dict, List, Tuple, Any

import torch

from nn_model.type_variants import EvaluationFields
from evaluation_tools.fields.prediction_analysis_fields import (
    BatchSummaryFields,
    PredictionDimensions,
    EvaluationPairsVariants,
    BatchJobParameters,
)


class BatchPredictionProcessor:
    """
    This class serves for processing predictions and aggregating statistics in separate batch.
    """

    # Summaries that are computed separately:
    separate_summaries = [BatchSummaryFields.SYNCHRONY]
    # Summaries that are computed pair-wise:
    paired_summaries = [
        BatchSummaryFields.MSE,
        BatchSummaryFields.PEARSON,
        BatchSummaryFields.PEARSON_SYNCHRONY,
        BatchSummaryFields.DRIFT_FREE_FORCED,
        BatchSummaryFields.DRIFT_DELTA,
    ]

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
    def pearson_per_dimension(
        first_tensor: torch.Tensor, second_tensor: torch.Tensor, dimension: int
    ) -> torch.Tensor:
        """
        Computes Pearson's CC per specified dimension.

        :param first_tensor: First tensor of values for CC computation.
        :param second_tensor: Second tensor of values for CC computation.
        :param dimension: Dimension to compute CC across.
        :return: Returns appropriate Pearson's CC in shape same as input by .
        """
        first_tensor = first_tensor.float()
        second_tensor = second_tensor.float()

        first_mean = first_tensor.mean(dim=dimension, keepdim=True)
        second_mean = second_tensor.mean(dim=dimension, keepdim=True)

        first_centered = first_tensor - first_mean
        second_centered = second_tensor - second_mean

        numerator = (first_centered * second_centered).sum(dim=dimension)
        denominator = (
            first_centered.square().sum(dim=dimension).sqrt()
            * second_centered.square().sum(dim=dimension).sqrt()
        )

        return numerator / (denominator + 1e-6)

    @staticmethod
    def pearson_per_experiment(
        first_tensor: torch.Tensor, second_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes mean for each experiment of person correlation coefficient per neuron.

        :param first_tensor: First tensor of original batch values.
        :param second_tensor: Second tensor of original batch values.
        :return: Returns mean of Pearson's CC for each experiment in shape `[batch_size]`.
        """
        return BatchPredictionProcessor.pearson_per_dimension(
            first_tensor, second_tensor, PredictionDimensions.TIME_STEP.value
        ).mean(
            dim=PredictionDimensions.NEURON.value - 1
        )  # Dimension of neurons is -1 because we summed over time before.

    @staticmethod
    def synchrony_curve(first_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes synchrony curve of the provided results. The sum of the neuronal spikes across
        all neurons per each time bin in each experiment.

        :param first_tensor: Original tensor from the batch.
        :return: Returns synchrony curve for each experiment in shape `[batch_size, time_steps]`.
        """
        return first_tensor.sum(dim=PredictionDimensions.NEURON.value)

    @staticmethod
    def mse_between_synchrony_curves(
        first_tensor: torch.Tensor, second_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes MSE between the synchrony curves per batch example.

        :param first_tensor: First tensor of original batch values.
        :param second_tensor: Second tensor of original batch values.
        :return: Returns MSE of the synchrony curves differences per example in shape `[batch_size]`.
        """
        first_synchrony = BatchPredictionProcessor.synchrony_curve(first_tensor)
        second_synchrony = BatchPredictionProcessor.synchrony_curve(second_tensor)

        return ((first_synchrony - second_synchrony) ** 2).mean(
            dim=PredictionDimensions.TIME_STEP.value
        )

    @staticmethod
    def pearson_synchrony(
        first_tensor: torch.Tensor, second_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes Pearson's CC for of the synchrony curves for the provided batch data.

        :param first_tensor: First tensor of original batch values.
        :param second_tensor: Second tensor of original batch values.
        :return: Returns Pearson's CCs for each batch example of their synchrony
        curves in shape `[batch_size]`.
        """

        return BatchPredictionProcessor.pearson_per_dimension(
            # Compute the synchrony curves and compute Pearson's CC across time for them.
            BatchPredictionProcessor.synchrony_curve(first_tensor),
            BatchPredictionProcessor.synchrony_curve(second_tensor),
            PredictionDimensions.TIME_STEP.value,
        )

    @staticmethod
    def drift_curve(
        first_tensor: torch.Tensor, second_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes drift curve between predictions and teacher-forced predictions. How does the
        free predictions (without hidden state reset) diminish their quality with the longer
        prediction sequence.

        :param first_tensor: First original batch tensor.
        :param second_tensor: Second original batch tensor.
        :return: Returns drift between the free prediction and teacher forced prediction.
        The shape of the result is `[batch_size, time_steps]`.
        """
        return ((first_tensor - second_tensor) ** 2).mean(
            dim=PredictionDimensions.NEURON.value
        )

    @staticmethod
    def drift_delta(
        first_tensor: torch.Tensor, second_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes drift delta for the batch of predictions and teacher-forced predictions.

        It computes the difference of the drift in each time step. By this we can determine
        how much does the predictions deteriorate with length of the sequence prediction.

        :param first_tensor: First original batch tensor.
        :param second_tensor: Second original batch tensor.
        :return: Returns delta (how does drift change) of each time step of the drift curve
        in shape `[batch_size, time_steps - 1]`.
        """
        batch_drift_curve = BatchPredictionProcessor.drift_curve(
            first_tensor, second_tensor
        )
        return batch_drift_curve[:, 1:] - batch_drift_curve[:, :-1]

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
    def apply_function_on_separate_results(
        batch_data: Dict[EvaluationFields, Dict[str, torch.Tensor]],
        fields_to_compute: List[EvaluationFields],
        function_to_apply,
    ) -> Dict[EvaluationFields, Dict[str, torch.Tensor]]:
        """
        Applies the provided function on all selected evaluation results.

        :param batch_data: Batch data to apply the results on.
        :param fields_to_compute: Evaluation fields on which we want to apply the provided function.
        :param function_to_apply: Function that we want to apply on the evaluation results.
        :return: Returns dictionary of all evaluation results after application of the provided function.
        """
        results = {}

        for evaluation_variant in fields_to_compute:
            # For each selected variant from the evaluation result apply the function.
            variant_data = batch_data[evaluation_variant]

            results[evaluation_variant] = {
                layer: function_to_apply(layer_data)
                for layer, layer_data in variant_data.items()
            }

        return results

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
            # Apply function for all selected pairs of evaluation result types.
            field1, field2 = pair.value
            variant_data1 = batch_data[field1]
            variant_data2 = batch_data[field2]

            results[pair] = BatchPredictionProcessor.apply_function_across_layers(
                variant_data1, variant_data2, function_to_apply
            )

        return results

    @staticmethod
    def set_proper_parameters_to_each_batch_action() -> (
        Dict[BatchSummaryFields, Dict[BatchJobParameters, Any]]
    ):
        """
        Initializes parameters for the batch job call to compute the arbitrary evaluation metric
        from the batch.

        :return: Returns dictionary of evaluation result pairs and function that needs to be passed
        to a batch job in order to compute the given batch summary specified in the dictionary key.
        """
        all_evaluation_pairs = list(EvaluationPairsVariants)
        all_target_prediction_pairs = [
            EvaluationPairsVariants.PREDICTION_TARGET_PAIR,
            EvaluationPairsVariants.TRAIN_LIKE_TARGET_PAIR,
        ]
        prediction_to_train_like_pair = [
            EvaluationPairsVariants.PREDICTION_TRAIN_LIKE_PAIR
        ]
        all_evaluation_fields = [
            EvaluationFields.PREDICTIONS,
            EvaluationFields.TARGETS,
            EvaluationFields.TRAIN_LIKE_PREDICTION,
        ]

        return {
            # MSE per experiment
            BatchSummaryFields.MSE: {
                BatchJobParameters.PAIRS_TO_COMPUTE: all_evaluation_pairs,
                BatchJobParameters.FUNCTION_TO_APPLY: BatchPredictionProcessor.mse_per_example,
            },
            # Pearson's CC per experiment
            BatchSummaryFields.PEARSON: {
                BatchJobParameters.PAIRS_TO_COMPUTE: all_evaluation_pairs,
                BatchJobParameters.FUNCTION_TO_APPLY: BatchPredictionProcessor.pearson_per_experiment,
            },
            # Synchrony curve per experiment
            BatchSummaryFields.SYNCHRONY: {
                BatchJobParameters.EVALUATION_FIELDS_TO_COMPUTE: all_evaluation_fields,
                BatchJobParameters.FUNCTION_TO_APPLY: BatchPredictionProcessor.synchrony_curve,
            },
            # MSE of synchrony curve per experiment
            BatchSummaryFields.MSE_SYNCHRONY: {
                BatchJobParameters.PAIRS_TO_COMPUTE: all_evaluation_pairs,
                BatchJobParameters.FUNCTION_TO_APPLY: BatchPredictionProcessor.mse_between_synchrony_curves,
            },
            # Pearson's CC of synchrony curve per experiment
            BatchSummaryFields.PEARSON_SYNCHRONY: {
                BatchJobParameters.PAIRS_TO_COMPUTE: all_evaluation_pairs,
                BatchJobParameters.FUNCTION_TO_APPLY: BatchPredictionProcessor.pearson_synchrony,
            },
            # Drift curve between the predictions and teacher-forced predictions.
            BatchSummaryFields.DRIFT_FREE_FORCED: {
                BatchJobParameters.PAIRS_TO_COMPUTE: prediction_to_train_like_pair,
                BatchJobParameters.FUNCTION_TO_APPLY: BatchPredictionProcessor.drift_curve,
            },
            # Delta of the drift curve of predictions and teacher-forced predictions.
            BatchSummaryFields.DRIFT_DELTA: {
                BatchJobParameters.PAIRS_TO_COMPUTE: prediction_to_train_like_pair,
                BatchJobParameters.FUNCTION_TO_APPLY: BatchPredictionProcessor.drift_delta,
            },
        }

    @staticmethod
    def process_batch_results(
        batch_data: Dict[EvaluationFields, Dict[str, torch.Tensor]],
    ) -> Dict[
        BatchSummaryFields,
        Dict[EvaluationPairsVariants | EvaluationFields, Dict[str, torch.Tensor]],
    ]:
        """
        Processes one batch with all specified computations of the statistical properties.

        :param batch_data: Batch data to load.
        :return: Returns all statistics for the provided batch of the data.
        """
        all_batch_action_parameters = (
            BatchPredictionProcessor.set_proper_parameters_to_each_batch_action()
        )

        separate_batch_results = {
            batch_summary_field: BatchPredictionProcessor.apply_function_on_separate_results(
                batch_data, *all_batch_action_parameters[batch_summary_field].values()
            )
            for batch_summary_field in all_batch_action_parameters
            if batch_summary_field in BatchPredictionProcessor.separate_summaries
            # If summary field is paired -> apply paired batch computation.
        }

        paired_batch_results = {
            batch_summary_field: BatchPredictionProcessor.apply_function_between_selected_pairs_of_results(
                batch_data, *all_batch_action_parameters[batch_summary_field].values()
            )
            for batch_summary_field in all_batch_action_parameters
            if batch_summary_field in BatchPredictionProcessor.paired_summaries
            # If summary field is paired -> apply paired batch computation.
        }

        return {**separate_batch_results, **paired_batch_results}
