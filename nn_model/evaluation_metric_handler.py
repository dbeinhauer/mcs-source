from collections import defaultdict
from typing import Dict, List, Union, NamedTuple

import torch

from nn_model.evaluation_metrics import Metric, NormalizedCrossCorrelation
from nn_model.type_variants import (
    EvaluationMetricVariants, LayerType)
import nn_model.globals
from nn_model.visible_neurons_handler import VisibleNeuronsHandler


class EvaluationMetricHandler:
    """
    Class that handles the evaluation metrics for the model.
    """
    
    def __init__(self, evaluation_metric):
        self.all_metrics_sum = self.init_all_metrics()
        self.evaluation_metric = evaluation_metric
    
    def init_all_metrics(self) -> Dict[EvaluationMetricVariants, Union[Metric, Dict[str, Metric]]]:
        """
        Resets all metrics to zero.
        """
        return {
            EvaluationMetricVariants.FULL_METRIC: Metric(0, 0, 0),
            EvaluationMetricVariants.LAYER_SPECIFIC: defaultdict(lambda: Metric(0, 0, 0)),
            EvaluationMetricVariants.VISIBLE_METRIC: Metric(0, 0, 0),
            EvaluationMetricVariants.INVISIBLE_METRIC: Metric(0, 0, 0),
        }
        
    def reset_all_metrics(self):
        """
        Resets all metrics to zero.
        """
        self.all_metrics_sum = self.init_all_metrics()
        
    def add_to_sum(self, metrics: Dict[EvaluationMetricVariants, Union[Metric, Dict[str, Metric]]]):
        """
        Adds the provided metrics to the sum of all metrics.

        :param metrics: Dictionary of metrics to be added.
        """
        for key, value in metrics.items():
            if isinstance(value, Metric):
                self.all_metrics_sum[key] += value
            else:
                for layer, metric in value.items():
                    self.all_metrics_sum[key][layer] += metric
                    
    def divide_from_sum(self, divisor: int) -> Dict[EvaluationMetricVariants, Union[Metric, Dict[str, Metric]]]:
        """
        Divides all metrics in the sum by the provided divisor.

        :param divisor: Divided metrics.
        """
        divided_metric = self.all_metrics_sum.copy()
        for metric_variant, metric_value in divided_metric.items():
            if isinstance(metric_value, Metric):
                divided_metric[metric_variant] /= divisor
            else:
                for layer in metric_value:
                    divided_metric[metric_variant][layer] /= divisor
                    
        return divided_metric
    
    
    def _compute_all_layers_evaluation(self, targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], keys: List[str]) -> Metric:
        """
        Computes evaluation score between vectors of predictions
        and targets across all layers.

        :param targets: Dictionary of targets for all layers.
        :param predictions: Dictionary of predictions for all layers.
        :param keys: Layer names sorted in expected order.
        :param evaluation_metric: Evaluation metric to be used for the evaluation.
        :return: Returns evaluation score between provided values across all layers.
        """
        def cat(tensors: Dict[str, torch.Tensor]):
            return torch.cat(
                [tensors[k] for k in keys],
                dim=-1,
            )

        # Concatenate predictions and targets across all layers.
        all_predictions = cat(predictions).to(nn_model.globals.DEVICE)
        all_targets = cat(targets).to(nn_model.globals.DEVICE)

        # Run the calculate function once on the concatenated tensors.
        return self.evaluation_metric.calculate(all_predictions, all_targets)
    
    def _compute_layer_specific_evaluation(self, targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, Metric]:
        layer_specific_metrics = {}
        for layer in keys:
            metric = self.evaluation_metric.calculate(
                predictions[layer].to(nn_model.globals.DEVICE),
                targets[layer].to(nn_model.globals.DEVICE),
            )
            layer_specific_metrics[layer] = metric
            
        return layer_specific_metrics
    
    def compute_all_evaluation_scores(
        self, targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], visible_neurons_handler: VisibleNeuronsHandler
    ) -> Dict[EvaluationMetricVariants, Union[Metric, Dict[str, Metric]]]:
        """
        Computes evaluation score between vectors of all prediction
        and all target layers for all variants.

        Between two vectors that were created by concatenating all predictions and
        all targets for all output layers.

        :param targets: dictionary of targets for all layers.
        :param predictions: dictionary of predictions for all layers.
        :param visible_neurons_handler: Handler for splitting visible and invisible neurons.
        :param evaluation_metric: Evaluation metric to be used for the evaluation.
        :return: Returns all evaluation variants. Overall evaluation score (CC_NORM and CC_ABS),
        over each layer separately, over visible and over invisible neurons separately.
        """
        # Avoid dict insertion order issue
        keys = list(targets.keys())
        all_metrics = self.init_all_metrics()

        all_metrics[EvaluationMetricVariants.FULL_METRIC] = self._compute_all_layers_evaluation(targets, predictions, keys)

        # batch_size =  targets[LayerType.V1_EXC_L4.value].shape[0]
        visible_targets, invisible_targets = visible_neurons_handler.split_visible_invisible_neurons(targets)
        visible_predictions, invisible_predictions = visible_neurons_handler.split_visible_invisible_neurons(predictions)
        all_metrics[EvaluationMetricVariants.VISIBLE_METRIC] = self._compute_all_layers_evaluation(visible_targets, visible_predictions, keys)
        all_metrics[EvaluationMetricVariants.INVISIBLE_METRIC] = self._compute_all_layers_evaluation(invisible_targets, invisible_predictions, keys)
        all_metrics[EvaluationMetricVariants.LAYER_SPECIFIC] = self._compute_layer_specific_evaluation(targets, predictions, keys)

        return all_metrics
