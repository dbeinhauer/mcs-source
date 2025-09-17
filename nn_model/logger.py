"""
This source defines custom logger class used in model execution.
"""

import logging
from typing import Dict, Union

import pandas as pd
import wandb

import nn_model.globals
from nn_model.evaluation_metrics import Metric
from nn_model.type_variants import (
    EvaluationMetricVariants,
)


# pylint: disable=W1203
class LoggerModel:
    """
    This class is used to encapsulate logging of all model execution operations.
    """

    def __init__(self):
        """
        Initializes the logger object.
        """
        self.logger = logging.getLogger(__name__)

    def print_experiment_info(self, arguments):
        """
        Prints basic information about the experiment.

        :param arguments: command line arguments.
        """
        print(
            "\n".join(
                [
                    "NEW EXPERIMENT",
                    "---------------------------------",
                    "Running with parameters:",
                    f"Model size: {nn_model.globals.SIZE_MULTIPLIER}",
                    f"Time step size: {nn_model.globals.TIME_STEP}",
                    f"Train subset size: {arguments.train_subset}",
                    f"Model subset variant: {arguments.subset_variant}",
                    f"Batch size: {arguments.train_batch_size}",
                    f"Learning rate: {arguments.learning_rate}",
                    f"Distance regularizer constant: {arguments.distance_regularizer}",
                    f"Sigma distance regularizer: {arguments.sigma_regularizer}",
                    f"Num epochs: {arguments.num_epochs}",
                    f"Model variant: {arguments.model}",
                    f"Number of hidden time steps: {arguments.num_hidden_time_steps}",
                    f"Neuron number of layers: {arguments.neuron_num_layers}",
                    f"Neurons layer sizes: {arguments.neuron_layer_size}",
                    f"Neuron rnn model variant: {arguments.neuron_rnn_variant}",
                    f"Neuron activation function: {arguments.neuron_activation_function}",
                    f"Neuron use residual connection: {arguments.neuron_residual}",
                    f"Gradient clipping: {arguments.gradient_clip}",
                    f"Optimizer type: {arguments.optimizer_type}",
                    f"Weight initialization: {arguments.weight_initialization}",
                    f"Use synaptic adaptation: {arguments.synaptic_adaptation}",
                    f"Synaptic adaptation layer size: {arguments.synaptic_adaptation_size}",
                    "Synaptic adaptation number of layers: "
                    + str(arguments.synaptic_adaptation_num_layers),
                    f"Visible neurons ratio: {arguments.visible_neurons_ratio}",
                ]
            )
        )

    def print_best_model_update(self, previous_best: float, current_metric: float):
        """
        Prints information while updating the new best model.

        :param previous_best: Previous best metric value.
        :param current_best: Current best metric value.
        """
        print(
            " ".join(
                [
                    f"Validation metric improved from {previous_best:.4f}",
                    f"to {current_metric:.4f}. Saving model...",
                ]
            )
        )

    def wandb_batch_loss(self, avg_time_loss):
        """
        Writes logs to `wandb` of the loss for the current batch.

        :param avg_time_loss: Loss to be logged.
        """
        wandb.log({"batch_loss": avg_time_loss})

    def print_epoch_loss(
        self, epoch_num: int, total_num_epochs: int, avg_epoch_loss: float
    ):
        """
        Prints average epoch loss and also sends it to `wandb`.

        :param epoch_num: Current epoch number.
        :param total_num_epochs: Total number of epochs.
        :param avg_epoch_loss: Average batch loss in epoch.
        """
        wandb.log(
            {
                "epoch_loss": avg_epoch_loss,
            }
        )
        print(
            " ".join(
                [
                    f"Epoch [{epoch_num}/{total_num_epochs}],",
                    f"Average Loss: {avg_epoch_loss:.4f}",
                ]
            )
        )

    def print_best_model_evaluation(self, best_metric: float):
        """
        Prints logger info while loading the model with best evaluation score.

        :param best_metric: Value of the best metric to be printed.
        """
        self.logger.info(
            f"Running final evaluation on model with best CC_NORM value: {best_metric:.4f}"
        )

    def wandb_batch_evaluation_logs(
        self,
        all_evaluation_metric: Dict[
            EvaluationMetricVariants, Union[Metric, Dict[str, Metric]]
        ],
    ):
        """
        Writes logs to `wandb` regarding the batch evaluation metrics.
        """
        for metric_variant, values in all_evaluation_metric.items():
            if isinstance(values, Metric):
                # Skip layer separate metrics.
                wandb.log({f"{metric_variant.value}_batch_cc_norm": values.cc_norm})
                wandb.log({f"{metric_variant.value}_batch_cc_abs": values.cc_abs})
                wandb.log(
                    {
                        f"{metric_variant.value}_batch_cc_abs_separate": values.cc_abs_separate
                    }
                )

    def print_current_evaluation_status(
        self,
        step_num: int,
        cc_norm_sum: float,
        cc_abs_sum: float,
        cc_abs_separate_sum: float,
    ):
        """
        Prints status of the current evaluation.

        :param step_num: Current evaluation step.
        :param cc_norm_sum: Current sum of all CC_NORM values.
        :param cc_abs_sum: Current sum of all CC_ABS values.
        """
        print(
            "".join(
                [
                    f"Average normalized cross correlation after step {step_num} is: ",
                    f"{cc_norm_sum/(step_num):.4f}",
                    "\n",
                    "Average Pearson's CC is: ",
                    f"{cc_abs_sum/(step_num):.4f}",
                    "Average separate Pearson's CC is: ",
                    "\n",
                    f"{cc_abs_separate_sum/(step_num):.4f}",
                ]
            )
        )

    def print_final_evaluation_results(
        self,
        all_avg_metric: Dict[
            EvaluationMetricVariants, Union[Metric, Dict[str, Metric]]
        ],
    ):
        """
        Prints final evaluation results and stores them also to `wandb` logs.

        :param all_avg_metric: Average CC_NORM and CC_ABS values across all layers.
        :param layer_specific: Average CC_NORM and CC_ABS values for each layer.
        """
        print("Final evaluation results:")
        print(30 * "-")
        for metric_variant, avg_metric in all_avg_metric.items():
            if isinstance(avg_metric, Metric):
                if metric_variant == EvaluationMetricVariants.FULL_METRIC:
                    wandb.log({"CC_NORM": avg_metric.cc_norm})
                    wandb.log({"CC_ABS": avg_metric.cc_abs})
                    wandb.log({"CC_SEPARATE_ABS": avg_metric.cc_abs_separate})
                    print(
                        f"Average normalized cross correlation: {avg_metric.cc_norm:.4f}"
                    )
                    print(f"Average Pearson's CC: {avg_metric.cc_abs:.4f}")
                    print(
                        f"Average Separate Pearson's CC: {avg_metric.cc_abs_separate:.4f}"
                    )
                else:
                    # Skip layer separate metrics.
                    wandb.log({f"CC_NORM_{metric_variant.value}": avg_metric.cc_norm})
                    wandb.log({f"CC_ABS_{metric_variant.value}": avg_metric.cc_abs})
                    wandb.log(
                        {
                            f"CC_SEPARATE_ABS_{metric_variant.value}": avg_metric.cc_abs_separate
                        }
                    )
                    print(f"CC_NORM_{metric_variant.value}: {avg_metric.cc_norm:.4f}")
                    print(f"CC_ABS_{metric_variant.value}: {avg_metric.cc_abs:.4f}")
                    print(
                        f"CC_ABS_SEPARATE_{metric_variant.value}: {avg_metric.cc_abs_separate:.4f}"
                    )
            else:
                # log correlation for each layer
                rows = []
                for layer_name, metric in avg_metric.items():
                    name = layer_name
                    # group per-layer metrics by their type (norm or abs)
                    wandb.log({"CC_NORM/" + name: metric.cc_norm})
                    wandb.log({"CC_ABS/" + name: metric.cc_abs})
                    rows.append(
                        {
                            "Layer": name,
                            "CC_NORM": metric.cc_norm,
                            "CC_ABS": metric.cc_abs,
                        }
                    )
                df = pd.DataFrame.from_records(rows).set_index("Layer").sort_index()
                print("\nPer-layer correlation summary")
                print(df.to_string(float_format="%.4f"))
