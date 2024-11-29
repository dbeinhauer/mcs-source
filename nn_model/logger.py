"""
This source defines custom logger class used in model execution.
"""

import logging

import wandb


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

    def print_best_model_update(self, previous_best: float, current_metric: float):
        """
        Prints information while updating the new best model.

        :param previous_best: Previous best metric value.
        :param current_best: Current best metric value.
        """
        self.logger.info(
            " ".join(
                [
                    f"Validation metric improved from {previous_best:.4f}",
                    f"to {current_metric:.4f}. Saving model...",
                ]
            )
        )

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
        self.logger.info(
            " ".join(
                [
                    f"Epoch [{epoch_num}/{total_num_epochs}],",
                    "Average Loss: {avg_epoch_loss:.4f}",
                ]
            )
        )

    def print_best_model_evaluation(self, best_metric: float):
        """
        Prints logger info while loading the model with best evaluation score.

        :param best_metric: _description_
        """
        self.logger.info(
            f"Running final evaluation on model with best CC_NORM value: {best_metric:.4f}"
        )

    def wand_batch_evaluation_logs(self, cc_norm: float, cc_abs: float):
        """
        Writes logs to `wandb` regarding the batch evaluation metrics.

        :param cc_norm: Batch CC_NORM value.
        :param cc_abs: Batch CC_ABS value.
        """
        wandb.log({"batch_cc_norm": cc_norm})
        wandb.log({"batch_cc_abs": cc_abs})

    def print_current_evaluation_status(
        self, step_num: int, cc_norm_sum: float, cc_abs_sum: float
    ):
        """
        Prints status of the current evaluation.

        :param step_num: Current evaluation step.
        :param cc_norm_sum: Current sum of all CC_NORM values.
        :param cc_abs_sum: Current sum of all CC_ABS values.
        """
        self.logger.info(
            "".join(
                [
                    f"Average normalized cross correlation after step {step_num} is: ",
                    f"{cc_norm_sum/(step_num):.4f}",
                    "\n",
                    "Average Pearson's CC is: ",
                    f"{cc_abs_sum/(step_num):.4f}",
                ]
            )
        )

    def print_final_evaluation_results(self, avg_cc_norm: float, avg_cc_abs: float):
        """
        Prints final evaluation results and stores them also to `wandb` logs.

        :param avg_cc_norm: Average CC_NORM value.
        :param avg_cc_abs: Average CC_ABS value.
        """
        self.logger.info(
            f"Final average normalized cross correlation is: {avg_cc_norm:.4f}"
        )
        self.logger.info(f"Final average Pearson's CC is: {avg_cc_abs:.4f}")
        wandb.log({"CC_NORM": avg_cc_norm})
        wandb.log({"CC_ABS": avg_cc_abs})
