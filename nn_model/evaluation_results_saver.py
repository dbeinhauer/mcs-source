"""
This source defines simple class that is used to save and load best model. 
And save model evaluation results.
"""

import os
from typing import Dict

import pickle
import torch

from nn_model.type_variants import EvaluationFields, PredictionTypes


class EvaluationResultsSaver:
    """
    Class that serves for storing and loading the files during model execution process.
    """

    def __init__(self, arguments):
        self.best_model_path = arguments.model_dir + arguments.model_filename
        if arguments.best_model_dir:
            # In case we specify the path of the best model -> use it
            self.best_model_path = arguments.best_model_dir

        self.full_evaluation_directory = arguments.full_evaluation_dir

    def save_predictions_batch(
        self,
        batch_index: int,
        all_predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        filename: str = "",
    ):
        """
        Saves averaged predictions and targets across the trials for the provided batch.

        :param batch_index: Index of the batch (just used for naming).
        :param predictions: Dictionary of predictions of the batch for all layers.
        Shape of tensor:(batch_size, time, num_neurons)
        :param targets: Dictionary of targets of the batch for all layers.
        Shape of tensor:(batch_size, time, num_neurons)
        :param filename: optional path of the file, if empty string then use default filename.
        Default filename:
            f"{args.full_evaluation_directory}/{args.model_filename}/batch_{batch_index}.pkl"
        """

        if not filename:
            # If the filename is not defined -> use the default one
            # Path: {full_evaluation_directory}/{model_filename}/batch_{batch_index}.pkl
            subdirectory_name = os.path.splitext(
                os.path.basename(self.best_model_path)
            )[0]
            subdirectory_path = os.path.join(
                self.full_evaluation_directory, subdirectory_name
            )
            os.makedirs(subdirectory_path, exist_ok=True)
            filename = os.path.join(subdirectory_path, f"batch_{batch_index}.pkl")

        # Save to a pickle file
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    EvaluationFields.PREDICTIONS.value: all_predictions[
                        PredictionTypes.FULL_PREDICTION.value
                    ],
                    EvaluationFields.TARGETS.value: targets,
                    EvaluationFields.RNN_PREDICTIONS.value: all_predictions[
                        PredictionTypes.RNN_PREDICTION.value
                    ],
                },
                f,
            )

    def save_full_evaluation(
        self,
        batch_index: int,
        all_predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ):
        """
        Saves all evaluation results together with its targets in appropriate format to
        pickle file.

        :param batch_index: ID of the batch used to determine filename (batch ID is part of it).
        :param predictions: Dictionary of predictions of the batch for all layers.
        Shape of tensor:(batch_size, time, num_neurons)
        :param targets: Dictionary of targets of the batch for all layers.
        Shape of tensor:(batch_size, time, num_neurons)
        """
        self.save_predictions_batch(
            batch_index,
            {
                prediction_type: {
                    layer: torch.mean(
                        prediction, dim=0
                    )  # Trials dimension is the first because we reshape it during prediction step.
                    for layer, prediction in predictions.items()
                }
                for prediction_type, predictions in all_predictions.items()
            },
            {
                layer: torch.mean(
                    target, dim=1
                )  # We want to skip the first time step + the trials dimension is the second.
                for layer, target in targets.items()
            },
        )
