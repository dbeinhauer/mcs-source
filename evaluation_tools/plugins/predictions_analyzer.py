"""
This script process the results of the evaluation runs.
"""

import os
import re
from typing import Dict, List

from tqdm import tqdm
import torch

import nn_model.globals
from evaluation_tools.fields.experiment_parameters_fields import (
    WandbExperimentVariants,
    GridSearchRunVariants,
    EvaluationRunVariants,
    AdditionalExperiments,
)
from evaluation_tools.plugins.wandb_processor import WandbProcessor
from evaluation_tools.scripts.pickle_manipulation import load_pickle_file
from nn_model.type_variants import EvaluationFields

from evaluation_tools.plugins.batch_prediction_processor import BatchPredictionProcessor


class PredictionsAnalyzer:
    """
    This class processes the evaluation results.
    """

    predictions_base_path = f"{nn_model.globals.PROJECT_ROOT}/thesis_results/"

    # Fields to load from prediction files.
    prediction_fields = [
        EvaluationFields.PREDICTIONS,
        EvaluationFields.TARGETS,
    ]

    # Predictions when we reset the previous time step with target.
    train_like_predictions = [
        EvaluationFields.TRAIN_LIKE_PREDICTION,
        EvaluationFields.TARGETS,
    ]

    # TODO: Prediction fields of the outer RNN (currently not functional).
    rnn_to_neuron_fields = [
        EvaluationFields.PREDICTIONS,
        EvaluationFields.RNN_PREDICTIONS,
    ]

    # All evaluation fields that we are interested in our analysis.
    evaluation_fields_to_load = [
        EvaluationFields.PREDICTIONS,
        EvaluationFields.TARGETS,
        EvaluationFields.TRAIN_LIKE_PREDICTION,
    ]

    def __init__(
        self,
        all_wandb_variants: Dict[
            WandbExperimentVariants,
            List[GridSearchRunVariants | EvaluationRunVariants | AdditionalExperiments],
        ] = {},
        evaluation_variants: List[EvaluationRunVariants] = [],
        additional_variants: List[AdditionalExperiments] = [],
    ):
        self.all_wandb_variants = all_wandb_variants
        self.evaluation_variants = evaluation_variants
        self.additional_variants = additional_variants

        # Load wandb results.
        self.wandb_processor = WandbProcessor(all_wandb_variants)

    @staticmethod
    def _load_all_responses_filenames(responses_dir: str = "") -> List[str]:
        """
        Loads all filenames from the responses directory (batches of responses).

        :param responses_dir: Path to directory containing neuronal responses,
        if `""` then do not search for response files).
        :returns: Returns list of paths to neuronal responses.
        """
        if responses_dir:
            return os.listdir(os.path.join(responses_dir))

        return []

    @staticmethod
    def _get_batch_from_responses_file(
        filename: str,
        responses_dir: str,
        keys_to_select: List[EvaluationFields],
    ) -> Dict[EvaluationFields, Dict[str, torch.Tensor]]:
        """
        Loads batch of selected evaluation results from the provided file.

        :param filename: File containing the evaluation results.
        :param responses_dir: Directory of all batch evaluation results for the model.
        :param keys_to_select: List of evaluation fields to load.
        :return: Returns loaded batch of selected evaluation results.
        """
        return {
            key: (
                {
                    layer: layer_value[:, 1:, :]
                    for layer, layer_value in all_layers_value.items()
                }
                if key == EvaluationFields.TARGETS
                else all_layers_value
            )  # Skip the first time step in case the loaded values are targets
            # (in the predictions we do not predict the first time step)
            for key, all_layers_value in load_pickle_file(
                responses_dir + "/" + filename
            ).items()
            if key in keys_to_select  # Take only selected Evaluation fields.
        }

    @staticmethod
    def load_all_model_predictions(responses_dir: str, subset: int = -1):

        responses_filenames = PredictionsAnalyzer._load_all_responses_filenames(
            responses_dir
        )
        if subset > 0:
            # Take only subset of responses.
            responses_filenames = responses_filenames[:subset]

        for i, response_filename in enumerate(tqdm(responses_filenames)):

            batch_evaluation_results = (
                PredictionsAnalyzer._get_batch_from_responses_file(
                    response_filename,
                    responses_dir,
                    PredictionsAnalyzer.evaluation_fields_to_load,
                )
            )
            return batch_evaluation_results

    def load_all_subset_variants_predictions(
        self, base_responses_dir: str, max_num_subsets: int = -1
    ):

        counter = 0
        pattern = re.compile(r"_sub-var-(\d+)_")  # Regex to capture var_num

        for response_subset_variant_directory in tqdm(os.listdir(base_responses_dir)):
            # Iterate through the base directory and find all subset variants model responses.
            full_path = os.path.join(
                base_responses_dir, response_subset_variant_directory
            )
            if os.path.isdir(full_path):
                # The content of the directory is a directory -> check for match with model variant.
                match = pattern.search(response_subset_variant_directory)
                if match:
                    # The path exists and corresponds to some model variant
                    # -> load all batches of the predictions
                    all_subset_variant_responses = (
                        PredictionsAnalyzer.load_all_model_predictions(full_path)
                    )

                    return BatchPredictionProcessor.process_batch_results(
                        all_subset_variant_responses
                    )

                    counter += 1
                    if max_num_subsets > 0 and counter >= max_num_subsets:
                        # Process only subset of model variants.
                        break


if __name__ == "__main__":
    base_dir = "/home/david/source/diplomka/testing_results/"
    prediction_analyzer = PredictionsAnalyzer()
    result = prediction_analyzer.load_all_subset_variants_predictions(base_dir)

    print(result)
