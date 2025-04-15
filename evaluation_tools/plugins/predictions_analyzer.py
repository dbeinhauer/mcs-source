"""
This script process the results of the evaluation runs.
"""

import os
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


class PredictionsAnalyzer:
    """
    This class processes the evaluation results.
    """

    predictions_base_path = f"{nn_model.globals.PROJECT_ROOT}/thesis_results/"

    # Fields to load from prediction files.
    prediction_fields = [
        EvaluationFields.PREDICTIONS.value,
        EvaluationFields.TARGETS.value,
    ]

    # TODO: Prediction fields of the outer RNN (currently not functional).
    rnn_to_neuron_fields = [
        EvaluationFields.PREDICTIONS.value,
        EvaluationFields.RNN_PREDICTIONS.value,
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
    def _get_data_from_responses_file(
        filename: str, responses_dir: str, keys_to_select
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in load_pickle_file(responses_dir + "/" + filename).items()
            if key in keys_to_select
        }

    def load_all_model_responses(self, responses_dir: str, subset: int = -1):

        responses_filenames = PredictionsAnalyzer._load_all_responses_filenames(
            responses_dir
        )
        if subset > 0:
            # Take only subset of responses.
            responses_filenames = responses_filenames[:subset]

        for i, response_filename in enumerate(tqdm(responses_filenames)):

            all_predictions_and_targets = (
                PredictionsAnalyzer._get_data_from_responses_file(
                    response_filename,
                    responses_dir,
                    PredictionsAnalyzer.prediction_fields,
                )
            )
