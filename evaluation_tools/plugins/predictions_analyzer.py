"""
This script process the results of the evaluation runs.
"""

import os
import re
from typing import Dict, List

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

import nn_model.globals
from evaluation_tools.fields.experiment_parameters_fields import (
    WandbExperimentVariants,
    GridSearchRunVariants,
    EvaluationRunVariants,
    AdditionalExperiments,
    NUM_EVALUATION_SUBSETS,
    NUM_EVALUATION_BATCHES,
)
from evaluation_tools.plugins.wandb_processor import WandbProcessor
from evaluation_tools.scripts.pickle_manipulation import load_pickle_file
from nn_model.type_variants import EvaluationFields
from evaluation_tools.plugins.batch_prediction_processor import BatchPredictionProcessor
from evaluation_tools.fields.prediction_analysis_fields import (
    BatchSummaryFields,
    EvaluationPairsVariants,
)


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

    full_evaluation_subdirectory = "/full_evaluation_results/"

    def __init__(
        self,
        # all_wandb_variants: Dict[
        #     WandbExperimentVariants,
        #     List[GridSearchRunVariants | EvaluationRunVariants | AdditionalExperiments],
        # ] = {},
        evaluation_variants: List[EvaluationRunVariants] = [],
        additional_variants: List[AdditionalExperiments] = [],
    ):
        # All wandb projects that we want to extract results from.
        # self.all_wandb_variants = all_wandb_variants
        # All evaluation runs on multiple model subsets that we want to run the evaluation on.
        self.evaluation_variants = evaluation_variants

        # All runs from other model experiments that we want to process.
        # TODO: (currently not used).
        self.additional_variants = additional_variants

        self.all_model_variant_results: Dict[
            EvaluationRunVariants,
            Dict[
                int,
                Dict[
                    int,
                    Dict[
                        BatchSummaryFields,
                        Dict[
                            EvaluationPairsVariants | EvaluationFields,
                            Dict[str, torch.Tensor],
                        ],
                    ],
                ],
            ],
        ] = {}

    # def to_numpy(self) -> Dict[
    #     EvaluationRunVariants,
    #     Dict[
    #         int,
    #         Dict[
    #             int,
    #             Dict[
    #                 BatchSummaryFields,
    #                 Dict[
    #                     EvaluationPairsVariants | EvaluationFields,
    #                     Dict[str, np.array],
    #                 ],
    #             ],
    #         ],
    #     ],
    # ]:
    #     return {
    #         evaluation_variant: {}
    #         for evaluation_variant, variant_value in self.all_model_variant_results.items()
    #     }

    def all_model_variants_summary_to_pandas(
        self,
        all_results: Dict[
            EvaluationRunVariants,
            Dict[
                int,
                Dict[
                    int,
                    Dict[
                        BatchSummaryFields,
                        Dict[
                            EvaluationPairsVariants | EvaluationFields,
                            Dict[str, torch.Tensor],
                        ],
                    ],
                ],
            ],
        ] = {},
    ) -> pd.DataFrame:
        rows = []

        if not all_results:
            all_results = self.all_model_variant_results

        for model_variant, model_variant_subsets in all_results.items():
            for subset_id, batches in model_variant_subsets.items():
                for batch_id, summaries in batches.items():
                    for summary_type, summary_results in summaries.items():
                        for (
                            prediction_variant,
                            all_layer_metrics,
                        ) in summary_results.items():
                            for (
                                layer_name,
                                layer_value,
                            ) in all_layer_metrics.items():
                                if isinstance(layer_value, torch.Tensor):
                                    layer_value = (
                                        layer_value.item()
                                        if layer_value.numel() == 1
                                        else layer_value.detach().cpu().numpy()
                                    )
                                rows.append(
                                    {
                                        "model_variant": model_variant,
                                        "subset_id": subset_id,
                                        "batch_id": batch_id,
                                        "summary_type": summary_type,
                                        "prediction_variant": prediction_variant,
                                        "layer_name": layer_name,
                                        "value": layer_value,
                                    }
                                )

        return pd.DataFrame(rows)

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
    def process_all_batches(responses_dir: str, subset: int = -1) -> Dict[
        int,
        Dict[
            BatchSummaryFields,
            Dict[EvaluationPairsVariants | EvaluationFields, Dict[str, torch.Tensor]],
        ],
    ]:
        """
        Processes all batches of the given model.

        :param responses_dir: Directory where all batch predictions for the model are stored.
        :param subset: In case we want to process only part of the dataset, defaults to -1
        :return: Returns dictionary of batch index as a key and its summary results as a value.
        """

        responses_filenames = PredictionsAnalyzer._load_all_responses_filenames(
            responses_dir
        )
        if subset > 0:
            # Take only subset of responses.
            responses_filenames = responses_filenames[:subset]

        processed_subset_results: Dict[
            int,
            Dict[
                BatchSummaryFields,
                Dict[
                    EvaluationPairsVariants | EvaluationFields, Dict[str, torch.Tensor]
                ],
            ],
        ] = {}

        for response_filename in responses_filenames:
            # Process all batches.
            batch_id = int(response_filename.split("_")[1].split(".")[0])

            batch_evaluation_results = (
                PredictionsAnalyzer._get_batch_from_responses_file(
                    response_filename,
                    responses_dir,
                    PredictionsAnalyzer.evaluation_fields_to_load,
                )
            )
            processed_subset_results[batch_id] = (
                BatchPredictionProcessor.process_batch_results(batch_evaluation_results)
            )

        return processed_subset_results

    @staticmethod
    def process_all_subset_variants_predictions(
        base_responses_dir: str, max_num_subsets: int = -1
    ) -> Dict[
        int,
        Dict[
            int,
            Dict[
                BatchSummaryFields,
                Dict[
                    EvaluationPairsVariants | EvaluationFields,
                    Dict[str, torch.Tensor],
                ],
            ],
        ],
    ]:
        """
        Processes predictions of all model subset variants.

        :param base_responses_dir: Base directory where all model subset variant predictions are stored.
        :param max_num_subsets: Maximum number of model subset variants to process, if -1 then all.
        :return: Returns dictionary with key specifying subset ID and value dictionary of all batch summaries.
        """

        counter = 0
        pattern = re.compile(r"_sub-var-(\d+)_")  # Regex to capture var_num

        all_model_subset_variant_analyses: Dict[
            int,
            Dict[
                int,
                Dict[
                    BatchSummaryFields,
                    Dict[
                        EvaluationPairsVariants | EvaluationFields,
                        Dict[str, torch.Tensor],
                    ],
                ],
            ],
        ] = {}

        for response_subset_variant_directory in tqdm(os.listdir(base_responses_dir)):
            # Iterate through the base directory and find all subset variants model responses.
            full_path_to_subset = os.path.join(
                base_responses_dir, response_subset_variant_directory
            )
            if os.path.isdir(full_path_to_subset):
                # The content of the directory is a directory -> check for match with model variant.
                match = pattern.search(response_subset_variant_directory)
                if match:
                    # The path exists and corresponds to some model variant
                    # -> load all batches of the predictions
                    variant_number = int(match.group(1))
                    all_model_subset_variant_analyses[variant_number] = (
                        PredictionsAnalyzer.process_all_batches(full_path_to_subset)
                    )

                    counter += 1
                    if max_num_subsets > 0 and counter >= max_num_subsets:
                        # Process only subset of model variants.
                        break

        return all_model_subset_variant_analyses

    def process_all_model_variants_predictions(
        self, base_dir: str, evaluation_variants: List[EvaluationRunVariants] = []
    ) -> Dict[
        EvaluationRunVariants,
        Dict[
            int,
            Dict[
                int,
                Dict[
                    BatchSummaryFields,
                    Dict[
                        EvaluationPairsVariants | EvaluationFields,
                        Dict[str, torch.Tensor],
                    ],
                ],
            ],
        ],
    ]:
        """
        Processes all model variants, their all subset variant and batches.

        :param base_dir: Base directory containing all evaluation results of all model variants.
        NOTE: The subdirectories containing the model variants results must match the values from
        `EvaluationRunVariants` values selected for the processing.
        :param evaluation_variants: List of variants to process, if `[]` then use default from `self`.
        :return: Returns evaluation results analysis for each batch of each model
        subset and each model variant.
        """

        if not evaluation_variants:
            evaluation_variants = self.evaluation_variants

        for evaluation_variant in evaluation_variants:
            # Run through all model variants.
            variant_base_dir = (
                base_dir
                + "/"
                + evaluation_variant.value
                + PredictionsAnalyzer.full_evaluation_subdirectory
            )
            print(f"Processing model variant: {evaluation_variant.value}")
            self.all_model_variant_results[evaluation_variant] = (
                PredictionsAnalyzer.process_all_subset_variants_predictions(
                    variant_base_dir
                )
            )

        return self.all_model_variant_results

    @staticmethod
    def _extract_original_results_to_new_order(
        model_subsets: Dict[
            int,
            Dict[
                int,
                Dict[
                    BatchSummaryFields,
                    Dict[
                        EvaluationPairsVariants | EvaluationFields,
                        Dict[str, torch.Tensor],
                    ],
                ],
            ],
        ],
    ) -> Dict[
        BatchSummaryFields,
        Dict[EvaluationPairsVariants | EvaluationFields, Dict[str, List[np.ndarray]]],
    ]:
        model_variant_reordered = {}

        # Sort model_subset_ids and batch_ids
        sorted_model_subsets = sorted(
            model_subsets.items()
        )  # List of (model_subset_id, batches)

        for model_subset_id, batches in sorted_model_subsets:
            sorted_batches = sorted(batches.items())  # List of (batch_id, layers)

            for batch_id, layers in sorted_batches:
                for evaluation_variant, batch_summary_values in layers.items():
                    if evaluation_variant not in model_variant_reordered:
                        model_variant_reordered[evaluation_variant] = {}

                    for (
                        evaluation_entities_involved,
                        all_layer_metrics,
                    ) in batch_summary_values.items():
                        if (
                            evaluation_entities_involved
                            not in model_variant_reordered[evaluation_variant]
                        ):
                            model_variant_reordered[evaluation_variant][
                                evaluation_entities_involved
                            ] = {}

                        for layer_name, tensor in all_layer_metrics.items():
                            array = tensor.detach().cpu().numpy()

                            # Init 2D structure: [model_subset][batch] = value
                            layer_store = model_variant_reordered[evaluation_variant][
                                evaluation_entities_involved
                            ].setdefault(
                                layer_name,
                                [
                                    [None for _ in range(NUM_EVALUATION_BATCHES)]
                                    for _ in range(NUM_EVALUATION_SUBSETS)
                                ],
                            )

                            # Map index from ID
                            # ms_idx = model_subset_ids.index(model_subset_id)
                            # b_idx = batch_ids.index(batch_id)

                            layer_store[model_subset_id][batch_id] = array

        return model_variant_reordered

    @staticmethod
    def _pad_batches(array_list_to_pad: List[np.ndarray]) -> List[np.ndarray]:
        """
        Pads all batch arrays to the maximal size with 0 at the end (typically in the time dimension,
        where we are missing the prediction sequence sometimes).

        :param array_list_to_pad: List of batched arrays.
        :return: Returns list of padded batches of data to uniform shape.
        """
        # Find the maximum shape across all arrays
        max_shape = np.array([a.shape for a in array_list_to_pad]).max(axis=0)

        # Pad each array with 0s at the end of each dimension
        padded_arrays = []
        for arr in array_list_to_pad:
            pad_width = [
                (0, max_dim - cur_dim) for cur_dim, max_dim in zip(arr.shape, max_shape)
            ]
            padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
            padded_arrays.append(padded)

        return padded_arrays

    def merge_all_subsets_and_batches(
        self,
        original_variants: Dict[
            EvaluationRunVariants,
            Dict[
                int,
                Dict[
                    int,
                    Dict[
                        BatchSummaryFields,
                        Dict[
                            EvaluationPairsVariants | EvaluationFields,
                            Dict[str, torch.Tensor],
                        ],
                    ],
                ],
            ],
        ] = {},
    ) -> Dict[
        EvaluationRunVariants,
        Dict[
            BatchSummaryFields,
            Dict[EvaluationPairsVariants | EvaluationFields, np.ndarray],
        ],
    ]:
        merged = {}

        if not original_variants:
            original_variants = self.all_model_variant_results

        for model_variant, model_variant_value in original_variants.items():

            variant_merged = PredictionsAnalyzer._extract_original_results_to_new_order(
                model_variant_value
            )

            # After collecting all lists, stack into a proper np.array with zero-padding
            for evaluation_variant in variant_merged:
                for evaluation_entities_involved in variant_merged[evaluation_variant]:
                    for layer_name, np_arrays_list in variant_merged[
                        evaluation_variant
                    ][evaluation_entities_involved].items():
                        # Convert all tensors to numpy arrays
                        padded_arrays = PredictionsAnalyzer._pad_batches(np_arrays_list)

                        # Stack into a single array -> shape: `[num_batches, ...]``
                        stacked = np.stack(padded_arrays, axis=0)

                        # Reshape into (model_subset, batch, ...) if dimensions match
                        num_model_subsets = len(model_variant_value)
                        num_batches = len(next(iter(model_variant_value.values()), {}))
                        if len(stacked) == num_model_subsets * num_batches:
                            stacked = stacked.reshape(
                                (num_model_subsets, num_batches) + stacked.shape[1:]
                            )

                        variant_merged[evaluation_variant][
                            evaluation_entities_involved
                        ][layer_name] = stacked
            merged[model_variant] = variant_merged

        return merged


if __name__ == "__main__":
    evaluation_results_base_dir = (
        "/home/david/source/diplomka/thesis_results/evaluation/"
    )
    model_variants = [
        EvaluationRunVariants.DNN_JOINT,
        EvaluationRunVariants.DNN_SEPARATE,
    ]
    additional_experiments = []

    # wandb_variants = {}
    prediction_analyzer = PredictionsAnalyzer(
        evaluation_variants=model_variants,
        additional_variants=additional_experiments,
    )
    result = prediction_analyzer.process_all_model_variants_predictions(
        evaluation_results_base_dir
    )

    merged_result = prediction_analyzer.merge_all_subsets_and_batches()

    print(merged_result)
