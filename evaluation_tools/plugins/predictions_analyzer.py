"""
This script processes the results of the evaluation runs.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

import nn_model.globals
from evaluation_tools.fields.experiment_parameters_fields import (
    WandbExperimentVariants,
    GridSearchRunVariants,
    ModelEvaluationRunVariant,
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
    PredictionAnalysisVariants,
    VisibilityVariants,
)
from nn_model.visible_neurons_handler import VisibleNeuronsHandler


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
        evaluation_variants: List[ModelEvaluationRunVariant] = [],
        additional_variants: List[AdditionalExperiments] = [],
    ):
        """
        :param evaluation_variants: Evaluation runs that we want to cover.
        :param additional_variants: Additional experiments that we want to cover.
        """
        # All evaluation runs on multiple model subsets that we want to run the evaluation on.
        self.evaluation_variants = evaluation_variants

        # All runs from other model experiments that we want to process.
        # TODO: (currently not used).
        self.additional_variants = additional_variants

        self.all_model_variant_results: Dict[
            ModelEvaluationRunVariant,
            Dict[
                int,
                Dict[
                    int,
                    Dict[
                        VisibilityVariants,
                        Dict[
                            BatchSummaryFields,
                            Dict[
                                PredictionAnalysisVariants,
                                Dict[str, torch.Tensor],
                            ],
                        ],
                    ],
                ],
            ],
        ] = {}

        self.final_evaluation_results: pd.DataFrame = None

    def to_pandas(self) -> pd.DataFrame:
        """
        Converts results to pandas and stacks all batches in shape:
            `[model_subsets, num_batches, {max_batch_size}]`

        :return: Returns prediction analysis results converted to pandas.
        """
        if not self.final_evaluation_results:
            self.final_evaluation_results = (
                PredictionsAnalyzer._convert_merged_to_pandas(
                    self._merge_all_subsets_and_batches()
                )
            )

        return self.final_evaluation_results

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
    def process_all_batches(
        responses_dir: str, subset: int = -1, visible_neurons_ratio: float = 1.0
    ) -> Dict[
        int,
        Dict[
            VisibilityVariants,
            Dict[
                BatchSummaryFields,
                Dict[PredictionAnalysisVariants, Dict[str, torch.Tensor]],
            ],
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
                VisibilityVariants,
                Dict[
                    BatchSummaryFields,
                    Dict[PredictionAnalysisVariants, Dict[str, torch.Tensor]],
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
            # TODO: Split results to visible and invisible neurons.
            visible_neurons_handler = VisibleNeuronsHandler(visible_neurons_ratio)
            visible_part, invisible_part = {}, {}
            for evaluation_type, evaluation_results in batch_evaluation_results.items():
                visible_part[evaluation_type], invisible_part[evaluation_type] = (
                    visible_neurons_handler.split_visible_invisible_neurons(
                        evaluation_results
                    )
                )

            prepared_for_processing = {
                VisibilityVariants.ALL_NEURONS: batch_evaluation_results,
                VisibilityVariants.VISIBLE_NEURONS: visible_part,
                VisibilityVariants.INVISIBLE_NEURONS: invisible_part,
            }

            # TODO: Add for loop to process all, visible and invisible neurons.
            processed_subset_results[batch_id] = {
                variant: BatchPredictionProcessor.process_batch_results(variant_data)
                for variant, variant_data in prepared_for_processing.items()
            }

        return processed_subset_results

    @staticmethod
    def process_all_subset_variants_predictions(
        base_responses_dir: str, max_num_subsets: int = -1
    ) -> Dict[
        int,
        Dict[
            int,
            Dict[
                VisibilityVariants,
                Dict[
                    BatchSummaryFields,
                    Dict[
                        PredictionAnalysisVariants,
                        Dict[str, torch.Tensor],
                    ],
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
                    VisibilityVariants,
                    Dict[
                        BatchSummaryFields,
                        Dict[
                            PredictionAnalysisVariants,
                            Dict[str, torch.Tensor],
                        ],
                    ],
                ],
            ],
        ] = {}

        for response_subset_variant_directory in os.listdir(base_responses_dir):
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

                    # Determine visible neurons ratio if specified in the directory name.
                    visible_part = 1.0
                    match_visible_part = re.search(
                        r"_visib-(\d+(?:\.\d+)?)_", full_path_to_subset
                    )
                    if match_visible_part:
                        # If the visible part is specified in the directory name, determine it.
                        visible_part = float(match_visible_part.group(1))
                    all_model_subset_variant_analyses[variant_number] = (
                        PredictionsAnalyzer.process_all_batches(
                            full_path_to_subset, visible_neurons_ratio=visible_part
                        )
                    )

                    counter += 1
                    if max_num_subsets > 0 and counter >= max_num_subsets:
                        # Process only subset of model variants.
                        break

        return all_model_subset_variant_analyses

    def process_all_model_variants_predictions(
        self,
        base_dir: str,
        evaluation_variants: List[ModelEvaluationRunVariant] = [],
        evaluation_variant_dir: str = "",
    ) -> Dict[
        ModelEvaluationRunVariant,
        Dict[
            int,
            Dict[
                int,
                Dict[
                    VisibilityVariants,
                    Dict[
                        BatchSummaryFields,
                        Dict[
                            PredictionAnalysisVariants,
                            Dict[str, torch.Tensor],
                        ],
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
        :param evaluation_variant_dir: In case `ModelEvaluationRunVariant.CUSTOM` is selected specify
        directory containing results.
        :return: Returns evaluation results analysis for each batch of each model
        subset and each model variant.
        """

        if not evaluation_variants:
            evaluation_variants = self.evaluation_variants

        for evaluation_variant in tqdm(evaluation_variants):
            variant_subdir = evaluation_variant.value
            if evaluation_variant == ModelEvaluationRunVariant.CUSTOM:
                # In case custom evaluation variant is selected, use the provided directory.
                variant_subdir = evaluation_variant_dir

            # Run through all model variants.
            variant_base_dir = (
                base_dir
                + "/"
                + variant_subdir
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
    def _reformat_original_results_to_new_order(
        model_subsets: Dict[
            int,
            Dict[
                int,
                Dict[
                    VisibilityVariants,
                    Dict[
                        BatchSummaryFields,
                        Dict[
                            PredictionAnalysisVariants,
                            Dict[str, torch.Tensor],
                        ],
                    ],
                ],
            ],
        ],
    ) -> Dict[
        VisibilityVariants,
        Dict[
            BatchSummaryFields,
            Dict[
                PredictionAnalysisVariants,
                Dict[str, List[List[Optional[np.ndarray]]]],
            ],
        ],
    ]:
        """
        Reshapes the original results of one model variant to format that allows merging
        all batches together and creating large np.ndarray out of list of lists of
        batches of each subset variant.

        :param model_subsets: Results of all model subsets.
        :return: Returns reformated results.
        """
        model_variant_reordered: Dict[
            VisibilityVariants,
            Dict[
                BatchSummaryFields,
                Dict[
                    PredictionAnalysisVariants,
                    Dict[str, List[List[Optional[np.ndarray]]]],
                ],
            ],
        ] = {}

        # Sort by model_subset_ids.
        sorted_model_subsets = sorted(
            model_subsets.items()
        )  # List of (model_subset_id, batches)

        # Iterate through the whole encapsulation and reorder it.
        for model_subset_id, batches in sorted_model_subsets:
            # Sort by batch ids.
            sorted_batches = sorted(batches.items())  # List of (batch_id, layers)

            for batch_id, all_evaluation_variants in sorted_batches:
                # Iterate each batch and change the order of encapsulation in the dictionary.
                for (
                    visibility_variant,
                    visibility_values,
                ) in all_evaluation_variants.items():
                    if visibility_variant not in model_variant_reordered:
                        model_variant_reordered[visibility_variant] = {}
                    for (
                        evaluation_variant,
                        batch_summary_values,
                    ) in visibility_values.items():
                        if (
                            evaluation_variant
                            not in model_variant_reordered[visibility_variant]
                        ):
                            model_variant_reordered[visibility_variant][
                                evaluation_variant
                            ] = {}

                        for (
                            evaluation_entities_involved,
                            all_layer_metrics,
                        ) in batch_summary_values.items():
                            if (
                                evaluation_entities_involved
                                not in model_variant_reordered[visibility_variant][
                                    evaluation_variant
                                ]
                            ):
                                model_variant_reordered[visibility_variant][
                                    evaluation_variant
                                ][evaluation_entities_involved] = {}

                            for layer_name, tensor in all_layer_metrics.items():
                                array = tensor.detach().cpu().numpy()

                                # Init 2D structure: [model_subset][batch] = value
                                layer_store = model_variant_reordered[
                                    visibility_variant
                                ][evaluation_variant][
                                    evaluation_entities_involved
                                ].setdefault(
                                    layer_name,
                                    [
                                        [None for _ in range(NUM_EVALUATION_BATCHES)]
                                        for _ in range(NUM_EVALUATION_SUBSETS)
                                    ],
                                )

                                layer_store[model_subset_id][batch_id] = array

        return model_variant_reordered

    @staticmethod
    def _determine_padding_shape(
        array_list_to_pad: List[List[Optional[np.ndarray]]],
    ) -> Tuple[int, ...]:
        """
        Determines the maximal shape of the batches (the padding shape).

        :param array_list_to_pad: List of batch results to find the batch size from.
        :return: Returns shape determining final shape for padding (max batch shape).
        """
        # Flatten and find max shape across all arrays
        flat_arrays = [
            arr for sublist in array_list_to_pad for arr in sublist if arr is not None
        ]
        if not flat_arrays:
            raise ValueError("No arrays found to pad.")

        # Shape for padding.
        return np.array([a.shape for a in flat_arrays]).max(axis=0)

    @staticmethod
    def _preallocate_final_results_array(
        array_list_to_pad: List[List[Optional[np.ndarray]]],
        max_shape_for_padding: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Initializes the final np.array for the model variant covering the final
        shape for further processing.

        :param array_list_to_pad: List of batches that we want to convert to large np.array.
        :param max_shape_for_padding: Maximal shape of the batches.
        :return: Returns the initialized final np.array.
        Final shape is: `[num_model_subsets, num_batches, {max_shape_for_padding}]`
        """

        # Figure out the final array dimensions.
        num_model_subsets = len(array_list_to_pad)
        num_batches = max(len(sublist) for sublist in array_list_to_pad)

        # Allocate empty array filled with zeros
        final_shape = (num_model_subsets, num_batches) + tuple(max_shape_for_padding)
        return np.zeros(final_shape, dtype=np.float32)

    @staticmethod
    def _pad_and_merge_batches(
        array_list_to_pad: List[List[Optional[np.ndarray]]],
    ) -> np.ndarray:
        """
        Pads all batch arrays to the maximal size with 0 at the end (typically in the time dimension,
        where we are missing the prediction sequence sometimes). At the end merge the
        padded arrays to the one large numpy array with new model subset and batch id dimension.

        :param array_list_to_pad: List of list of batched arrays per each model subset.
        :return: Returns numpy array of padded batches of data to uniform shape.
        The result shape is:    `[model_subset, num_batches, {padded_batch_shape}]`
        """
        # Shape for padding.
        max_shape_for_padding = PredictionsAnalyzer._determine_padding_shape(
            array_list_to_pad
        )

        # Preallocated np.array for the final padded result storage.
        final_array = PredictionsAnalyzer._preallocate_final_results_array(
            array_list_to_pad, max_shape_for_padding
        )

        for subset_id, all_batches in enumerate(array_list_to_pad):
            for batch_id, arr in enumerate(all_batches):
                if arr is None:
                    continue  # leave as zeros

                # Pad with 0 at the end in case there are missing values.
                pad_width = [
                    (0, max_dim - cur_dim)
                    for cur_dim, max_dim in zip(arr.shape, max_shape_for_padding)
                ]
                padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
                final_array[subset_id, batch_id] = padded

        return final_array

    @staticmethod
    def _merge_model_variant_results(
        all_model_variant_subsets: Dict[
            int,
            Dict[
                int,
                Dict[
                    VisibilityVariants,
                    Dict[
                        BatchSummaryFields,
                        Dict[
                            PredictionAnalysisVariants,
                            Dict[str, torch.Tensor],
                        ],
                    ],
                ],
            ],
        ],
    ) -> Dict[
        VisibilityVariants,
        Dict[
            BatchSummaryFields,
            Dict[PredictionAnalysisVariants, Dict[str, np.ndarray]],
        ],
    ]:
        """
        Reformats one model results to one large np.array covering all batch results
        of the model.

        :param all_model_variant_subsets: All model results of different neuron subsets.
        :return: Returns analysis results in new shape with results
        in shape `[num_model_subsets, num_batches, {max_batch_shape}]`.
        """
        padded_model_variant: Dict[
            VisibilityVariants,
            Dict[
                BatchSummaryFields,
                Dict[PredictionAnalysisVariants, Dict[str, np.ndarray]],
            ],
        ] = {}

        # First reformat the results to have model subset and batch ids
        # together with the batch values. Then pad the all the batches

        # for visibility_variant, variant_value in all_model_variant_subsets.items():
        for (
            visibility_variant,
            visibility_variant_value,
        ) in PredictionsAnalyzer._reformat_original_results_to_new_order(
            all_model_variant_subsets
        ).items():
            padded_model_variant[visibility_variant] = {}
            for (
                evaluation_variant,
                evaluation_variant_value,
            ) in visibility_variant_value.items():
                # ) in PredictionsAnalyzer._reformat_original_results_to_new_order(
                #     all_model_variant_subsets
                # ).items():
                padded_model_variant[visibility_variant][evaluation_variant] = {}
                for (
                    evaluation_entities_involved,
                    entity_values,
                ) in evaluation_variant_value.items():
                    padded_model_variant[visibility_variant][evaluation_variant][
                        evaluation_entities_involved
                    ] = {}
                    for layer_name, np_arrays_list in entity_values.items():
                        # Convert all tensors to numpy arrays
                        padded_model_variant[visibility_variant][evaluation_variant][
                            evaluation_entities_involved
                        ][layer_name] = PredictionsAnalyzer._pad_and_merge_batches(
                            np_arrays_list
                        )

        return padded_model_variant

    def _merge_all_subsets_and_batches(
        self,
        original_variants: Dict[
            ModelEvaluationRunVariant,
            Dict[
                int,
                Dict[
                    int,
                    Dict[
                        VisibilityVariants,
                        Dict[
                            BatchSummaryFields,
                            Dict[
                                PredictionAnalysisVariants,
                                Dict[str, torch.Tensor],
                            ],
                        ],
                    ],
                ],
            ],
        ] = {},
    ) -> Dict[
        ModelEvaluationRunVariant,
        Dict[
            VisibilityVariants,
            Dict[
                BatchSummaryFields,
                Dict[PredictionAnalysisVariants, Dict[str, np.ndarray]],
            ],
        ],
    ]:
        """
        Reformats all results to format that the analysis results data are stored in the
        one large np.array in shape `[num_model_subsets, num_batches, {max_batch_shape}]`.

        :param original_variants: What model variants to evaluate, if `{}` then default.
        :return: Returns reshaped and merged results with np.array
        in shape: `[num_model_subsets, num_batches, {max_batch_shape}]`.
        """
        merged = {}

        if not original_variants:
            # Set default variants to process.
            original_variants = self.all_model_variant_results

        for model_variant, all_model_variant_subsets in original_variants.items():
            merged[model_variant] = PredictionsAnalyzer._merge_model_variant_results(
                all_model_variant_subsets
            )

        return merged

    @staticmethod
    def _convert_merged_to_pandas(
        merged_data: Dict[
            ModelEvaluationRunVariant,
            Dict[
                VisibilityVariants,
                Dict[
                    BatchSummaryFields,
                    Dict[PredictionAnalysisVariants, Dict[str, np.ndarray]],
                ],
            ],
        ],
    ) -> pd.DataFrame:
        """
        Converts merged preprocessed results to pandas dataframe.

        :param merged_data: Preprocessed results to convert to pandas.
        :return: Returns pandas.DataFrame representation of the results.
        """
        rows = []

        for model_variant, all_visibilities in merged_data.items():
            for visibility_variant, all_analyses in all_visibilities.items():
                for analysis_name, analysis_results in all_analyses.items():
                    for variant_type, all_layer_results in analysis_results.items():
                        for layer_name, array in all_layer_results.items():
                            value = array  # Store raw array, or modify below
                            rows.append(
                                {
                                    "model_variant": model_variant,
                                    "visibility_variant": visibility_variant,
                                    "analysis_name": analysis_name,
                                    "variant_type": variant_type,
                                    "layer_name": layer_name,
                                    "value": value,
                                }
                            )

        return pd.DataFrame(rows)
