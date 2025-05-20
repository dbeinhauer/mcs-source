"""
This scripts serves for loading the analysis results.
"""

import os
from typing import Dict, List, Tuple, Iterable

from itertools import product
import pandas as pd

from evaluation_tools.fields.dataset_analyzer_fields import (
    DatasetVariantField,
)

from evaluation_tools.fields.dataset_parameters import (
    ALL_TIME_STEP_VARIANTS,
    ALL_SUBSET_IDS,
)
from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.fields.experiment_parameters_fields import (
    ModelEvaluationRunVariant,
)
from evaluation_tools.scripts.pickle_manipulation import load_pickle_file


class ResultsLoader:
    """
    Class that serves for loading the analyses results.
    """

    def __init__(self, results_to_load: Dict[EvaluationProcessorChoices, str]):
        """
        :param results_to_load: All results that we are interested in loading.
        """

        # What processed evaluation results should we load.
        self.results_to_load = results_to_load

        self.all_results: Dict[EvaluationProcessorChoices, pd.DataFrame] = {}

    @property
    def get_all_results(
        self,
    ) -> Dict[EvaluationProcessorChoices, pd.DataFrame]:
        """
        :return: Returns all loaded results.
        """
        return self.all_results

    def load_all_results(self) -> Dict[EvaluationProcessorChoices, pd.DataFrame]:
        """
        Loads all analysis results of all variants.

        :return: Returns dictionary of each analysis variant.
        """
        self.load_full_dataset()
        self.load_subset_dataset()
        self.load_wandb_results()
        self.load_evaluation_analyses()

        return self.all_results

    def load_general_results(
        self,
        base_path: str,
        results_variant: EvaluationProcessorChoices,
        variants_to_iterate: List[Tuple[str, Iterable]],
    ) -> pd.DataFrame:
        """
        Base function encapsulating common logic for loading the preprocessed results for analysis.

        :param base_path: Base path containing the preprocessed results, if `""` then load default.
        :param results_variant: Variant of the results to load.
        :param variants_to_iterate: What variants corresponds to each encapsulated dictionary
        structure of the input values.
        :return: Returns preprocessed analysis results as pandas dataframe.
        """
        if not base_path:
            # Path not specified -> use default one.
            base_path = self.results_to_load[results_variant]

        all_dfs = []

        # Extract column names and value lists
        column_names = [name for name, _ in variants_to_iterate]
        variant_values = [list(values) for _, values in variants_to_iterate]

        # Iterate over the cartesian product of all variant values
        for variant_combo in product(*variant_values):
            # Create a dict: {column_name: value}
            combo_dict = dict(zip(column_names, variant_combo))

            # Generate filename from the combination
            # Order of columns matters for consistent filename formatting
            filename_parts = [results_variant.value] + [
                (
                    str(val.value) if hasattr(val, "value") else str(val)
                )  # Check for enum values
                for val in (combo_dict[col] for col in column_names)
            ]
            filename = "-".join(filename_parts) + ".pkl"
            path_to_data = base_path + filename

            if os.path.exists(path_to_data):
                # Load and annotate
                dataset = load_pickle_file(path_to_data)
                for col, val in combo_dict.items():
                    dataset[col] = val

                all_dfs.append(dataset)
            else:
                print(f"File {filename} does not exist. Skipping!")

        # Combine into a single DataFrame
        full_df = pd.concat(all_dfs, ignore_index=True)
        self.all_results[results_variant] = full_df

        return full_df

    def load_full_dataset(self, base_path: str = "") -> pd.DataFrame:
        """
        Loads all full dataset analyses and concatenates them to one large pd.DataFrame.

        :param base_path: Path to directory containing all results to load.
        If `""` then load from the default path.
        :return: Returns dataframe containing all results.
        """
        return self.load_general_results(
            base_path,
            EvaluationProcessorChoices.FULL_DATASET_ANALYSIS,
            [
                ("dataset_variant", DatasetVariantField._member_map_.values()),
                ("time_step", ALL_TIME_STEP_VARIANTS),
            ],
        )

    def load_subset_dataset(self, base_path: str = "") -> pd.DataFrame:
        """
        Loads all subset dataset analyses and concatenates them to one large pd.DataFrame.

        :param base_path: Path to directory containing all results to load.
        If `""` then load from the default path.
        :return: Returns dataframe containing all results.
        """
        return self.load_general_results(
            base_path,
            EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS,
            [
                ("dataset_variant", DatasetVariantField._member_map_.values()),
                ("subset_id", ALL_SUBSET_IDS),
            ],
        )

    def load_wandb_results(self, path: str = "") -> pd.DataFrame:
        """
        Loads all weights and biases results from extracted pickled dataframe.

        :param path: Path to pickled wandb data. If `""` then load from the default path.
        :return: Returns loaded wandb results in pandas dataframe.
        """
        if not path:
            # Load default path to wandb results.
            path = self.results_to_load[EvaluationProcessorChoices.WANDB_ANALYSIS]

        wandb_results = load_pickle_file(path)
        self.all_results[EvaluationProcessorChoices.WANDB_ANALYSIS] = wandb_results

        return wandb_results

    def load_evaluation_analyses(self, base_path: str = "") -> pd.DataFrame:
        """
        Loads all analyses of the evaluation results of all model variants.

        :param base_path: Path to directory containing all results to load.
        If `""` then load from the default path.
        :return: Returns dataframe containing all results.
        """
        return self.load_general_results(
            base_path,
            EvaluationProcessorChoices.PREDICTION_ANALYSIS,
            [
                (
                    "model_variant",
                    list(ModelEvaluationRunVariant),
                    # [
                    #     ModelEvaluationRunVariant.SIMPLE_LEAKYTANH,
                    #     ModelEvaluationRunVariant.DNN_JOINT,
                    #     ModelEvaluationRunVariant.DNN_SEPARATE,
                    #     ModelEvaluationRunVariant.RNN_BACKPROPAGATION_5,
                    #     ModelEvaluationRunVariant.RNN_BACKPROPAGATION_10,
                    #     ModelEvaluationRunVariant.SYN_ADAPT_LGN_BACKPROPAGATION_5,
                    #     ModelEvaluationRunVariant.DNN_JOINT_POISSON,
                    #     ModelEvaluationRunVariant.
                    # ],
                )
            ],
        )
