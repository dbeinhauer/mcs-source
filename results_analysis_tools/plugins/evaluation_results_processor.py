"""
This script defines logic under processing the evaluation results.
"""

from typing import Dict

import numpy as np
import pandas as pd

from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    DatasetVariantField,
)
from evaluation_tools.fields.experiment_parameters_fields import (
    WandbExperimentVariants,
    GridSearchRunVariants,
    ModelEvaluationRunVariant,
    AdditionalExperiments,
)

from evaluation_tools.fields.prediction_analysis_fields import (
    BatchSummaryFields,
    EvaluationPairsVariants,
    PredictionAnalysisVariants,
)


class EvaluationResultsProcessor:
    """
    This class encapsulates the logic for processing all of the evaluation results.
    """

    # All model names that we want to use in plotting.
    model_names_mapping_for_plotting = {
        ModelEvaluationRunVariant.SIMPLE_TANH: "simple (tanh)",
        ModelEvaluationRunVariant.SIMPLE_LEAKYTANH: "simple (leakytanh)",
        ModelEvaluationRunVariant.DNN_JOINT: "dnn joint",
        ModelEvaluationRunVariant.DNN_SEPARATE: "dnn separate",
        ModelEvaluationRunVariant.RNN_BACKPROPAGATION_5: "rnn (5 steps)",
        ModelEvaluationRunVariant.RNN_BACKPROPAGATION_10: "rnn (10 steps)",
        ModelEvaluationRunVariant.SYN_ADAPT_LGN_BACKPROPAGATION_5: "syn adapt lgn (5 steps)",
        ModelEvaluationRunVariant.SYN_ADAPT_LGN_BACKPROPAGATION_10: "syn adapt lgn (10 steps)",
    }

    # WANDB RESULTS LOADING:
    @staticmethod
    def get_wandb_results(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        :param all_results: All analyses results dictionary.
        :return: Returns all wandb results.
        """
        return all_results[EvaluationProcessorChoices.WANDB_ANALYSIS]

    @staticmethod
    def get_wandb_grid_search_results(
        all_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        :param all_results: All wandb results.
        :return: Returns all wandb grid search results.
        """
        return all_results[
            all_results["experiment_variant"] == WandbExperimentVariants.GRID_SEARCH
        ]

    @staticmethod
    def get_wandb_evaluation_results(
        all_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        :param all_results: All wandb results.
        :return: Returns all wandb evaluation results.
        """
        return all_results[
            all_results["experiment_variant"] == WandbExperimentVariants.EVALUATION
        ]

    @staticmethod
    def get_wandb_dataset_subset_results(
        all_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        :param all_results: All wandb results.
        :return: Returns all wandb results on different dataset subset (example subsets).
        """
        df = all_results[
            all_results["experiment_variant"] == WandbExperimentVariants.ADDITIONAL
        ]

        return df[df["model_variant"] == AdditionalExperiments.DATASET_SUBSET_SIZE]

    # EVALUATION RESULTS LOADING:
    @staticmethod
    def get_evaluation_results(
        all_results: Dict[EvaluationProcessorChoices, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        :param all_results: All analyses results dictionary.
        :return: Returns evaluation_results.
        """
        return all_results[EvaluationProcessorChoices.PREDICTION_ANALYSIS]

    @staticmethod
    def get_model_variant(
        df: pd.DataFrame, model_variant: ModelEvaluationRunVariant
    ) -> pd.DataFrame:
        """
        :param df: Dataframe to take the analysis.
        :param model_variant: Type of the model variant on the dataset.
        :return: Returns dataframe rows that match the selected model variant.
        """
        return df[df["model_variant"] == model_variant]

    @staticmethod
    def get_analysis_type(
        df: pd.DataFrame, analysis_name: BatchSummaryFields
    ) -> pd.DataFrame:
        """
        :param df: Dataframe to take the analysis.
        :param analysis_name: Type of the analysis on the dataset.
        :return: Returns dataframe rows that match the selected analysis.
        """
        return df[df["analysis_name"] == analysis_name]

    @staticmethod
    def get_data_variant(
        df: pd.DataFrame, data_variant: PredictionAnalysisVariants
    ) -> pd.DataFrame:
        """
        :param df: Dataframe to take the analysis.
        :param data_variant: Type of the data variant of the analysis (either target, prediction,
        teacher-forced predictions or appropriate pairs based on the analysis type). Note that one
        needs to be careful when using this function as the data variant is not always defined for the
        selected analysis.
        :return: Returns dataframe rows that match the selected data variant.
        """
        return df[df["data_variant"] == data_variant]

    @staticmethod
    def map_model_name_for_plotting(
        df: pd.DataFrame, variant_string: str = "model_variant"
    ) -> pd.DataFrame:
        """
        Maps model variants with their appropriate names used in the plots.

        :param df: Dataframe for plotting.
        :param variant_string: Name of the column specifying the model variant,
        defaults to "model_variant"
        :return: Returns dataframe with model variant column values mapped to their plot names.
        """
        df[variant_string] = df[variant_string].map(
            EvaluationResultsProcessor.model_names_mapping_for_plotting
        )

        return df
