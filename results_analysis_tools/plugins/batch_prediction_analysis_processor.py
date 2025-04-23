"""
This code encapsulates the processing of the prediction batched analysis results.
"""

from typing import Dict, Union, List

import numpy as np
import pandas as pd

from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from results_analysis_tools.plugins.evaluation_results_processor import (
    EvaluationResultsProcessor,
)
from evaluation_tools.fields.prediction_analysis_fields import (
    BatchSummaryFields,
    EvaluationPairsVariants,
)
from evaluation_tools.fields.experiment_parameters_fields import (
    WandbExperimentVariants,
    GridSearchRunVariants,
    ModelEvaluationRunVariant,
    AdditionalExperiments,
)

from nn_model.type_variants import LayerType
import nn_model.globals


class BatchPredictionAnalysisProcessor:

    num_trials = 20

    def __init__(self, all_results: Dict[EvaluationProcessorChoices, pd.DataFrame]):
        self.all_batch_evaluation_results = (
            EvaluationResultsProcessor.get_evaluation_results(all_results)
        )

    @staticmethod
    def _mean_synchrony(df: pd.DataFrame) -> pd.DataFrame:
        """
        Averages synchrony across all examples per layer in each time bin.

        :param df: Original dataframe to prepare synchrony.
        :return: Returns dataframe with columns
        ["model_variant", "layer_name", "variant_type", "subset_index", "time", "synchrony"]
        """
        rows = []
        for _, row in df.iterrows():
            model = row["model_variant"]
            layer = row["layer_name"]
            variant = row["variant_type"]
            sync = row[
                "value"
            ]  # shape: [num_model_subsets, num_batches, batch_size, time_steps]

            num_subsets, _, _, time_steps = sync.shape

            for subset_idx in range(num_subsets):
                # Summarize the synchrony across all experiments
                # (we need to divide by number of trials because the results are already sums across trials).
                summed_curve = (
                    sync[subset_idx].mean(axis=(0, 1))
                    / BatchPredictionAnalysisProcessor.num_trials
                )  # shape: [time_steps]
                for t in range(time_steps):
                    rows.append(
                        {
                            "model_variant": model,
                            "layer_name": layer,
                            "layer_size": nn_model.globals.MODEL_SIZES[layer],
                            "variant_type": variant,
                            "subset_index": subset_idx,
                            "time": t,
                            # Make the synchrony to be a ration of firing neurons from the layer.
                            "synchrony": summed_curve[t]
                            / nn_model.globals.MODEL_SIZES[layer],
                        }
                    )

        return pd.DataFrame(rows)

    @staticmethod
    def _select_specific_model_layer_variant(
        df: pd.DataFrame,
        model_variants: List[ModelEvaluationRunVariant] = [],
        layers_to_plot: List[str] = [],
        variants_to_plot: List[EvaluationPairsVariants] = [],
    ) -> pd.DataFrame:
        """
        Selects only provided model, layer or prediction variants from the dataframe.

        :param df: Dataframe for selection.
        :param model_variants: List of model variants to include, if `[]` then skip and keep all.
        :param layers_to_plot: List of layers to include, if `[]` then skip and keep all.
        :param variants_to_plot: List of evaluation variants to include, if `[]` then skip and keep all.
        :return: Returns subset of the original dataframe based on the provided selection parameters.
        """
        if model_variants:
            # Take only selected model variants.
            df = df[df["model_variant"].isin(model_variants)]
        if layers_to_plot:
            # Take only selected layers.
            df = df[df["layer_name"].isin(layers_to_plot)]
        if variants_to_plot:
            # Take only selected variant types.
            df = df[df["variant_type"].isin(variants_to_plot)]

        return df

    @staticmethod
    def _add_artificial_first_time_step(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds artificial first time step for nicer plotting.

        In our data we do not have the first time step because we predict from the second one.
        This function adds artificial first time step as the mean across all time steps to ensure
        nicer and quality plots.

        :param df: Dataframe to modify.
        :return: Returns dataframe with additional first artificial time step.
        """
        mean_rows = (
            df.groupby(["model_variant", "layer_name", "variant_type"])["synchrony"]
            .mean()
            .reset_index()
            .assign(time=-1)  # artificial time step
        )

        return pd.concat([mean_rows, df], ignore_index=True)

    def prepare_for_plotting_synchrony_curves(
        self,
        original_df: pd.DataFrame = None,
        model_variants: List[ModelEvaluationRunVariant] = [],
        layers_to_plot: List[str] = [],
        variants_to_plot: List[EvaluationPairsVariants] = [],
    ) -> pd.DataFrame:
        """
        Prepare data for plotting the synchrony curves.

        :param df: Dataframe for selection.
        :param model_variants: List of model variants to include, if `[]` then skip and keep all.
        :param layers_to_plot: List of layers to include, if `[]` then skip and keep all.
        :param variants_to_plot: List of evaluation variants to include, if `[]` then skip and keep all.
        :return: Returns subset of the original dataframe based on the provided selection parameters.
        """

        if original_df is None:
            # Dataframe not specified -> use default one.
            original_df = self.all_batch_evaluation_results

        synchrony_curves = EvaluationResultsProcessor.get_analysis_type(
            original_df, BatchSummaryFields.SYNCHRONY
        )

        # Select subset to be plotted.
        df_selection = (
            BatchPredictionAnalysisProcessor._select_specific_model_layer_variant(
                synchrony_curves,
                model_variants=model_variants,
                layers_to_plot=layers_to_plot,
                variants_to_plot=variants_to_plot,
            )
        )

        # Apply all necessary processing steps in order to prepare the synchrony curve for plotting.
        return EvaluationResultsProcessor.ensure_layer_type_order(
            EvaluationResultsProcessor.ensure_model_type_order(
                BatchPredictionAnalysisProcessor._add_artificial_first_time_step(
                    EvaluationResultsProcessor.map_eval_to_values(
                        EvaluationResultsProcessor.map_model_name_for_plotting(
                            BatchPredictionAnalysisProcessor._mean_synchrony(
                                df_selection
                            )
                        ),
                        "variant_type",
                    )
                )
            )
        )
