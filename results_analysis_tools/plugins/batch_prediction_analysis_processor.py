"""
This code encapsulates the processing of the prediction batched analysis results.
"""

from typing import Dict, Union, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


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
from results_analysis_tools.plugins.wandb_summary_processor import WandbSummaryProcessor

from nn_model.type_variants import LayerType
import nn_model.globals


class BatchPredictionAnalysisProcessor:
    """
    Class encapsulating processing batched analysis of evaluation results.
    """

    num_trials = 20

    models_tbptt = [
        ModelEvaluationRunVariant.RNN_BACKPROPAGATION_5,
        ModelEvaluationRunVariant.RNN_BACKPROPAGATION_10,
        ModelEvaluationRunVariant.SYN_ADAPT_LGN_BACKPROPAGATION_5,
    ]

    def __init__(self, all_results: Dict[EvaluationProcessorChoices, pd.DataFrame]):
        self.all_batch_evaluation_results = (
            EvaluationResultsProcessor.get_evaluation_results(all_results)
        )
        self.wandb_processor = WandbSummaryProcessor(all_results)

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

        df = pd.concat([mean_rows, df], ignore_index=True)
        df["time"] += 1
        return df

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

    def prepare_pearson_cc_synchrony(
        self,
        df: pd.DataFrame = None,
        prediction_variants=("predictions", "train_like_predictions"),
        target_variant="targets",
    ) -> pd.DataFrame:
        """
        Computes Pearson CC for synchrony curves between specified predictions and target.

        :param df: Input DataFrame with columns: model_variant, layer_name, variant_type, subset_index, time, synchrony
        :param prediction_variants: Tuple of prediction variant types to compare (default: free + TF)
        :param target_variant: Label for the target synchrony curves
        :return: DataFrame with columns: model_variant, layer_name, subset_variant, variant_type, pearson_synchrony
        """
        df = self.prepare_for_plotting_synchrony_curves(df)

        all_results = []

        for pred_variant in prediction_variants:
            df_filtered = df[df["variant_type"].isin([pred_variant, target_variant])]

            sync_wide = (
                df_filtered.pivot_table(
                    index=["model_variant", "layer_name", "subset_index", "time"],
                    columns="variant_type",
                    values="synchrony",
                )
                .dropna()
                .reset_index()
            )

            for (model, layer, trial), group in sync_wide.groupby(
                ["model_variant", "layer_name", "subset_index"]
            ):
                pred = group[pred_variant].values
                target = group[target_variant].values
                pearson, _ = pearsonr(pred, target)

                all_results.append(
                    {
                        "model_variant": model,
                        "layer_name": layer,
                        "subset_variant": trial,
                        "variant_type": pred_variant,
                        "pearson_synchrony": pearson,
                    }
                )

        return pd.DataFrame(all_results)

    def combine_overall_and_synchrony_pearson_for_plotting(self) -> pd.DataFrame:
        """
        :return: Returns dataframe with both synchrony and overall pearson CC.
        """
        synchrony_pearson = self.prepare_pearson_cc_synchrony()
        overall_pearson = self.wandb_processor.prepare_raw_for_plotting()

        synchrony_pearson = synchrony_pearson[
            synchrony_pearson["variant_type"].isin(["predictions"])
        ]

        df_combined = pd.merge(
            synchrony_pearson[
                ["model_variant", "layer_name", "subset_variant", "pearson_synchrony"]
            ],
            overall_pearson.rename(columns={"CC_ABS": "pearson_overall"})[
                ["model_variant", "subset_variant", "pearson_overall"]
            ],
            on=["model_variant", "subset_variant"],
        )

        # Melt:
        df_combined = df_combined.melt(
            id_vars=["model_variant", "layer_name", "subset_variant"],
            value_vars=["pearson_overall", "pearson_synchrony"],
            var_name="Metric",
            value_name="Value",
        )

        # Compute mean across layers:
        return EvaluationResultsProcessor.ensure_model_type_order(
            (
                df_combined.groupby(["model_variant", "subset_variant", "Metric"])[
                    "Value"
                ]
                .mean()
                .reset_index()
            )
        )

    def prepare_for_free_forced_drift_plot(self) -> pd.DataFrame:
        """ "
        :return: Prepared drift free-forced data for plotting temporal behavior.
        """
        drift_df = EvaluationResultsProcessor.get_analysis_type(
            self.all_batch_evaluation_results,
            BatchSummaryFields.DRIFT_FREE_FORCED,
        )

        drift_df = drift_df[
            drift_df["model_variant"].isin(
                BatchPredictionAnalysisProcessor.models_tbptt
            )
        ]

        rows = []

        for _, row in drift_df.iterrows():
            model = row["model_variant"]
            layer = row["layer_name"]
            drift = row["value"]  # shape: [num_subsets, num_batches, batch_size, time]

            # Collapse all trials into one big pool
            mean_drift = drift.mean(axis=(0, 1, 2))  # → shape: [time]

            for t, d in enumerate(mean_drift):
                rows.append(
                    {"model_variant": model, "layer_name": layer, "time": t, "drift": d}
                )

        return EvaluationResultsProcessor.map_model_name_for_plotting(
            pd.DataFrame(rows)
        )
