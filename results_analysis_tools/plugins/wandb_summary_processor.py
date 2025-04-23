"""
This encapsulates the processing of the weights and biases results.
"""

from typing import Dict, Union

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from itertools import product

from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from results_analysis_tools.plugins.evaluation_results_processor import (
    EvaluationResultsProcessor,
)


class WandbSummaryProcessor:
    """
    This class encapsulates all processing on results from weights and biases.
    """

    # Mapping of the metric names to be used in the plot.
    metric_name_mapping_for_plot = {"CC_ABS": "Pearson CC", "CC_NORM": "Normalized CC"}
    # Mapping of metric name for tables (compressed versions).
    metric_name_mapping_for_table = {"Pearson CC": "P-CC", "Normalized CC": "N-CC"}

    # Mapping of evaluation hyperparameters for the table of overview.
    evaluation_hyperparameters_name_mapping = {
        "model_variant": "Model Variant",
        "epochs": "Epochs",
        "learning_rate": "lr",
        "neuron_model_layer_size": "n-ls",
        "neuron_model_num_layers": "n-nl",
        "neuron_model_is_residual": "n-res",
        "synaptic_adaptation_size": "s-ls",
        "synaptic_adaptation_num_layers": "s-nl",
        "num_backpropagation_time_steps": "n-tbptt",
    }

    def __init__(self, all_results: Dict[EvaluationProcessorChoices, pd.DataFrame]):
        self.all_wandb_results = EvaluationResultsProcessor.get_wandb_results(
            all_results
        )

    @property
    def get_results(self) -> pd.DataFrame:
        return self.all_wandb_results

    def prepare_model_comparison_summary_for_plotting(
        self, original_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Prepares evaluation results across all the models for plotting their comparison in box plots.

        :param original_df: All wandb results, if `None` then default.
        :return: Returns the preprocessed evaluation results across different models from wandb
        prepared to be plotted.
        """

        if original_df is None:
            # If dataframe not provided -> take the evaluation results from the default one.
            original_df = EvaluationResultsProcessor.get_wandb_evaluation_results(
                self.all_wandb_results
            )

        # Map model types with their strings for plotting.
        original_df = EvaluationResultsProcessor.map_model_name_for_plotting(
            original_df
        )

        # Combine both CC_ABS and CC_NORM for the plot.
        df_melted = pd.melt(
            original_df,
            id_vars="model_variant",
            value_vars=["CC_ABS", "CC_NORM"],
            var_name="Correlation Type",
            value_name="Correlation Value",
        )
        # Rename metric names for plotting.
        df_melted["Correlation Type"] = df_melted["Correlation Type"].map(
            WandbSummaryProcessor.metric_name_mapping_for_plot
        )

        df_melted = EvaluationResultsProcessor.ensure_model_type_order(df_melted)

        return df_melted

    def prepare_latex_evaluation_setup_table(
        self, original_df: pd.DataFrame = None
    ) -> str:
        """
        Creates LaTeX table from the evaluation setup.

        :param original_df: Dataframe of Wandb evaluation results.
        :return: Returns string LaTeX table representation of the table.
        """
        if original_df is None:
            # If dataframe not provided -> take the evaluation results from the default one.
            original_df = EvaluationResultsProcessor.get_wandb_evaluation_results(
                self.all_wandb_results
            )

        # Map model types with their strings for plotting.
        original_df = EvaluationResultsProcessor.map_model_name_for_plotting(
            original_df
        )

        # Filter for subset_variant == 0
        filtered_df = original_df[original_df["subset_variant"] == 0]

        table_df = filtered_df[
            WandbSummaryProcessor.evaluation_hyperparameters_name_mapping.keys()
        ].drop_duplicates()

        # Convert enums to strings
        table_df["model_variant"] = table_df["model_variant"].apply(
            lambda x: x.name if hasattr(x, "name") else str(x)
        )

        table_df = table_df.rename(
            columns=WandbSummaryProcessor.evaluation_hyperparameters_name_mapping
        )

        # Convert to LaTeX
        latex_table = table_df.to_latex(
            index=False,
            escape=True,
            caption="Setup of the models in evaluation:",
            label="tab:model_configs",
            column_format="lrrrrrrrr",
        )

        return latex_table

    def prepare_evaluation_results_summary(
        self, return_latex: bool = False
    ) -> Union[pd.DataFrame, str]:
        """
        :param return_latex: Flag whether return LaTeX table summary.
        :return: Returns summary of model evaluation results.
        """

        df = self.prepare_model_comparison_summary_for_plotting()
        df["Correlation Type"] = df["Correlation Type"].map(
            WandbSummaryProcessor.metric_name_mapping_for_table
        )

        summary_stats = (
            df.groupby(["model_variant", "Correlation Type"])["Correlation Value"]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Pivot to wide format
        summary_table = summary_stats.pivot(
            index="model_variant", columns="Correlation Type", values=["mean", "std"]
        )

        # Flatten MultiIndex columns
        summary_table.columns = [
            f"{metric} ({stat})" for stat, metric in summary_table.columns
        ]
        summary_table = summary_table.reset_index()

        # Round values
        summary_table = summary_table.round(4)

        # Sort (by normalized CC mean)
        summary_table = summary_table.sort_values(by="N-CC (mean)", ascending=False)

        if return_latex:
            # Export to LaTeX
            return summary_table.to_latex(
                index=False,
                escape=True,
                float_format="%.4f",
                caption="Summary of Model Performance Metrics:",
                label="tab:model_summary_extended",
                column_format="lrrrrrrrr",
            )

        return summary_table

    def mann_whitney_paired_evaluation_models_test_cc_norm(
        self, original_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Runs pair-wise one-sided paired Mann-Whitney U-test between the normalized
        cross correlation values of the model variants.
        The test assumes hypothesis:
            H_0: x <= y
            H_1: x > y

        :param original_df: Dataframe of model results.
        :return: Returns paired p-value matrix of each ordered pairs of model variants.
        """

        original_df = self.prepare_model_comparison_summary_for_plotting()

        # Filter for Normalized CC
        ncc_df = original_df[original_df["Correlation Type"] == "Normalized CC"]

        # Prepare model list
        models = sorted(ncc_df["model_variant"].unique())

        # Initialize p-value matrix
        pval_matrix = pd.DataFrame(index=models, columns=models, dtype=float)

        # Pairwise one-sided Mann-Whitney U tests
        for model_i, model_j in product(models, repeat=2):
            if model_i == model_j:
                pval_matrix.loc[model_i, model_j] = np.nan
                continue

            x = ncc_df[ncc_df["model_variant"] == model_i]["Correlation Value"]
            y = ncc_df[ncc_df["model_variant"] == model_j]["Correlation Value"]

            # One-sided test: H0: x <= y, H1: x > y
            stat, p = mannwhitneyu(x, y, alternative="greater")
            pval_matrix.loc[model_i, model_j] = p

        return pval_matrix
