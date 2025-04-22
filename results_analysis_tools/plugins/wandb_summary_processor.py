"""
This encapsulates the processing of the weights and biases results.
"""

from typing import Dict

import pandas as pd

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

        # Step 1: Filter for subset_variant == 0
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

        # Step 4: Convert to LaTeX
        latex_table = table_df.to_latex(
            index=False,
            escape=True,
            caption="Setup of the models in evaluation:",
            label="tab:model_configs",
            column_format="lrrrrrrrr",
        )

        return latex_table
