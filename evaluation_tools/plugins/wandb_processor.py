"""
This script defines functionality for exporting the results from Weights and Biases.
"""

from typing import Dict, List

import pandas as pd
import wandb

from evaluation_tools.fields.experiment_parameters_fields import (
    WANDB_ENTITY,
    GridSearchRunVariants,
    ModelEvaluationRunVariant,
    AdditionalExperiments,
    WandbExperimentVariants,
)


class WandbProcessor:
    """
    This class serves for loading and processing of the results taken from
    weights and biases API.
    """

    # Properties from wandb that we want to keep.
    columns_to_keep = [
        "experiment_variant",
        "model_variant",
        "CC_ABS",
        "CC_NORM",
        "epochs",
        "learning_rate",
        "subset_variant",
        "neuron_model_layer_size",
        "neuron_model_num_layers",
        "neuron_model_is_residual",
        "synaptic_adaptation",
        "synaptic_adaptation_size",
        "synaptic_adaptation_num_layers",
        "num_backpropagation_time_steps",
    ]

    def __init__(
        self,
        all_variants: Dict[
            WandbExperimentVariants,
            List[
                GridSearchRunVariants
                | ModelEvaluationRunVariant
                | AdditionalExperiments
            ],
        ],
    ):
        self.all_variants = all_variants

        self.all_results: Dict[
            WandbExperimentVariants,
            Dict[
                GridSearchRunVariants
                | ModelEvaluationRunVariant
                | AdditionalExperiments,
                pd.DataFrame,
            ],
        ] = {}
        self.all_results_pandas: pd.DataFrame = None

    @property
    def get_all_results(self) -> pd.DataFrame:
        """
        :return: Returns all wandb results as pandas dataframe.
        """
        if self.all_results_pandas:
            # In case we have already converted the results to pandas
            # -> just return the results.
            return self.all_results_pandas

        # Results not converted to pandas-> convert them and return.
        return self.to_pandas()

    def to_pandas(
        self,
        wandb_results: Dict[
            WandbExperimentVariants,
            Dict[
                GridSearchRunVariants
                | ModelEvaluationRunVariant
                | AdditionalExperiments,
                pd.DataFrame,
            ],
        ] = {},
    ) -> pd.DataFrame:
        """
        Converts loaded data from wandb to pandas and filters only selected columns.

        :param wandb_results: wandb_results to process, if `{}` then process default.
        :return: Returns converted wandb results to pandas.
        """
        rows = []

        if not wandb_results:
            wandb_results = self.all_results

        for experiment_variant, model_variant in wandb_results.items():
            for model_variant, df in model_variant.items():
                df_copy = df.copy()
                df_copy["experiment_variant"] = experiment_variant
                df_copy["model_variant"] = model_variant

                # Flatten 'summary' and 'config' if they exist
                if "summary" in df_copy.columns:
                    summary_df = df_copy["summary"].apply(pd.Series)
                    df_copy = df_copy.drop(columns=["summary"]).join(summary_df)

                if "config" in df_copy.columns:
                    config_df = df_copy["config"].apply(pd.Series)
                    df_copy = df_copy.drop(columns=["config"]).join(config_df)

                rows.append(df_copy)

        if rows:
            # In case there is some summary -> create the proper DataFrame
            # and take only the selected columns.
            self.all_results_pandas = pd.concat(rows, ignore_index=True)[
                WandbProcessor.columns_to_keep
            ]
            return self.all_results_pandas
        else:
            return pd.DataFrame()

    @staticmethod
    def load_results(
        variant: (
            GridSearchRunVariants | ModelEvaluationRunVariant | AdditionalExperiments
        ),
    ) -> pd.DataFrame:
        """
        Loads dataframe from one project in wandb.

        :param variant: Variant to load.
        :return: Returns variant configuration and results as dataframe object.
        """
        api = wandb.Api()

        # Project is specified by <entity/project-name>
        runs = api.runs(f"{WANDB_ENTITY}/{variant.value}")

        summary_list, config_list, name_list = [], [], []
        for run in runs:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)

            # .config contains the hyperparameters.
            config_list.append({k: v for k, v in run.config.items()})

            # .name is the human-readable name of the run.
            name_list.append(run.name)

        runs_df = pd.DataFrame(
            {"summary": summary_list, "config": config_list, "name": name_list}
        )

        return runs_df

    def load_all_results(
        self,
        all_variants: Dict[
            WandbExperimentVariants,
            List[
                GridSearchRunVariants
                | ModelEvaluationRunVariant
                | AdditionalExperiments
            ],
        ] = {},
    ) -> Dict[
        WandbExperimentVariants,
        Dict[
            GridSearchRunVariants | ModelEvaluationRunVariant | AdditionalExperiments,
            pd.DataFrame,
        ],
    ]:
        """
        Loads all selected variants of wandb results with config.

        :param all_variants: Variants to load, if `{}` then process default.
        :return: Returns loaded variants as dataframes in dictionary structure.
        """

        if not all_variants:
            # Variants not provided -> use default.
            all_variants = self.all_variants

        for wandb_experiment_type, wand_experiments in all_variants.items():
            self.all_results[wandb_experiment_type] = {}
            for experiment_variant in wand_experiments:
                self.all_results[wandb_experiment_type][experiment_variant] = (
                    WandbProcessor.load_results(experiment_variant)
                )

        return self.all_results
