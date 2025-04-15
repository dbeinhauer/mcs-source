"""
This script defines functionality for exporting the results from Weights and Biases.
"""

from typing import Dict, List

import pandas as pd
import wandb

from evaluation_tools.fields.experiment_parameters_fields import (
    WANDB_ENTITY,
    GridSearchRunVariants,
    EvaluationRunVariants,
    AdditionalExperiments,
    WandbExperimentVariants,
)


class WandbProcessor:
    """
    This class serves for loading and processing of the results taken from
    weights and biases API.
    """

    def __init__(
        self,
        all_variants: Dict[
            WandbExperimentVariants,
            List[GridSearchRunVariants | EvaluationRunVariants | AdditionalExperiments],
        ],
    ):
        self.all_variants = all_variants

        self.all_results = WandbProcessor.load_all_results(all_variants)

    @staticmethod
    def load_results(
        variant: GridSearchRunVariants | EvaluationRunVariants | AdditionalExperiments,
    ) -> pd.DataFrame:
        """
        Loads dataframe from one project in wandb.

        :param variant: Variant to load.
        :return: Returns variant configuration and results as dataframe object.
        """
        api = wandb.Api()

        # Project is specified by <entity/project-name>
        runs = api.runs(f"{WANDB_ENTITY}/{variant}")

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

    @staticmethod
    def load_all_results(
        all_variants: Dict[
            WandbExperimentVariants,
            List[GridSearchRunVariants | EvaluationRunVariants | AdditionalExperiments],
        ],
    ) -> Dict[
        WandbExperimentVariants,
        Dict[
            GridSearchRunVariants | EvaluationRunVariants | AdditionalExperiments,
            pd.DataFrame,
        ],
    ]:
        """
        Loads all selected variants of wandb results with config.

        :param all_variants: Variants to load.
        :return: Returns loaded variants as dataframes in dictionary structure.
        """
        all_results: Dict[
            WandbExperimentVariants,
            Dict[
                GridSearchRunVariants | EvaluationRunVariants | AdditionalExperiments,
                pd.DataFrame,
            ],
        ] = {}

        for wandb_experiment_type, wand_experiments in all_variants.items():
            for experiment_variant in wand_experiments:
                all_results[wandb_experiment_type][experiment_variant] = (
                    WandbProcessor.load_results(experiment_variant)
                )

        return all_results
