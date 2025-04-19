"""
This script serves as main part that loads and processes the analysis results.
"""

from typing import Dict, Any, List, Tuple, Iterable
import pickle
import re
import os
from enum import Enum

from itertools import product
import pandas as pd
import numpy as np

from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    HistogramFields,
    StatisticsFields,
    DatasetVariantField,
    DatasetDimensions,
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

import nn_model.globals

from results_analysis_tools.plugins.results_loader import ResultsLoader
from results_analysis_tools.plugins.histogram_processor import DatasetHistogramProcessor
from results_analysis_tools.fields.experiment_analyses import (
    PlottingVariants,
    PluginVariants,
)


class ResultAnalyzer:
    """
    This class serves for statistical analysis of all processed results.
    """

    def __init__(self, results_to_load: Dict[EvaluationProcessorChoices, str]):

        # Loads all analysis results.
        self.results_loader = ResultsLoader(results_to_load)
        self.all_results = self.results_loader.load_all_results()

        # Plugins
        self.all_plugins = {
            PluginVariants.DATASET_HISTOGRAM_PROCESSOR: DatasetHistogramProcessor(
                self.all_results
            )
        }

    @property
    def get_all_results(self) -> Dict[EvaluationProcessorChoices, pd.DataFrame]:
        """
        :return: Returns all analyses results.
        """
        return self.all_results

    def prepare_dataframe_for_plot(
        self, variant: PlottingVariants, is_test: bool = False
    ) -> pd.DataFrame:

        if variant == PlottingVariants.TIME_BIN_COUNT_RATIO:
            return self.all_plugins[
                PluginVariants.DATASET_HISTOGRAM_PROCESSOR
            ].prepare_for_plotting(is_test=is_test)

        # Plotting variant not implement yet.
        return None

    def get_mean_spike_counts(
        self, is_test: bool
    ) -> Dict[int, List[Tuple[int, float]]]:
        return self.all_plugins[
            PluginVariants.DATASET_HISTOGRAM_PROCESSOR
        ].compute_spike_count_distribution(is_test=is_test)

    @staticmethod
    def format_distribution_as_latex_table(
        variant_dict: Dict[int, List[Tuple[int, float]]], title: str
    ) -> str:
        # Step 1: Get all time_steps and all bins (assumes bins are the same for all)
        time_steps = sorted(variant_dict.keys())
        bins = sorted({b for counts in variant_dict.values() for b, _ in counts})

        # Step 2: Build a dictionary: bin → list of densities per time_step
        bin_to_densities = {b: [] for b in bins}
        for b in bins:
            for ts in time_steps:
                density_dict = dict(variant_dict[ts])
                bin_to_densities[b].append(
                    density_dict.get(b, 0.0)
                )  # default to 0 if missing

        # Step 3: Format as LaTeX
        header = " & ".join(
            ["\\textbf{Bin}"] + [f"\\textbf{{{ts} ms}}" for ts in time_steps]
        )
        lines = [
            f"\\textbf{{{title}}}",
            "\\begin{tabular}{r | " + " ".join(["r"] * len(time_steps)) + "}",
        ]
        lines.append(header + " \\\\ \\hline")

        for b in bins:
            row = (
                f"{b} & "
                + " & ".join([f"{d:.4f}" for d in bin_to_densities[b]])
                + " \\\\"
            )
            lines.append(row)

        lines.append("\\end{tabular}")
        return "\n".join(lines)


if __name__ == "__main__":
    EVALUATION_RESULTS_BASE = "/analysis_results"
    analysis_paths = {
        EvaluationProcessorChoices.FULL_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.FULL_DATASET_ANALYSIS.value}/",
        EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS.value}/",
        # EvaluationProcessorChoices.WANDB_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.WANDB_ANALYSIS.value}/",
        EvaluationProcessorChoices.PREDICTION_ANALYSIS: f"{nn_model.globals.PROJECT_ROOT}{EVALUATION_RESULTS_BASE}/{EvaluationProcessorChoices.PREDICTION_ANALYSIS.value}/",
    }
    result_analyzer = ResultAnalyzer(analysis_paths)

    train = result_analyzer.get_mean_spike_counts(False)
    test = result_analyzer.get_mean_spike_counts(True)

    print(
        ResultAnalyzer.format_distribution_as_latex_table(
            train, "Spike count distribution"
        )
    )
    print()
    print(
        ResultAnalyzer.format_distribution_as_latex_table(
            test, "Spike count distribution"
        )
    )

    # for variant, name in [(train, "Train: "), (test, "Test: ")]:
    #     print(name)
    #     for time_step, counts in variant.items():
    #         print(f"Time step: {int(time_step)}")
    #         for bin, count in counts:
    #             print(f"{int(bin)}: {float(count)}")
