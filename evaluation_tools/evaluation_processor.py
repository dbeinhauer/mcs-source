"""
This source code defines class used for analysis and plotting the model evaluation results
and dataset analysis.
"""

import sys
import os
from typing import List, Dict, Tuple, Optional, Any
import random
import argparse
from enum import Enum

import pandas as pd
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nn_model.globals
from nn_model.type_variants import (
    LayerType,
    NeuronModulePredictionFields,
    PathPlotDefaults,
    PathDefaultFields,
    EvaluationMeanVariants,
)
from nn_model.dataset_loader import SparseSpikeDataset  # , different_times_collate_fn
from nn_model.model_executer import ModelExecuter
from nn_model.type_variants import EvaluationFields, PathDefaultFields
from nn_model.dictionary_handler import DictionaryHandler

from evaluation_tools.plugins.dataset_analyzer import DatasetAnalyzer
from evaluation_tools.plugins.wandb_processor import WandbProcessor
from evaluation_tools.plugins.predictions_analyzer import PredictionsAnalyzer
from evaluation_tools.fields.dataset_analyzer_fields import (
    AnalysisFields,
    HistogramFields,
    StatisticsFields,
    DatasetDimensions,
    DatasetVariantField,
)
from evaluation_tools.fields.experiment_parameters_fields import (
    WandbExperimentVariants,
    GridSearchRunVariants,
    ModelEvaluationRunVariant,
    AdditionalExperiments,
    AllWandbVariants,
)
from evaluation_tools.fields.evaluation_processor_fields import (
    EvaluationProcessorChoices,
)
from evaluation_tools.scripts.pickle_manipulation import (
    load_pickle_file,
    store_pickle_file,
)
from execute_model import get_subset_variant_name

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use the second GPU


def different_times_collate_fn(batch):
    """
    Function that deals with loading data in batch that differ in time
    duration (there are fragments of dataset that are slightly different
    in time duration (the start and end parts of the experiment run)).
    It pads the missing time with zeros (pads missing blank stage at the end).

    NOTE: Variant for evaluation.

    :param batch: batch of data to pad (tuple of input and output).
    :return: Returns tuple of padded batch of input and output data.
    """
    # Find the maximum size in the second (time) dimension.
    max_size = nn_model.globals.NORMAL_NUM_TIME_STEPS

    # Initialize a list to hold the result dictionaries.
    num_dicts = len(batch[0])
    result = [{} for _ in range(num_dicts)]

    # Loop over each index in the tuples
    for i in range(num_dicts):
        # Get all dictionaries at the current index across all tuples
        dicts_at_index = [tup[i] for tup in batch]

        # Get the keys (assuming all dictionaries have the same keys)
        keys = dicts_at_index[0].keys()

        # For each key, concatenate the tensors from all dictionaries at the current index
        for key in keys:

            # Collect all tensors associated with the current key
            tensors_to_concat = [
                torch.nn.functional.pad(
                    d[key],
                    (0, 0, 0, max_size - d[key].size(1)),
                    "constant",
                    0,
                )
                for d in dicts_at_index
            ]

            # Concatenate tensors along a new dimension (e.g., dimension 0)
            result[i][key] = torch.stack(tensors_to_concat, dim=0)

    # Convert the list of dictionaries into a tuple
    return tuple(result)


class EvaluationProcessor:
    """
    Class used for processing the dataset and evaluation results
    for further statistical analysis.
    """

    target_subdirectory_prefix = "V1_"

    mean_responses_fields = [
        EvaluationFields.PREDICTIONS.value,
        EvaluationFields.TARGETS.value,
    ]
    rnn_to_neuron_fields = [
        EvaluationFields.PREDICTIONS.value,
        EvaluationFields.RNN_PREDICTIONS.value,
    ]

    # All Weights and Biases projects that we are interested in processing.
    wandb_variants_to_process = {
        WandbExperimentVariants.GRID_SEARCH: [
            GridSearchRunVariants.SIMPLE_TANH,
            GridSearchRunVariants.SIMPLE_LEAKYTANH,
            GridSearchRunVariants.DNN,
            GridSearchRunVariants.RNN,
            GridSearchRunVariants.SYNAPTIC_ADAPTATION,
        ],
        WandbExperimentVariants.EVALUATION: [
            ModelEvaluationRunVariant.SIMPLE_TANH,
            ModelEvaluationRunVariant.SIMPLE_LEAKYTANH,
            ModelEvaluationRunVariant.DNN_JOINT,
            ModelEvaluationRunVariant.DNN_SEPARATE,
            ModelEvaluationRunVariant.RNN_BACKPROPAGATION_5,
            ModelEvaluationRunVariant.RNN_BACKPROPAGATION_10,
            ModelEvaluationRunVariant.SYN_ADAPT_BACKPROPAGATION_5,
            # EvaluationRunVariants.SYN_ADAPT_BACKPROPAGATION_10,
        ],
        WandbExperimentVariants.ADDITIONAL: [
            AdditionalExperiments.DATASET_SUBSET_SIZE,
            # AdditionalExperiments.MODEL_SIZES,
        ],
    }

    model_variants_for_evaluation = [
        ModelEvaluationRunVariant.DNN_JOINT,
        ModelEvaluationRunVariant.DNN_SEPARATE,
    ]

    additional_experiments = []

    def __init__(
        self,
        train_dataset_dir: str,
        test_dataset_dir: str,
        arguments,
        all_wandb_variants: Dict[
            WandbExperimentVariants,
            List[AllWandbVariants
            ],
        ] = {},
        model_variants_for_evaluation: List[ModelEvaluationRunVariant] = [],
        additional_experiments: List[AdditionalExperiments] = [],
        process_test: bool = False,
        responses_dir: str = "",
        dnn_responses_path: str = "",
        neurons_path: str = "",
        skip_dataset_load: bool = False,
        data_workers_kwargs={},
    ):
        """
        Initializes tool that is used for analysis of the model responses.

        :param dataset_dir: Directory containing original dataset we want to analyse.
        :param responses_dir: Directory containing averaged responses of the model and
        its targets per trial.
        """
        # TODO: make loading of the files optional

        # Set the input directories
        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.responses_dir = responses_dir

        # TODO: Probably will delete (not functional currently)
        self.dnn_responses_path = dnn_responses_path

        # Initialize batch size (for loader) as default batch size for test dataset.
        self.batch_size = nn_model.globals.TEST_BATCH_SIZE

        # Initialize data loaders and datasets.
        self.train_dataset, self.train_loader, self.test_dataset, self.test_loader = (
            self._init_data_loaders(arguments, skip_dataset_load, data_workers_kwargs)
        )

        # Selected neurons for the plotting:
        self.selected_neurons = None
        if neurons_path:
            self.selected_neurons = EvaluationProcessor.load_selected_neurons(
                neurons_path
            )

        # Selected image batches for plotting:
        self.images_batches = EvaluationProcessor.randomly_select_batches()
        # Selected image indices for each selected batch selected for plotting.
        self.batch_image_indices = EvaluationProcessor.randomly_select_img_index(
            range(0, nn_model.globals.TEST_BATCH_SIZE), len(self.images_batches)
        )

        # ------------------------------------NEW REFINED VERSION------------

        self.all_model_predictions_base_dir = arguments.evaluation_results_base_dir

        # Analyzer of the dataset.
        self.dataset_analyzer = DatasetAnalyzer(process_test)

        # Weight and Biases results processor.
        if not all_wandb_variants:
            # Variants not defined -> use the default ones.
            all_wandb_variants = EvaluationProcessor.wandb_variants_to_process
        self.wandb_processor = WandbProcessor(all_wandb_variants)

        # Processor of the model prediction results.
        if not model_variants_for_evaluation:
            # Evaluation variants not selected -> use the default.
            model_variants_for_evaluation = (
                EvaluationProcessor.model_variants_for_evaluation
            )
        if not additional_experiments:
            # Additional experiments not selected -> use the default.
            additional_experiments = EvaluationProcessor.additional_experiments
        self.predictions_analyzer = PredictionsAnalyzer(
            model_variants_for_evaluation, additional_experiments
        )

    def _init_dataloader(
        self, is_test=True, model_subset_path: str = "", data_workers_kwargs={}
    ) -> Tuple[SparseSpikeDataset, DataLoader]:
        """
        Initializes dataset and dataloader objects.

        :param is_test: Flag whether load test dataset (multitrial dataset).
        :param model_subset_path: Path to subset indices, if `""` then full dataset.
        :param data_workers_kwargs: Kwargs of the DataLoader class.
        :return: Returns tuple of initialized dataset object and DataLoader object.
        """
        input_layers, output_layers = DictionaryHandler.split_input_output_layers(
            nn_model.globals.ORIGINAL_SIZES
        )
        dataset_dir = self.test_dataset_dir
        if not is_test:
            dataset_dir = self.train_dataset_dir

        dataset = SparseSpikeDataset(
            dataset_dir,
            input_layers,
            output_layers,
            model_subset_path=model_subset_path,
            is_test=is_test,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Load the dataset always in the same order.
            **data_workers_kwargs,
        )

        return dataset, loader

    def _init_data_loaders(
        self, arguments, skip_dataset_load: bool, data_workers_kwargs: Dict
    ) -> Tuple[
        Optional[SparseSpikeDataset],
        Optional[DataLoader],
        Optional[SparseSpikeDataset],
        Optional[DataLoader],
    ]:
        """
        Initializes raw dataset and dataloaders if specified.

        :param arguments: Command line arguments.
        :param skip_dataset_load: Flag whether skip the objects initialization.
        :param data_workers_kwargs: Data workers kwargs.
        :return: Returns tuple of both train and test dataset and dataloader objects.
        """
        if skip_dataset_load:
            # Skip raw dataset usage -> return empty object.
            return None, None, None, None

        # Select correct subset (or no subset if specified)/
        model_subset_path = get_subset_variant_name(
            (
                # Do not use model subset -> full dataset analysis.
                ""
                if arguments.action == EvaluationProcessorChoices.FULL_DATASET_ANALYSIS
                # Use default subset path from the model.
                else nn_model.globals.DEFAULT_PATHS[PathDefaultFields.SUBSET_DIR.value]
            ),
            subset_variant=(
                # Make sure we do not specify subset ID if full dataset analysis.
                -1
                if EvaluationProcessorChoices.FULL_DATASET_ANALYSIS
                else arguments.dataset_subset_id
            ),
        )

        train_dataset, train_loader = self._init_dataloader(
            is_test=False,
            model_subset_path=model_subset_path,
            data_workers_kwargs=data_workers_kwargs,
        )
        test_dataset, test_loader = self._init_dataloader(
            model_subset_path=model_subset_path,
            data_workers_kwargs=data_workers_kwargs,
        )
        return train_dataset, train_loader, test_dataset, test_loader

    @staticmethod
    def load_selected_neurons(path: str) -> Dict:
        """
        Loads selected neurons for output layers from pickle file.

        :param path: Path of the pickle file where the neuron IDs are stored.
        :return: Returns dictionary of loaded neurons IDs for each layer.
        """
        selected_neurons = load_pickle_file(path)

        return {
            layer: data
            for layer, data in selected_neurons.items()
            if layer.startswith(EvaluationProcessor.target_subdirectory_prefix)
        }

    @staticmethod
    def randomly_select_batches(
        num_batches: int = 90, subset_size: int = 10
    ) -> List[int]:
        """
        Randomly selects batches subset.

        :param num_batches: Total number of batches.
        :param selection_size: Size of the selection subset.
        :return: Returns list of randomly selected indices of the batches.
        """
        return random.sample(np.arange(num_batches).tolist(), subset_size)

    @staticmethod
    def randomly_select_img_index(selection_range, selection_size) -> List[int]:
        """
        Randomly selects indices of the images (each index per each batch).

        :param selection_range: Range of possible indices.
        :param selection_size: How many indices we want to select.
        :return: Returns list of randomly selected image indices.
        """
        return random.choices(selection_range, k=selection_size)

    def run_dataset_processing(
        self, process_test: bool, subset: int = -1
    ) -> Dict[AnalysisFields, Any]:
        """
        Runs dataset processing analysis. Either for the full dataset or
        for the subsets based on the initialized data loader.

        :param process_test: Whether to process test dataset, else process train dataset.
        :param subset: What part to process.
        """
        # Select correct dataset loader.
        loader = self.test_loader if process_test else self.train_loader

        # Run full dataset analysis and get the results.
        self.dataset_analyzer.full_analysis_run(loader, process_test, subset=subset)
        return self.dataset_analyzer.get_all_processing_results

    def run_wandb_results_processing(
        self,
        all_wandb_variants: Dict[
            WandbExperimentVariants,
            List[AllWandbVariants
            ],
        ] = {},
    ) -> pd.DataFrame:
        """
        Loads all selected projects from Weights and Biases and filters only
        relevant fields. At the end converts the data to pandas DataFrame.

        :param all_wandb_variants: Optionally process different variants, if `{}` then default.
        :return: Returns Weights and Biases results as pandas DataFrame.
        """
        self.wandb_processor.load_all_results(all_wandb_variants)
        return self.wandb_processor.to_pandas()

    def run_prediction_processing(
        self,
        base_dir: str = "",
        evaluation_variants: List[ModelEvaluationRunVariant] = [],
    ) -> pd.DataFrame:
        """
        Processes predictions of all model variants for each neuron subset variants
        and for all batches and converts the processing results to pandas DataFrame.

        :param base_dir: Base directory where all model variants predictions are stored,
        if `""` then use default base.
        :param evaluation_variants: What evaluation variants we are interested, if `[]` then default.
        :return: Returns processed all models predictions as pandas dataframe for each batch of data.
        """

        if not base_dir:
            # If not defined otherwise set base directory with the
            # evaluation results as default.
            base_dir = self.all_model_predictions_base_dir

        self.predictions_analyzer.process_all_model_variants_predictions(
            base_dir, evaluation_variants=evaluation_variants
        )

        return self.predictions_analyzer.to_pandas()


def main(arguments):

    # Set time step selected for analysis.
    nn_model.globals.reinitialize_time_step(arguments.time_step)

    train_dir = nn_model.globals.DEFAULT_PATHS[PathDefaultFields.TRAIN_DIR.value]
    test_dir = nn_model.globals.DEFAULT_PATHS[PathDefaultFields.TEST_DIR.value]

    model_name = "model-10_sub-var-9_step-20_lr-7.5e-06_simple_optim-steps-1_neuron-layers-5-size-10-activation-leakytanh-res-False_hid-time-1_grad-clip-10000.0_optim-default_weight-init-default_synaptic-False-size-10-layers-1"

    responses_dir = f"/home/david/source/diplomka/thesis_results/simple/full_evaluation_results/{model_name}/"

    dnn_responses_dir = f"/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/neuron_model_responses/{model_name}.pth"
    neurons_path = f"/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/neurons/model_size_{int(nn_model.globals.SIZE_MULTIPLIER*100)}_subset_10.pkl"

    num_data_workers = args.num_data_workers
    workers_enabled = num_data_workers > 0
    data_workers_kwargs = {
        "collate_fn": different_times_collate_fn,
        "num_workers": num_data_workers,  # number of workers which will supply data to GPU
        "pin_memory": workers_enabled,  # speed up data transfer to GPU
        "prefetch_factor": (
            num_data_workers // 2 if workers_enabled else None
        ),  # try to always have 4 samples ready for the GPU
        "persistent_workers": workers_enabled,  # keep the worker threads alive
    }
    # Whether we want to process the test dataset.
    process_test = arguments.dataset_variant == DatasetVariantField.TEST.value
    # Generate dataset analysis.
    evaluation_processor = EvaluationProcessor(
        train_dir,
        test_dir,
        arguments,
        process_test=process_test,
        data_workers_kwargs=data_workers_kwargs,
    )

    results = None
    if arguments.action in [
        EvaluationProcessorChoices.FULL_DATASET_ANALYSIS.value,
        EvaluationProcessorChoices.SUBSET_DATASET_ANALYSIS.value,
    ]:
        # Run dataset analysis.
        results = evaluation_processor.run_dataset_processing(
            process_test,
            subset=arguments.processing_subset,
        )
    elif arguments.action == EvaluationProcessorChoices.WANDB_ANALYSIS.value:
        # Extract Weights and Biases results.
        results = evaluation_processor.run_wandb_results_processing()
    elif arguments.action == EvaluationProcessorChoices.PREDICTION_ANALYSIS.value:
        # Process model predictions.
        evaluation_variants = []
        if not arguments.model_evaluation_variant:
            # Model variant not selected -> analyze all of them.
            evaluation_variants = list(ModelEvaluationRunVariant)
        else:
            evaluation_variants = [
                ModelEvaluationRunVariant(arguments.model_evaluation_variant)
            ]

        results = evaluation_processor.run_prediction_processing(
            arguments.evaluation_results_base_dir, evaluation_variants
        )

    if arguments.results_save_path:
        # Optionally save the results to pickle.
        store_pickle_file(arguments.results_save_path, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute model training or evaluation."
    )
    # General settings.
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=[i.value for i in EvaluationProcessorChoices._member_map_.values()],
        help="What type of action do.",
    )
    parser.add_argument(
        "--results_save_path",
        type=str,
        default="",
        help="Where to store the results (if `" "` then do not save).",
    )
    # Dataset processing
    parser.add_argument(
        "--dataset_variant",
        type=str,
        default=DatasetVariantField.TRAIN.value,
        choices=[i.value for i in DatasetVariantField._member_map_.values()],
        help="What dataset to evaluate.",
    )
    parser.add_argument(
        "--dataset_subset_id",
        type=int,
        default=-1,
        help="Variant ID of the subset of the size defined in `nn_model.globals.SIZE_MULTIPLIER, "
        "if `-1` then do not take any subset variant (default subset or full subset).",
    )
    parser.add_argument(
        "--num_data_workers",
        type=int,
        default=2,
        help="Number of CPUs (cores) for data loading.",
    )
    parser.add_argument(
        "--processing_subset",
        type=int,
        default=-1,
        help="Whether to process only subset of data (if `-1` then process all data).",
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=20,
        help="Size of the time bin to analyze.",
    )
    # Evaluation results processing.
    parser.add_argument(
        "--evaluation_results_base_dir",
        type=str,
        default=f"{nn_model.globals.PROJECT_ROOT}/thesis_results/evaluation/",
        help="Base directory where all model variants evaluation results are stored.",
    )
    parser.add_argument(
        "--model_evaluation_variant",
        type=str,
        default="",
        choices=[i.value for i in ModelEvaluationRunVariant._member_map_.values()],
        help="What model variant to process.",
    )

    args = parser.parse_args()
    main(args)
