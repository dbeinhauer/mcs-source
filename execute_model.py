"""
This script serves as the interface that is used to execute experiments and evaluations.
"""

import os
import argparse
from typing import Dict

import wandb

import nn_model.globals
from nn_model.model_executer import ModelExecuter
from nn_model.type_variants import (
    ModelTypes,
    PathDefaultFields,
    OptimizerTypes,
    WeightsInitializationTypes,
)
from nn_model.logger import LoggerModel

# from nn_model.evaluation_results_saver import EvaluationResultsSaver

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use the second GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def init_wandb(arguments):
    """
    Initializes Weights and Biases tracking.

    :param arguments: Command line arguments.
    """

    config = {
        "learning_rate": arguments.learning_rate,
        "epochs": arguments.num_epochs,
        "batch_size": nn_model.globals.TRAIN_BATCH_SIZE,
        "model": arguments.model,
        "neuron_model_num_layers": arguments.neuron_num_layers,
        "neuron_model_layer_size": arguments.neuron_layer_size,
        "neuron_model_is_residual": not arguments.neuron_not_residual,
        "model_size": nn_model.globals.SIZE_MULTIPLIER,
        "time_step_size": nn_model.globals.TIME_STEP,
        "num_hidden_time_steps": arguments.num_hidden_time_steps,
        "train_subset_size": arguments.train_subset,
        "gradient_clip": arguments.gradient_clip,
        "optimizer_type": arguments.optimizer_type,
        "weight_initialization": arguments.weight_initialization,
    }

    if arguments.best_model_evaluation or arguments.debug:
        # Disable weights and biases tracking if there is only evaluation or debugging.
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_DISABLED"] = "false"

    wandb.init(
        project=f"V1_spatio_temporal_model_{nn_model.globals.SIZE_MULTIPLIER}",
        config=config,
    )


def init_model_path(arguments) -> str:
    """
    Initializes path where to store the best model parameters.

    By default (if not specified other version in `arguments`) the path is in format:
        ```
        arguments.model_dir/ +
            model_ +
            [train-subset-{train_subset_size}] +
            lr-{learning_rate} +
            _{model_type} +
            _residual-{True/False} +
            _neuron_layers-{num_neuron_layers} +
            _neuron-size-{size_neuron_layer} +
            _num-hidden-time-steps-{num_hidden_time_steps} +
            .pth
        ```
        or in case `arguments.model_filename` is defined:
        `arguments.model_dir/arguments.model_filename`


    :param arguments: command line arguments
    :return: Returns the path where the best model parameters should be stored.
    """
    if not arguments.model_filename:
        # Model filename not defined -> use format from docstring
        train_subset_string = ""
        if arguments.train_subset < 1.0:
            # Subset for training specified.
            train_subset_string = f"train-subset-{arguments.train_subset}"
        return "".join(
            [
                f"model-{int(nn_model.globals.SIZE_MULTIPLIER*100)}",
                train_subset_string,
                f"_step-{nn_model.globals.TIME_STEP}",
                f"_lr-{str(arguments.learning_rate)}",
                f"_{arguments.model}",
                f"_residual-{not arguments.neuron_not_residual}",
                f"_neuron-layers-{arguments.neuron_num_layers}",
                f"_neuron-size-{arguments.neuron_layer_size}",
                f"_num-hidden-time-steps-{arguments.num_hidden_time_steps}",
                f"_gradient-clip-{arguments.gradient_clip}",
                f"_optimizer-type-{arguments.optimizer_type}",
                f"_weight-initialization-{arguments.weight_initialization}",
                ".pth",
            ]
        )

    return arguments.model_filename


def set_model_execution_parameters(
    epoch_evaluation_subset: int = 10,
    debug_stop_index: int = -1,
    final_evaluation_subset: int = -1,
    best_model_evaluation_subset: int = -1,
) -> Dict[str, int]:
    """
    Creates setup dictionary for model evaluation.

    :param epoch_evaluation_subset: How many batches do we want run evaluation
    after each train epoch.
    :param debug_stop_index: How many batches do we want to train (for debugging).
    If `-1` then all batches.
    :param final_evaluation_subset: How many batches do we want to run evaluation
    after training is finished on the best model.
    :param best_model_evaluation_subset: How many batches do we want to run evaluation
    when evaluating the best model.
    :return: Returns dictionary that serves as setup for model execution.
    """
    return {
        "epoch_evaluation_subset": epoch_evaluation_subset,
        "debug_stop_index": debug_stop_index,
        "final_evaluation_subset": final_evaluation_subset,
        "best_model_evaluation_subset": best_model_evaluation_subset,
    }


def main(arguments):
    """
    Perform model training and evaluation for the given setup specified
    in command line arguments.

    :param arguments: command line arguments.
    """
    init_wandb(arguments)

    # Initialize model path (if not specified in the arguments).
    arguments.model_filename = init_model_path(arguments)

    logger = LoggerModel()
    logger.print_experiment_info(arguments)
    model_executer = ModelExecuter(arguments)

    # Set parameters for the execution.
    execution_setup = set_model_execution_parameters()
    if arguments.debug:
        execution_setup = set_model_execution_parameters(
            epoch_evaluation_subset=1,
            debug_stop_index=1,
            final_evaluation_subset=1,
            best_model_evaluation_subset=1,
        )

    if not arguments.best_model_evaluation and not arguments.neuron_model_responses:
        # Train the model used the given parameters.
        model_executer.train(
            continuous_evaluation_kwargs={
                "epoch_offset": 1,
                "evaluation_subset_size": execution_setup["epoch_evaluation_subset"],
            },
            debugging_stop_index=execution_setup["debug_stop_index"],
        )

        model_executer.evaluation(
            subset_for_evaluation=execution_setup["final_evaluation_subset"]
        )
    else:
        if arguments.neuron_model_responses:
            # Save neuron DNN models outputs.
            model_executer.evaluation_results_saver.save_neuron_model_responses(
                model_executer.evaluate_neuron_models(),
                arguments.neuron_model_responses_dir,
            )
        if arguments.best_model_evaluation:
            # Run full evaluation on the best trained model.
            model_executer.evaluation(
                subset_for_evaluation=execution_setup["best_model_evaluation_subset"],
                save_predictions=arguments.save_all_predictions,
            )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute model training or evaluation."
    )
    # Paths and directories:
    parser.add_argument(
        "--train_dir",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[PathDefaultFields.TRAIN_DIR.value],
        help="Directory where train dataset is stored.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[PathDefaultFields.TEST_DIR.value],
        help="Directory where tests dataset is stored.",
    )
    parser.add_argument(
        "--subset_dir",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[PathDefaultFields.SUBSET_DIR.value],
        help="Directory where model subset indices are stored.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[PathDefaultFields.MODEL_DIR.value],
        help="Directory where to store the best model parameters.",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="",
        help="Filename where to store the best model.",
    )
    parser.add_argument(
        "--experiment_selection_path",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[
            PathDefaultFields.EXPERIMENT_SELECTION_PATH.value
        ],
        help="Path to selected experiments used for model analysis during evaluation.",
    )
    parser.add_argument(
        "--neuron_selection_path",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[
            PathDefaultFields.NEURON_SELECTION_PATH.value
        ],
        help="Path to selected neuron IDs used for model analysis during evaluation.",
    )
    parser.add_argument(
        "--selection_results_dir",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[
            PathDefaultFields.SELECTION_RESULTS_DIR.value
        ],
        help="Path to selected neuron IDs used for model analysis during evaluation.",
    )
    parser.add_argument(
        "--full_evaluation_dir",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[
            PathDefaultFields.FULL_EVALUATION_DIR.value
        ],
        help="Directory where the results of the evaluation should be saved in case of saving all evaluation predictions.",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="",
        help="Directory where the results of the evaluation should be saved in case of saving all evaluation predictions.",
    )
    parser.add_argument(
        "--neuron_model_responses_dir",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[
            PathDefaultFields.NEURON_MODEL_RESPONSES_DIR.value
        ],
        help="Directory where the results of neuron DNN model on testing range should be stored (filename is best model name).",
    )
    # Training parameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00001,
        help="Learning rate to use in model training.",
    )
    # Training parameters
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default=OptimizerTypes.DEFAULT.value,
        choices=[optimizer_type.value for optimizer_type in OptimizerTypes],
        help="Optimizer type (either default or learning rate specific).",
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=10000.0,
        help="Gradient clipping max norm.",
    )
    parser.add_argument(
        "--weight_initialization",
        type=str,
        default=WeightsInitializationTypes.DEFAULT.value,
        choices=[weights_type.value for weights_type in WeightsInitializationTypes],
        help="Which type of weights initialization we want to use.",
    )
    # Model parameters:
    parser.add_argument(
        "--model",
        type=str,
        default=ModelTypes.RNN_SEPARATE.value,
        choices=[model_type.value for model_type in ModelTypes],
        help="Model variant that we want to use.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs for training the model.",
    )
    parser.add_argument(
        "--neuron_num_layers",
        type=int,
        default=5,
        help="Number of hidden layers we want to use in feed-forward model of a neuron.",
    )
    parser.add_argument(
        "--neuron_layer_size",
        type=int,
        default=10,
        help="Size of the layers we want to use in feed-forward model of a neuron.",
    )
    parser.set_defaults(neuron_not_residual=False)
    parser.add_argument(
        "--neuron_not_residual",
        action="store_true",
        help="Whether we want to use residual connections in feed-forward model of a neuron.",
    )
    parser.add_argument(
        "--num_hidden_time_steps",
        type=int,
        default=1,
        help="Number of hidden time steps in RNN (to use backtracking through time (not just use known targets)).",
    )
    # Dataset analysis:
    parser.add_argument(
        "--train_subset",
        type=float,
        default=1.0,
        help="Number of batches to select as train subset (for model training performance).",
    )
    # Evaluation options:
    parser.set_defaults(best_model_evaluation=False)
    parser.add_argument(
        "--best_model_evaluation",
        action="store_true",
        help="Runs only evaluation on the best saved model for the given parameters.",
    )
    parser.set_defaults(save_all_predictions=False)
    parser.add_argument(
        "--save_all_predictions",
        action="store_true",
        help="Whether we want to store all model predictions in final evaluation.",
    )
    parser.set_defaults(neuron_model_responses=False)
    parser.add_argument(
        "--neuron_model_responses",
        action="store_true",
        help="Whether we want to get neuron DNN model responses for given range of input data.",
    )
    # Debugging:
    parser.set_defaults(debug=False)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Start debugging mode.",
    )

    args = parser.parse_args()
    main(args)
