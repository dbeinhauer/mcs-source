"""
This script serves as the interface that is used to execute experiments and evaluations.
"""

import os
import socket
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
    NeuronActivationTypes,
    RNNTypes,
    LossTypes,
)

from nn_model.logger import LoggerModel

hostname = socket.gethostname()

# Select the GPU to use in case we are working in CGG server.
if hostname in ["mayrau", "dyscalculia", "chicxulub.ms.mff.cuni.cz"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use the second GPU

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def init_wandb(
    arguments,
    project_name=f"V1_spatio_temporal_model_{nn_model.globals.SIZE_MULTIPLIER}",
):
    """
    Initializes Weights and Biases tracking.

    :param arguments: Command line arguments.
    :param project_name: Name of wandb project.
    """

    config = {
        "learning_rate": arguments.learning_rate,
        "epochs": arguments.num_epochs,
        "batch_size": arguments.train_batch_size,
        "model": arguments.model,
        "neuron_model_num_layers": arguments.neuron_num_layers,
        "neuron_model_layer_size": arguments.neuron_layer_size,
        "neuron_model_is_residual": arguments.neuron_residual,
        "neuron_activation_function": arguments.neuron_activation_function,
        "neuron_rnn_variant": arguments.neuron_rnn_variant,
        "model_size": nn_model.globals.SIZE_MULTIPLIER,
        "time_step_size": nn_model.globals.TIME_STEP,
        "num_hidden_time_steps": arguments.num_hidden_time_steps,
        "train_subset_size": arguments.train_subset,
        "subset_variant": arguments.subset_variant,
        "gradient_clip": arguments.gradient_clip,
        "optimizer_type": arguments.optimizer_type,
        "num_backpropagation_time_steps": arguments.num_backpropagation_time_steps,
        "weight_initialization": arguments.weight_initialization,
        "synaptic_adaptation": arguments.synaptic_adaptation,
        "synaptic_adaptation_size": arguments.synaptic_adaptation_size,
        "synaptic_adaptation_num_layers": arguments.synaptic_adaptation_num_layers,
        "synaptic_adaptation_only_lgn": arguments.synaptic_adaptation_only_lgn,
        "param_red": arguments.parameter_reduction,
        "loss": arguments.loss,
    }

    if arguments.debug:
        # Disable weights and biases tracking if there is only evaluation or debugging.
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_DISABLED"] = "false"

    # Load wandb API key.
    with open(f"{nn_model.globals.PROJECT_ROOT}/.wandb_api_key", "r") as f:
        api_key = f.read().strip()

    # Login to W&B using the key
    wandb.login(key=api_key)

    wandb.init(
        # project=f"V1_spatio_temporal_model_{nn_model.globals.SIZE_MULTIPLIER}",
        project=project_name,
        config=config,
    )


def init_model_path(arguments) -> str:
    """
    Initializes path where to store the best model parameters.

    By default the format of the filename is exhaustive list of all model parameters.

    :param arguments: command line arguments
    :return: Returns the path where the best model parameters should be stored.
    """
    if not arguments.model_filename:
        # Model filename not defined -> use format from docstring
        train_subset_string = ""
        if arguments.train_subset < 1.0:
            # Subset for training specified.
            train_subset_string = f"_train-sub-{arguments.train_subset}"

        subset_variant_string = (
            f"_sub-var-{arguments.subset_variant}"
            if arguments.subset_variant != -1
            else ""
        )

        only_lgn = "-lgn" if arguments.synaptic_adaptation_only_lgn else ""
        return "".join(
            [
                f"model-{int(nn_model.globals.SIZE_MULTIPLIER*100)}",
                train_subset_string,
                subset_variant_string,
                f"_step-{nn_model.globals.TIME_STEP}",
                f"_lr-{str(arguments.learning_rate)}",
                f"_{arguments.model}",
                f"_optim-steps-{arguments.num_backpropagation_time_steps}",
                "_neuron",
                f"-layers-{arguments.neuron_num_layers}",
                f"-size-{arguments.neuron_layer_size}",
                f"-activation-{arguments.neuron_activation_function}",
                f"-res-{arguments.neuron_residual}",
                f"_hid-time-{arguments.num_hidden_time_steps}",
                f"_grad-clip-{arguments.gradient_clip}",
                f"_optim-{arguments.optimizer_type}",
                f"_weight-init-{arguments.weight_initialization}",
                f"_p-red-{arguments.parameter_reduction}",
                f"_loss-{arguments.loss}",
                "_synaptic",
                f"-{arguments.synaptic_adaptation}",
                f"-size-{arguments.synaptic_adaptation_size}",
                f"-layers-{arguments.synaptic_adaptation_num_layers}",
                only_lgn,
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


def get_subset_variant_name(subset_path: str, subset_variant: int = -1) -> str:
    """
    Creates subset indices filename based on the selected variant.

    :param subset_path: Path containing model subset indices (without subset specified).
    :param subset_variant: Variant of the subset (if `-1` then let it be).
    :return: Returns model subset path with subset specified in format.
        `{subset_path_without_extension}_variant_{subset_variant}.{extension}"
    """

    if subset_variant != -1:
        # Subset variant specified -> add it for the variant to be loaded.
        splitted_path = subset_path.split(".")
        return splitted_path[0] + f"_variant_{subset_variant}." + splitted_path[-1]

    return subset_path


def main(arguments):
    """
    Perform model training and evaluation for the given setup specified
    in command line arguments.

    :param arguments: command line arguments.
    """
    if arguments.wandb_project_name:
        init_wandb(arguments, arguments.wandb_project_name)
    else:
        init_wandb(arguments)

    # Initialize model path (if not specified in the arguments).
    arguments.model_filename = init_model_path(arguments)

    arguments.subset_dir = get_subset_variant_name(
        arguments.subset_dir, arguments.subset_variant
    )

    logger = LoggerModel()
    logger.print_experiment_info(arguments)
    model_executer = ModelExecuter(arguments)

    # Log number of trainable parameters
    parameter_count = sum(
        p.numel() for p in model_executer.model.parameters() if p.requires_grad
    )
    wandb.config.update({"parameter_count": parameter_count})

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
        # Train the model using the given parameters.
        model_executer.train(
            continuous_evaluation_kwargs={
                "epoch_offset": 1,
                "evaluation_subset_size": execution_setup["epoch_evaluation_subset"],
            },
            debugging_stop_index=execution_setup["debug_stop_index"],
        )

        model_executer.evaluation(
            subset_for_evaluation=execution_setup["final_evaluation_subset"],
            save_predictions=arguments.save_all_predictions,
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


def init_parser() -> argparse.ArgumentParser:
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
        default="",
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
        help="Directory where the results of the evaluation should be saved in case of saving "
        "all evaluation predictions.",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="",
        help="Directory where the results of the evaluation should be saved in case of saving "
        "all evaluation predictions.",
    )
    parser.add_argument(
        "--neuron_model_responses_dir",
        type=str,
        default=nn_model.globals.DEFAULT_PATHS[
            PathDefaultFields.NEURON_MODEL_RESPONSES_DIR.value
        ],
        help="Directory where the results of neuron DNN model on testing range should be stored "
        "(filename is best model name).",
    )
    # Technical setup:
    parser.add_argument(
        "--num_data_workers",
        type=int,
        default=0,
        help="Number of CPU threads to use as workers for DataLoader. "
        "This can help if the GPU utilization is unstable (jumping between 0 and 100), "
        "because it's waiting for data.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=nn_model.globals.TRAIN_BATCH_SIZE,
        help="Batch size for training.",
    )
    # Training parameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00001,
        help="Learning rate to use in model training.",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default=OptimizerTypes.DEFAULT.value,
        choices=[optimizer_type.value for optimizer_type in OptimizerTypes],
        help="Optimizer type (either default or learning rate specific).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=LossTypes.POISSON.value,
        choices=[loss_type.value for loss_type in LossTypes],
        help="Loss to use during training.",
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
        help="Neuron model variant that we want to use.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs for training the model.",
    )
    parser.add_argument(
        "--num_hidden_time_steps",
        type=int,
        default=1,
        help="Number of hidden time steps in RNN of the whole model "
        "(in case it is set to 1 the the model would just predict the following visible "
        "time step (without additional hidden steps in between)).",
    )
    parser.add_argument(
        "--num_backpropagation_time_steps",
        type=int,
        default=1,
        help="Number of time steps for the backpropagation through time. It specifies"
        "how many time steps we want to perform till the next optimizer step.",
    )

    parser.add_argument(
        "--neuron_num_layers",
        type=int,
        default=5,
        help="Number of hidden layers we want to use in the model of the neuron.",
    )
    parser.add_argument(
        "--neuron_layer_size",
        type=int,
        default=10,
        help="Size of the layers of the neuron model.",
    )
    parser.set_defaults(neuron_residual=False)
    parser.add_argument(
        "--neuron_residual",
        action="store_true",
        help="Whether we want to use residual connections in the model of a neuron "
        "(and in the synaptic adaptation module).",
    )
    parser.add_argument(
        "--neuron_rnn_variant",
        type=str,
        default=RNNTypes.GRU.value,
        help="Variant of the RNN model we use in the neuron and synaptic adaption model.",
    )
    parser.add_argument(
        "--neuron_activation_function",
        type=str,
        default=NeuronActivationTypes.LEAKYTANH.value,
        choices=[activation_type.value for activation_type in NeuronActivationTypes],
        help="Final activation function of the neuron model.",
    )
    parser.set_defaults(synaptic_adaptation=False)
    parser.add_argument(
        "--synaptic_adaptation",
        action="store_true",
        help="Whether we want to use synaptic adaptation RNN module.",
    )
    parser.add_argument(
        "--synaptic_adaptation_size",
        type=int,
        default=10,
        help="Size of the layer in the synaptic adaptation RNN module.",
    )
    parser.add_argument(
        "--synaptic_adaptation_num_layers",
        type=int,
        default=3,
        help="Number of layers in the synaptic adaptation RNN module.",
    )
    parser.set_defaults(synaptic_adaptation_only_lgn=False)
    parser.add_argument(
        "--synaptic_adaptation_only_lgn",
        action="store_true",
        help="Whether we want to use synaptic adaptation RNN module only on LGN layer.",
    )
    # Parameter reduction
    parser.set_defaults(parameter_reduction=False)
    parser.add_argument(
        "--parameter_reduction",
        action="store_true",
        help="Model will run with reduced number of trainable parameters.",
    )
    # Dataset analysis:
    parser.add_argument(
        "--train_subset",
        type=float,
        default=1.0,
        help="Number of batches to select as train subset "
        "(for modeling training performance on different dataset size).",
    )
    parser.add_argument(
        "--subset_variant",
        type=int,
        default=-1,
        help="Variant of the subset if (-1) then it is used subset without `_variant` suffix.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="",
        help="Name of the Weights and Biases project (if custom).",
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
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    main(args)
