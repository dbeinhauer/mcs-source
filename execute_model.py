"""
This script serves as the interface that is used to execute experiments and evaluations.
"""

import os
import argparse

import wandb

import nn_model.globals
from nn_model.model_executer import ModelExecuter
from nn_model.type_variants import ModelTypes


def init_wandb(arguments):
    """
    Initializes Weights and Biases tracking.

    :param arguments: Command line arguments.
    """

    config = {
        "learning_rate": arguments.learning_rate,
        "epochs": arguments.num_epochs,
        "batch_size": nn_model.globals.train_batch_size,
        "model": arguments.model,
        "neuron_model_num_layers": arguments.neuron_num_layers,
        "neuron_model_layer_size": arguments.neuron_layer_size,
        "neuron_model_is_residual": not arguments.neuron_not_residual,
        "model_size": nn_model.globals.SIZE_MULTIPLIER,
        "time_step_size": nn_model.globals.TIME_STEP,
        "num_hidden_time_steps": arguments.num_hidden_time_steps,
    }

    if arguments.best_model_evaluation:
        # Disable weights and biases tracking if there is only evaluation.
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_DISABLED"] = "false"

    if arguments.debug:
        os.environ["WANDB_DISABLED"] = "true"

    wandb.init(
        project=f"V1_spatio_temporal_model_{nn_model.globals.SIZE_MULTIPLIER}",
        config=config,
    )
    # wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}


def main(arguments):
    """
    Perform model training and evaluation for the given setup specified
    in command line arguments.

    :param arguments: command line arguments.
    """

    # start a new experiment
    #  capture a dictionary of hyperparameters with config
    # if not args.best_model_evaluation:
    init_wandb(arguments)

    model_executer = ModelExecuter(arguments)

    evaluation_subset = 10
    debugging_stop_index = -1
    final_evaluation_subset = -1
    best_model_evaluation_subset = -1

    if arguments.debug:
        evaluation_subset = 1
        debugging_stop_index = 1
        final_evaluation_subset = 1
        best_model_evaluation_subset = 1

    if not arguments.best_model_evaluation:
        # Train the model used the given parameters.
        model_executer.train(
            continuous_evaluation_kwargs={
                "epoch_offset": 1,
                "evaluation_subset_size": evaluation_subset,
            },
            debugging_stop_index=debugging_stop_index,
        )

        model_executer.evaluation(subset_for_evaluation=final_evaluation_subset)
    else:
        # if arguments.save_all_predictions:
        model_executer.evaluation(
            subset_for_evaluation=-best_model_evaluation_subset,
            save_predictions=arguments.save_all_predictions,
        )
    # else:
    #     model_executer.evaluation()

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--train_dir",
        type=str,
        default=f"/home/beinhaud/diplomka/mcs-source/dataset/train_dataset/compressed_spikes/trimmed/size_{nn_model.globals.TIME_STEP}",
        help="Directory where train dataset is stored.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=f"/home/beinhaud/diplomka/mcs-source/dataset/test_dataset/compressed_spikes/trimmed/size_{nn_model.globals.TIME_STEP}",
        help="Directory where tests dataset is stored.",
    )
    parser.add_argument(
        "--subset_dir",
        type=str,
        default=f"/home/beinhaud/diplomka/mcs-source/dataset/model_subsets/size_{int(nn_model.globals.SIZE_MULTIPLIER*100)}.pkl",
        help="Directory where model subset indices are stored.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/best_models/",
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
        default="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/experiments/experiments_subset_10.pkl",
        help="Path to selected experiments used for model analysis during evaluation.",
    )
    parser.add_argument(
        "--neuron_selection_path",
        type=str,
        default=f"/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/neurons/model_size_{int(nn_model.globals.SIZE_MULTIPLIER*100)}_subset_10.pkl",
        help="Path to selected neuron IDs used for model analysis during evaluation.",
    )
    parser.add_argument(
        "--selection_results_dir",
        type=str,
        default="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/neuron_responses/",
        help="Path to selected neuron IDs used for model analysis during evaluation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=[model_type.value for model_type in ModelTypes],
        help="Model variant that we want to use.",
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
    # parser.set_defaults(neuron_not_residual=True)
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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.007,
        help="Learning rate to use in model training.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs for training the model.",
    )
    parser.set_defaults(best_model_evaluation=False)
    parser.add_argument(
        "--best_model_evaluation",
        action="store_true",
        help="Runs only evaluation on the best saved model for the given parameters.",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="",
        help="Directory where the results of the evaluation should be saved in case of saving all evaluation predictions.",
    )
    parser.set_defaults(save_all_predictions=False)
    parser.add_argument(
        "--save_all_predictions",
        action="store_true",
        help="Whether we want to store all model predictions in final evaluation.",
    )
    parser.add_argument(
        "--full_evaluation_dir",
        type=str,
        default="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/full_evaluation_results/",
        help="Directory where the results of the evaluation should be saved in case of saving all evaluation predictions.",
    )
    parser.set_defaults(debug=True)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Start debugging mode.",
    )

    args = parser.parse_args()
    main(args)
