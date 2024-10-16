"""
Generates random subset of either all neuronal layers of the model
with the current setup (defined in globals). Or does random train-test 
split for the given data range.
"""

import os
import random
import argparse
import pickle
from typing import Dict

import numpy as np

import globals
from type_variants import LayerType


def random_subset(layer_size, sample_size):
    """
    Randomly selects indices of the subset for the given layer.

    :param layer_size: original size of the layer.
    :param sample_size: size of the wanted subset.
    :return: Returns array of indices of the randomly selected subset.
    """
    return np.array(sorted(random.sample(range(0, layer_size), sample_size)))


def random_model_layers() -> Dict:
    """
    Randomly selects subsets of all the model layers.

    :return: Returns dictionary where key is layer name and values are `np.arrays`
    of indices of the corresponding subsets.
    """
    return {
        LayerType.X_ON.value: random_subset(
            globals.ORIGINAL_X_ON_SIZE, globals.X_ON_SIZE
        ),
        LayerType.X_OFF.value: random_subset(
            globals.ORIGINAL_X_OFF_SIZE, globals.X_OFF_SIZE
        ),
        LayerType.V1_Exc_L4.value: random_subset(
            globals.ORIGINAL_L4_EXC_SIZE, globals.L4_EXC_SIZE
        ),
        LayerType.V1_Inh_L4.value: random_subset(
            globals.ORIGINAL_L4_INH_SIZE, globals.L4_INH_SIZE
        ),
        LayerType.V1_Exc_L23.value: random_subset(
            globals.ORIGINAL_L23_EXC_SIZE, globals.L23_EXC_SIZE
        ),
        LayerType.V1_Inh_L23.value: random_subset(
            globals.ORIGINAL_L23_INH_SIZE, globals.L23_INH_SIZE
        ),
    }


def generate_model_subset(arguments):
    """
    Generates indices of the model subset of the size specified in `globals`.
    After generation it stores the subset indices into pickle file.

    :param arguments: command line arguments.
    """
    if arguments.filename is None:
        # Filename not defined -> use default: "size_{model_size}.pkl"
        arguments.filename = f"size_{int(globals.SIZE_MULTIPLIER*100)}.pkl"

    file_path = os.path.join(arguments.output_directory, arguments.filename)

    # Check if the file already exists
    if os.path.exists(file_path):
        if arguments.rewrite:
            # We want to rewrite the subset indices with the new ones.
            print(f"File {arguments.filename} exists, rewriting...")
        else:
            # We do not want to rewrite already existing indices -> skip generation
            print(f"File {arguments.filename} exists, skipping creation.")
            return
    else:
        # The subset indices file does not exist -> generate one
        print(f"File {arguments.filename} does not exist, creating a new one...")

    # Save dictionary to a pickle file.
    with open(file_path, "wb") as pickle_file:
        pickle.dump(random_model_layers(), pickle_file)


def train_test_split(arguments):
    """
    Performs train/test split for example indices.

    NOTE: not used now as we do not have all example multi-trial.

    :param arguments: command line arguments.
    """
    if arguments.filename is None:
        arguments.filename = f"size_{int(arguments.test_ratio*100)}.pkl"

    file_path = os.path.join(arguments.output_directory, arguments.filename)

    # Check if the file already exists
    if os.path.exists(file_path):
        if arguments.rewrite:
            # We want to rewrite the subset indices with the new ones.
            print(f"File {arguments.filename} exists, rewriting...")
        else:
            # We do not want to rewrite already existing indices -> skip generation
            print(f"File {arguments.filename} exists, skipping creation.")
            return
    else:
        # The subset indices file does not exist -> generate one
        print(f"File {arguments.filename} does not exist, creating a new one...")

    # Save dictionary to a pickle file
    with open(file_path, "wb") as pickle_file:
        test_examples_indices = (
            random_subset(arguments.num_examples, int(arguments.test_ratio * 100))
            + arguments.offset
        )
        pickle.dump(test_examples_indices, pickle_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly select subset of either layer or dataset examples."
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="Path to the output directory where processed files will be saved.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="layer_subset",
        choices=["layer_subset", "train_test"],
        help="Type of the task ('layer_subset' or 'train_test').",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="""If the old file should be rewritten in case the new one is of
        the same filename (otherwise skip the creation of new subset).""",
    )
    parser.set_defaults(rewrite=False)
    parser.add_argument(
        "--filename", type=str, default=None, help="Filename of the result."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=500,
        help="Number of examples for train-test split.",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Ratio of the test set."
    )
    parser.add_argument(
        "--offset", type=int, default=500, help="Offset of the generated numbers."
    )

    args = parser.parse_args()

    if args.task == "layer_subset":
        generate_model_subset(args)
    elif args.task == "train_test":
        train_test_split(args)
