"""
This script serves for random generating of model subsets.
"""

from typing import Dict, Optional
import argparse
import random
import pickle

import nn_model.globals
import nn_model.visible_neurons_handler as VisibleNeuronsHandler

SUBSET_DIRECTORY = "generated_subsets/"

def generate_random_subset(subset_ratio: float, original_sizes: Dict[str, int]) -> Dict[str, int]:
    """
    Generates a random subset of indices for each layer based on the specified ratio.

    :param subset_ratio: Ratio of the subset size for each layer (0.0 - 1.0).
    :param original_sizes: Dictionary containing the original sizes of each layer.
    :return: Returns a dictionary with layer names as keys and lists of indices as values.
    """
    subset_dict = {}

    for layer_name, layer_size in original_sizes.items():
        subset_size = max(
            1, int(subset_ratio * layer_size)
        )  # Ensure at least one index
        subset_dict[layer_name] = sorted(random.sample(range(layer_size), subset_size))

    return subset_dict


def one_subset_generation(subset_ratio, output_file, original_sizes: Dict[str, int]):
    subset_dict = generate_random_subset(subset_ratio, original_sizes)

    with open(output_file, "wb") as f:
        pickle.dump(subset_dict, f)

    print(f"Subset saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random subset of indices for each layer."
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=0.1,
        help="Ratio of the subset size for each layer (0.0 - 1.0).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Path to save the subset dictionary as a pickle file.",
    )
    parser.add_argument(
        "--num_subsets",
        type=int,
        default=1,
        help="Number of subset variants to generate.",
    )
    parser.set_defaults(visible_neurons=False)
    parser.add_argument(
        "--visible_neurons",
        action="store_true",
        help="Whether we want to generate visible neurons subset.",
    )
    
    args = parser.parse_args()

    ratio_percentage = int(args.subset_ratio * 100)
    preliminary_output_file = args.output_file
    original_sizes = {}
    
    if args.output_file == "":
        if args.visible_neurons:
            preliminary_output_file = VisibleNeuronsHandler.get_visible_indices_path(args.subset_ratio, directory_path = SUBSET_DIRECTORY)
            original_sizes = nn_model.globals.MODEL_SIZES
        else:
            preliminary_output_file = SUBSET_DIRECTORY + f"size_{ratio_percentage}.pkl"
            original_sizes = nn_model.globals.ORIGINAL_SIZES

    for i in range(args.num_subsets):
        if args.num_subsets > 1:
            args.output_file = preliminary_output_file + f"size_{ratio_percentage}_variant_{i}.pkl"

        one_subset_generation(args.subset_ratio, args.output_file, original_sizes)


if __name__ == "__main__":
    main()
