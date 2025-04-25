"""
This script serves for random generating of model subsets.
"""

import argparse
import random
import pickle
from nn_model import globals

SUBSET_DIRECTORY = "model_subsets/"


def generate_random_subset(subset_ratio):
    original_sizes = globals.ORIGINAL_SIZES
    subset_dict = {}

    for layer_name, layer_size in original_sizes.items():
        subset_size = max(
            1, int(subset_ratio * layer_size)
        )  # Ensure at least one index
        subset_dict[layer_name] = sorted(random.sample(range(layer_size), subset_size))

    return subset_dict


def one_subset_generation(subset_ratio, output_file):
    subset_dict = generate_random_subset(subset_ratio)

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

    args = parser.parse_args()

    ratio_percentage = int(args.subset_ratio * 100)

    if args.output_file == "":
        args.output_file = SUBSET_DIRECTORY + f"size_{ratio_percentage}.pkl"

    for i in range(args.num_subsets):
        if args.num_subsets > 1:
            args.output_file = (
                SUBSET_DIRECTORY + f"size_{ratio_percentage}_variant_{i}.pkl"
            )

        one_subset_generation(args.subset_ratio, args.output_file)


if __name__ == "__main__":
    main()
