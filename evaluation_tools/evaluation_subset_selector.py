import argparse
import numpy as np
import os
import random
import sys
import pickle

# Add the parent directory to the Python path to access globals.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../nn_model")))
import nn_model.globals


def get_exp_number(filename):
    """Extract the exp_number from the filename. We expect: spikes_trial_{num_trial}_{image_id}.npz"""
    return filename.split("_")[-1]


def select_random_subset(model_sizes, data_files, num_indices, num_examples):
    """Select random subset of indices and examples, ensuring unique exp_number for each example."""
    selected_indices = {}

    # For each layer, select random indices from the range (0, model_size - 1)
    for layer, model_size in model_sizes.items():
        selected_indices[layer] = np.random.choice(
            model_size, num_indices, replace=False
        )

    # Ensure unique exp_number for the examples
    selected_examples = []
    exp_numbers = set()

    while len(selected_examples) < num_examples:
        random_file = random.choice(data_files)
        exp_number = get_exp_number(random_file)

        if exp_number not in exp_numbers:
            exp_numbers.add(exp_number)
            selected_examples.append(random_file.split("/")[-1])

    return selected_indices, selected_examples


def main(args):
    # Get model sizes from globals.MODEL_SIZES
    model_sizes = nn_model.globals.MODEL_SIZES

    # Load dataset examples
    data_files = os.listdir(args.dataset_path)
    data_files = [
        os.path.join(args.dataset_path, file)
        for file in data_files
        if file.endswith(".npz")
    ]

    # Select random subset
    selected_indices, selected_examples = select_random_subset(
        model_sizes, data_files, args.num_indices, args.num_examples
    )

    print(selected_indices)
    print(selected_examples)

    neuron_indices_path = "".join(
        [
            args.output_indices_path,
            f"model_size_{int(nn_model.globals.SIZE_MULTIPLIER*100)}",
            f"_subset_{args.num_indices}",
            ".pkl",
        ]
    )

    example_subset_path = "".join(
        [
            args.output_examples_path,
            f"experiments_subset_{args.num_examples}",
            ".pkl",
        ]
    )

    # Save selected indices and examples to pickle files (if needed)
    with open(neuron_indices_path, "wb") as f:
        pickle.dump(selected_indices, f)

    with open(example_subset_path, "wb") as f:
        pickle.dump(selected_examples, f)

    print("Random subsets saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select random subset of indices and examples."
    )

    # Argument for paths
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/beinhaud/diplomka/mcs-source/dataset/test_dataset/compressed_spikes/trimmed/size_20/X_ON",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--output_indices_path",
        type=str,
        default="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/neurons/",
        help="Path to save the selected indices pickle file.",
    )
    parser.add_argument(
        "--output_examples_path",
        type=str,
        default="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/experiments/",
        help="Path to save the selected examples pickle file.",
    )

    # Argument for subset sizes
    parser.add_argument(
        "--num_indices",
        type=int,
        default=10,
        help="Number of random indices to select for each layer.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of random examples to select.",
    )

    # Parse arguments
    args = parser.parse_args()

    main(args)
