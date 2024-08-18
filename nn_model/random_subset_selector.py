"""
Generates random subset of either all neuronal layers of the model
with the current setup (defined in globals). Or does random train-test 
split for the given data range.
"""
import random
import argparse
import pickle

import numpy as np

import globals


def random_subset(layer_size, sample_size):
    return np.array(sorted(random.sample(range(0, layer_size), sample_size)))

def random_model_layers():
    return {
        "X_ON": random_subset(globals.ORIGINAL_X_ON_SIZE, globals.X_ON_SIZE),
        "X_OFF": random_subset(globals.ORIGINAL_X_OFF_SIZE, globals.X_OFF_SIZE),
        "V1_Exc_L4": random_subset(globals.ORIGINAL_L4_EXC_SIZE, globals.L4_EXC_SIZE),
        "V1_Inh_L4": random_subset(globals.ORIGINAL_L4_INH_SIZE, globals.L4_INH_SIZE),
        "V1_Exc_L23": random_subset(globals.ORIGINAL_L23_EXC_SIZE, globals.L23_EXC_SIZE),
        "V1_Inh_L23": random_subset(globals.ORIGINAL_L23_INH_SIZE, globals.L23_INH_SIZE),
    }

def generate_model_subset(args):
    if args.filename is None:
        args.filename = f"size_{int(globals.SIZE_MULTIPLIER*100)}.pkl"
    # Save dictionary to a pickle file
    with open(args.output_directory + args.filename, 'wb') as pickle_file:
        pickle.dump(random_model_layers(), pickle_file)

def train_test_split(args):
    if args.filename is None:
        args.filename = f"size_{int(args.test_ratio*100)}.pkl"
    # Save dictionary to a pickle file
    with open(args.output_directory + args.filename, 'wb') as pickle_file:
        test_examples_indices = random_subset(args.num_examples, int(args.test_ratio*100)) + args.offset
        pickle.dump(test_examples_indices, pickle_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly select subset of either layer or dataset examples.")
    parser.add_argument("output_directory", type=str, help="Path to the output directory where processed files will be saved.")
    parser.add_argument("--task", type=str, help="Type of the task ('layer_subset' or 'train_test').")
    parser.add_argument("--filename", type=str, default=None, help="Filename of the result.")
    parser.add_argument("--num_examples", type=int, default=500, help="Number of examples for train-test split.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of the test set.")
    parser.add_argument("--offset", type=int, default=500, help="Offset of the generated numbers.")

    args = parser.parse_args()

    if args.task == "layer_subset":
        generate_model_subset(args)
    elif args.task == "train_test":
        train_test_split(args)