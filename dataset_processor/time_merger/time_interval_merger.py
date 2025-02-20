"""
Loads the dataset and merges spikes for given time interval for each neuron.
Finally, saves the new dataset into the specified directory.
"""

import glob
import os
import argparse

import scipy.sparse as sp
from tqdm import tqdm

# File postfixes.
ORIGINAL_FILE_POSTFIX = ".npz"
NEW_FILE_POSTFIX = ".npz"

# Sheets subdirectory.
SUBDIRECTORIES = [
    "V1_Exc_L23",
    "V1_Exc_L4",
    "V1_Inh_L23",
    "V1_Inh_L4",
    "X_OFF",
    "X_ON",
]


def process_and_save_matrices(
    input_directory: str,
    subdirectory: str,
    output_directory: str,
    time_interval_size: int,
):
    """
    Loads all matrices from given directory, merges spikes for
    time intervals of given size for all neurons and stores new
    matrices to given path.
    :param input_directory: directory containing data for processing.
    :param subdirectory: subdirectory containing spikes for
    specific neuronal layer.
    :param output_directory: directory where the results will be stored.
    :param time_interval_size: length of the time interval for which we
    want to merge the information of spikes.
    """
    # Ensure the output directory exists.
    os.makedirs(output_directory + "/" + subdirectory, exist_ok=True)

    # Find all .npz files in the specified directory
    files = glob.glob(f"{input_directory}/{subdirectory}/*.npz")

    for file in tqdm(files):
        sparse_matrix = sp.load_npz(file)
        dense_matrix = sparse_matrix.toarray()

        num_neurons, time_duration = dense_matrix.shape
        new_time_duration = time_duration // time_interval_size

        # Reshape the matrix to sum over the specified time interval
        reshaped_matrix = dense_matrix[
            :, : new_time_duration * time_interval_size
        ].reshape(num_neurons, new_time_duration, time_interval_size)
        new_matrix = reshaped_matrix.sum(axis=2)

        # Save the new matrix in the output directory (and change the filename).
        new_file_name = os.path.join(
            output_directory + "/" + subdirectory + "/",
            os.path.basename(file).replace(ORIGINAL_FILE_POSTFIX, NEW_FILE_POSTFIX),
        )
        sp.save_npz(new_file_name, sp.csr_matrix(new_matrix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process sparse matrices by summing over specified time intervals."
    )
    parser.add_argument(
        "input_directory",
        type=str,
        help="""Path to the input directory containing sheets subdirectories
        with .npz files with appropriate neurons.""",
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="""Path to the output directory where processed files will
        be saved (in appropriate subdirectories).""",
    )
    parser.add_argument(
        "--time_interval",
        type=int,
        default=10,
        help="Size of the time interval to merge the values.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Extract only 1 sheet. Select from: [V1_Exc_L23,V1_Exc_L4, V1_Inh_L23, V1_Inh_L4, X_OFF, X_ON]",
    )

    args = parser.parse_args()

    # Process only the given sheet (otherwise all sheets).
    if args.sheet is not None and args.sheet in SUBDIRECTORIES:
        SUBDIRECTORIES = [args.sheet]

    for subdirectory in SUBDIRECTORIES:
        print(f"Processing subdirectory: {subdirectory}")
        process_and_save_matrices(
            args.input_directory,
            subdirectory,
            args.output_directory,
            args.time_interval,
        )
