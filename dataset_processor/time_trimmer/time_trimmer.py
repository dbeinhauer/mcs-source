"""
Trims the spikes matrix to multiple smaller ones representing smaller time interval
and stores them into new destination.

It should be used to trim the time intervals the way that there should be
blank part (typically second half), whole stimuli (image) part and first
half of the blank part (these can be repeated). To ensure correct format used
in the neural simulator model (the sizes depend on the used dataset).
"""

from typing import List

import os
import argparse

from tqdm import tqdm
import scipy.sparse as sp
import numpy as np

FILE_POSTFIX = ".npz"


def trim_sparse_matrix(
    matrix: np.ndarray, interval_size: int = 712, start_offset: int = 151
) -> List[np.ndarray]:
    """
    Trims the sparse matrix into sub-matrices of a given interval size. It skips the first blank
    stage and trims each pair natural-blank stimulus. The last experiment in the sequence
    is smaller since there is not last blank stage
    (should be padded to uniform size in the future processing).

    :param matrix: Numpy matrix for trimming in shape `(time_duration, num_neurons)`.
    :param interval_size: Size of the wanted time interval. Typically one experiment
    stimulus+blank stage. (first and last will have different sizes).
    :param start_offset: Offset from which start trimming (to skip first blank part).
    :return: Returns list of trimmed numpy matrices in different
    shape `(num_neurons, time_duration)`.
    """
    trimmed_matrices = []
    num_time_steps, _ = matrix.shape

    # Start counting from the offset (we want to start from the first stimulus -> skip first blank).
    # Trim the matrix in time dimension.
    for i in range(start_offset, num_time_steps, interval_size):
        # For each new matrix switch dimensions (for future processing).
        trimmed_matrices.append(matrix[i : i + interval_size, :].transpose(1, 0))

    return trimmed_matrices


def process_directory(input_dir: str, output_dir: str, interval_size: int = 712):
    """
    Load all sparse matrices from the input directory, trim them, and store
    them in sparse representation to the output directory.

    :param input_dir: input directory where the matrices are stored.
    :param output_dir: output directory where to store the new matrices.
    :param interval_size: size of the new time interval
    (should be set based on the dataset).
    """
    # Process each target file in the input directory.
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(FILE_POSTFIX):
            file_path = os.path.join(input_dir, filename)

            # Load the matrix and switch the shape to (time_duration, num_neurons).
            matrix = sp.load_npz(file_path)
            matrix = matrix.toarray().transpose(1, 0)

            # Get the original experiment ID from the filename
            # (should be in format: 'spikes_{number}.npz').
            matrix_base_number = int(os.path.splitext(filename)[0].split("_")[-1])

            # Get rid of the image number and postfix.
            file_prefix = filename.split("/")[-1].split(
                f"_{matrix_base_number}{FILE_POSTFIX}"
            )[0]

            trimmed_matrices = trim_sparse_matrix(matrix, interval_size)

            # Save each trimmed matrix
            for i, trimmed_matrix in enumerate(trimmed_matrices):
                # Output format should correspond to original number + trimmed order number.
                # The original filenames numbers should be in format that the new ones
                # should not overlap (padding should be at least 100 and maximal new
                # matrices should be 100).
                output_filename = os.path.join(
                    output_dir + "/",
                    file_prefix + "_" + str(matrix_base_number + i) + FILE_POSTFIX,
                )
                sp.save_npz(output_filename, sp.csr_matrix(trimmed_matrix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the given matrices to smaller ones in the time interval dimension."
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
        help="""Path to the output directory where processed files will be saved
        (in appropriate subdirectories).""",
    )
    parser.add_argument(
        "--interval_size", type=int, default=712, help="Size of new time interval."
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="",
        help="Extract only 1 sheet from: [V1_Exc_L23, V1_Exc_L4, V1_Inh_L23, V1_Inh_L4, X_OFF, X_ON]",
    )

    args = parser.parse_args()

    SUBDIRECTORIES = [
        "V1_Exc_L23",
        "V1_Exc_L4",
        "V1_Inh_L23",
        "V1_Inh_L4",
        "X_OFF",
        "X_ON",
    ]

    if args.sheet:
        # Trim only selected layer data.
        if args.sheet in SUBDIRECTORIES:
            SUBDIRECTORIES = [args.sheet]

    for subdirectory in SUBDIRECTORIES:
        # Process all layers.
        print(f"Processing subdirectory: {subdirectory}")
        process_directory(
            input_dir=args.input_directory + "/" + subdirectory,
            output_dir=args.output_directory + "/" + subdirectory,
            interval_size=args.interval_size,
        )
