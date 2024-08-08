import os
import scipy.sparse as sp
# from scipy.sparse import load_npz
import numpy as np
import argparse

from tqdm import tqdm


file_prefix = "spikes_"
file_postfix = ".npz"


def trim_sparse_matrix(matrix, interval_size: int=712, start_offset: int=76):
    """
    Trim the sparse matrix into submatrices of a given interval size.
    If the matrix cannot be evenly divided, the last submatrix will be smaller.
    """
    num_time_steps, _ = matrix.shape
    trimmed_matrices = []

    # for start_row in range(0, n_rows, interval_size):
    for i in range(start_offset, num_time_steps, interval_size):
        if i == start_offset:
            trimmed_matrices.append(matrix[:i+interval_size, :].transpose(1, 0))
        else:
            trimmed_matrices.append(matrix[i:i+interval_size, :].transpose(1, 0))
    
    return trimmed_matrices


def process_directory(input_dir, output_dir, interval_size=712):
    """
    Load all sparse matrices from the input directory, trim them, 
    and save the trimmed matrices to the output directory.
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # Process each file in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(file_postfix):
            file_path = os.path.join(input_dir, filename)
            
            # Load the matrix
            matrix = sp.load_npz(file_path)        
            matrix = matrix.toarray().transpose(1, 0)

            matrix_base_number = int(os.path.splitext(filename)[0].split("_")[-1])

            # Trim the matrix
            trimmed_matrices = trim_sparse_matrix(matrix, interval_size)
            
            # Save each trimmed matrix
            for i, trimmed_matrix in enumerate(trimmed_matrices):
                output_filename = os.path.join(
                        output_dir + "/", 
                        file_prefix + str(matrix_base_number + i) + file_postfix,
                    )
                sp.save_npz(output_filename, sp.csr_matrix(trimmed_matrix))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sparse matrices by summing over specified time intervals.")
    parser.add_argument("input_directory", type=str, help="Path to the input d irectory containing .npz files.")
    parser.add_argument("output_directory", type=str, help="Path to the output directory where processed files will be saved.")
    parser.add_argument("--interval_size", type=int, default=712, help="Size of the time interval to sum the values.")
    parser.add_argument("--sheet", type=str, default=None, help="Extract only 1 sheet from: [V1_Exc_L23,V1_Exc_L4, V1_Inh_L23, V1_Inh_L4, X_OFF, X_ON]")

    args = parser.parse_args()

    # INPUT_DIRECTORY = "/home/david/source/diplomka/dataset/spikes"
    SUBDIRECTORIES = [
        "V1_Exc_L23",
        "V1_Exc_L4",
        "V1_Inh_L23",  
        "V1_Inh_L4",  
        "X_OFF",
        "X_ON",
    ]

    if args.sheet != None:
        if args.sheet in SUBDIRECTORIES:
            SUBDIRECTORIES = [args.sheet]
    
    for subdirectory in SUBDIRECTORIES:
        print(f"Processing subdirectory: {subdirectory}")
        process_directory(input_dir=args.input_directory  + "/" + subdirectory, output_dir=args.output_directory + "/" + subdirectory , interval_size=args.interval_size)
