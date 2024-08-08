"""
Loads the dataset and merges spikes for each interval of given spikes.
"""
import numpy as np
import scipy.sparse as sp
import glob
import argparse
import os

from tqdm import tqdm


def process_and_save_matrices(input_directory, subdirectory, output_directory, time_interval):
    # Ensure the output directory exists
    os.makedirs(output_directory + "/" + subdirectory, exist_ok=True)
    
    # Find all .npz files in the specified directory
    files = glob.glob(f"{input_directory}/{subdirectory}/*.npz")
    
    for file in tqdm(files):
        # Load the sparse matrix
        sparse_matrix = sp.load_npz(file)
        
        # Convert to dense if necessary
        dense_matrix = sparse_matrix.toarray()

        # Get the number of experiments and time duration
        num_neurons, time_duration = dense_matrix.shape
        
        # Calculate the new time duration after summing
        new_time_duration = time_duration // time_interval
        
        # Reshape the matrix to sum over the specified time interval
        reshaped_matrix = dense_matrix[:, :new_time_duration * time_interval].reshape(num_neurons, new_time_duration, time_interval)
        
        # Sum over the last axis to get the new matrix
        new_matrix = reshaped_matrix.sum(axis=2)

        # Save the new matrix in the output directory
        file_name = os.path.basename(file)
        # print(file_name)
        new_file_name = os.path.join(output_directory + "/" + subdirectory + "/", 
                                    file_name.replace('.npz', f'_summed.npz'))
        sp.save_npz(new_file_name, sp.csr_matrix(new_matrix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sparse matrices by summing over specified time intervals.")
    parser.add_argument("input_directory", type=str, help="Path to the input d irectory containing .npz files.")
    parser.add_argument("output_directory", type=str, help="Path to the output directory where processed files will be saved.")
    parser.add_argument("--time_interval", type=int, default=10, help="Size of the time interval to sum the values.")
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
        process_and_save_matrices(
            # INPUT_DIRECTORY, 
            args.input_directory,
            subdirectory, 
            args.output_directory, 
            args.time_interval
        )
