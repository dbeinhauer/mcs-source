import numpy as np
import scipy.sparse as sp
import glob
import argparse
import os

def process_and_save_matrices(input_directory, subdirectory, output_directory, time_interval):
    # Ensure the output directory exists
    os.makedirs(output_directory + "/" + subdirectory, exist_ok=True)
    
    # Find all .npz files in the specified directory
    files = glob.glob(f"{input_directory}/{subdirectory}/*.npz")
    
    for file in files:
        # Load the sparse matrix
        sparse_matrix = sp.load_npz(file)
        
        # Convert to dense if necessary
        dense_matrix = sparse_matrix.todense()
        
        # Get the number of experiments and time duration
        num_experiments, time_duration = dense_matrix.shape
        
        # Calculate the new time duration after summing
        new_time_duration = time_duration // time_interval
        
        # Initialize the new matrix
        new_matrix = np.zeros((num_experiments, new_time_duration))
        
        for i in range(new_time_duration):
            start = i * time_interval
            end = (i + 1) * time_interval
            new_matrix[:, i] = dense_matrix[:, start:end].sum(axis=1).A1  # .A1 to convert to 1D array
        
        # Save the new matrix in the output directory
        file_name = os.path.basename(file)
        print(file_name)
        new_file_name = os.path.join(output_directory + "/" + subdirectory + "/", 
                                    file_name.replace('.npz', f'_summed_{time_interval}.npz'))
        sp.save_npz(new_file_name, sp.csr_matrix(new_matrix))
        print(f"Saved {new_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sparse matrices by summing over specified time intervals.")
    parser.add_argument("input_directory", type=str, help="Path to the input directory containing .npz files.")
    parser.add_argument("output_directory", type=str, help="Path to the output directory where processed files will be saved.")
    parser.add_argument("--time_interval", type=int, default=10, help="Size of the time interval to sum the values.")
    
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
    
    for subdirectory in SUBDIRECTORIES:
        process_and_save_matrices(
            # INPUT_DIRECTORY, 
            args.input_directory,
            subdirectory, 
            args.output_directory, 
            args.time_interval
        )
