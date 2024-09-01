#!/bin/bash

# Script to copy selected subset of the dataset into the given directory.
# Used mainly for testing tools for modifying the dataset.

# Directory to copy the subset of files to
destination_directory="dataset/dataset_raw/artificial_spikes_50_examples/"

path_to_files="dataset/spikes/"
filestring="spikes_%d.npz"

sheets=(
    "V1_Exc_L23/"
    "V1_Exc_L4/"
    "V1_Inh_L23/"  
    "V1_Inh_L4/"  
    "X_OFF/"
    "X_ON/"
)

# Loop through numbers 1 to 50
for sheet in "${sheets[@]}"; 
do 
    destination="$destination_directory$sheet"
    for (( i=50000; i<55000; i+=100 ));
    do
        # Format the filename with the current number
        filename=$(printf $filestring $i)
        
        path_to_file="$path_to_files$sheet$filename"

        # Check if the file exists
        if [[ -f "$path_to_file" ]]; then
            # Copy the file to the destination directory
            echo $path_to_file
            cp "$path_to_file" "$destination"
        fi
    done
    echo $destination
done
