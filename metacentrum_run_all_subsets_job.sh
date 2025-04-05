#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Runs the provided bash job for multiple subset variants."
    echo "Usage: $0 <num_of_subsets> <bash_job_filename>"
    exit 1
fi

# Read arguments
NUM_ITERATIONS=$1
BASH_JOB_FILENAME=$2

# Check if the provided bash job file exists
if [ ! -f "$BASH_JOB_FILENAME" ]; then
    echo "Error: File '$BASH_JOB_FILENAME' not found!"
    exit 1
fi

# Iterate and run the bash job with the iteration number as a parameter
for ((i = 0; i < NUM_ITERATIONS; i++)); do
    echo "Running iteration $i..."
    bash "$BASH_JOB_FILENAME" "$i"
done

echo "All iterations completed."
