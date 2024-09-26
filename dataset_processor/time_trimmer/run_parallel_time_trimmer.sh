#!/bin/bash

# Run wintermute batch job to trim all sheets of spikes.

# Ensure the script receives the necessary arguments
if [ "$#" -ne 1 ]; then
    echo "Run wintermute batch jobs to trim the given sheet for all spikes in the given directory."
    echo "Usage: $0 <dataset_variant>"
    echo ""
    echo "Dataset variant one of ['train', 'test']."
    exit 1
fi

# Base directory containing all datasets.
BASE_DIR="/home/beinhaud/diplomka/mcs-source/dataset/"

# Subdirectories containg original spikes (input) and trimmed spikes (output).
SPIKES_SUBDIR="spikes/"
TRIMMED_SPIKES_SUBDIR="trimmed_spikes/"

# Set the subdirectory of the dataset (either train or test data). They differ by number of trials.
if [[ "$1" == "train" ]]; then
    DATASET_VARIANT_SUBDIR="train_dataset/"
elif [[ "$1" == "test" ]]; then
    DATASET_VARIANT_SUBDIR="test_dataset/"
else
    echo "Error: Argument must be 'train' or 'test'"
    exit 1
fi

BASE_DIR="$BASE_DIR""$DATASET_VARIANT_SUBDIR"

# Set of all sheet variants.
sheets=(
    "V1_Exc_L23"
    "V1_Exc_L4"
    "V1_Inh_L23"  
    "V1_Inh_L4"  
    "X_OFF"
    "X_ON"
)

# Set input and output directories:
INPUT_DIRECTORY="$BASE_DIR""$SPIKES_SUBDIR"
OUTPUT_DIRECTORY="$BASE_DIR""$TRIMMED_SPIKES_SUBDIR"

for sheet in "${sheets[@]}"; 
do
    sbatch /home/beinhaud/diplomka/mcs-source/dataset_processor/time_trimmer/run_time_trimmer.sh \
        $INPUT_DIRECTORY $OUTPUT_DIRECTORY $sheet
done