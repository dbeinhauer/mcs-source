#!/bin/bash

# Ensure the script receives the necessary arguments
# if [ -z "$1" ]; then
if [ "$#" -ne 2 ]; then
    echo "Run wintermute batch job to merge time intervals."
    echo "Usage: $0 <merged_interval_size> <dataset_variant>"
    echo ""
    echo "Dataset variant one of ['train', 'test']."
    exit 1
fi

# Base directory containing all datasets.
BASE_DIR="/home/beinhaud/diplomka/mcs-source/dataset/"

# Subdirectories containg original spikes (input) and trimmed spikes (output).
TRIMMED_SPIKES_SUBDIR="trimmed_spikes/"
MERGED_SPIKES_SUBDIR="compressed_spikes/trimmed"

# Set the subdirectory of the dataset (either train or test data). They differ by number of trials.
if [[ "$2" == "train" ]]; then
    DATASET_VARIANT_SUBDIR="train_dataset/"
elif [[ "$2" == "test" ]]; then
    DATASET_VARIANT_SUBDIR="test_dataset/"
else
    echo "Error: Argument must be 'train' or 'test'"
    exit 1
fi

BASE_DIR="$BASE_DIR""$DATASET_VARIANT_SUBDIR"

# INPUT_DIRECTORY="/home/beinhaud/diplomka/mcs-source/dataset/trimmed_spikes"
# OUTPUT_PART="/home/beinhaud/diplomka/mcs-source/dataset/compressed_spikes/trimmed"

INPUT_DIRECTORY="$BASE_DIR""$TRIMMED_SPIKES_SUBDIR"
OUTPUT_PART="$BASE_DIR""$MERGED_SPIKES_SUBDIR"

interval_size=$1

OUTPUT_DIRECTORY="$OUTPUT_PART/size_$interval_size"

sheets=(
    "V1_Exc_L23"
    "V1_Exc_L4"
    "V1_Inh_L23"  
    "V1_Inh_L4"  
    "X_OFF"
    "X_ON"
)

for sheet in "${sheets[@]}"; 
do
    sbatch /home/beinhaud/diplomka/mcs-source/dataset_processor/time_merger/run_time_merger.sh \
        $INPUT_DIRECTORY $OUTPUT_DIRECTORY $interval_size $sheet
done