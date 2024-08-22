#!/bin/bash

# Ensure the script receives the necessary arguments
if [ -z "$1" ]; then
    echo "Run wintermute batch job to merge time intervals"
    echo "Usage: $0 <time_interval_length>"
    exit 1
fi

INPUT_DIRECTORY="/home/beinhaud/diplomka/mcs-source/dataset/trimmed_spikes"
OUTPUT_PART="/home/beinhaud/diplomka/mcs-source/dataset/compressed_spikes/trimmed"

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