#!/bin/bash

# Run wintermute batch job to trim all sheets of spikes.

INPUT_DIRECTORY="/home/beinhaud/diplomka/mcs-source/dataset/spikes"
OUTPUT_DIRECTORY="/home/beinhaud/diplomka/mcs-source/dataset/trimmed_spikes"

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
    sbatch dataset_processor/time_trimmer/run_time_trimmer.sh \
        $INPUT_DIRECTORY $OUTPUT_DIRECTORY $sheet
done