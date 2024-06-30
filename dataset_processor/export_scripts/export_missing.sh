#!/bin/bash


# Script that runs dataset extraction for the missing part of dataset 
# (parts that failed during proper extraction)

# Specify dataset directory (to check the existing values and store the rest).
paths=(
    # "/home/beinhaud/diplomka/dataset_creation/dataset/spikes/X_ON/"
    # "/home/beinhaud/diplomka/dataset_creation/dataset/spikes/X_OFF/"
    "/home/beinhaud/diplomka/dataset_creation/dataset/spikes/V1_Exc_L4/"
    # "/home/beinhaud/diplomka/dataset_creation/dataset/spikes/V1_Inh_L4/"
    # "/home/beinhaud/diplomka/dataset_creation/dataset/spikes/V1_Exc_L23/"
    # "/home/beinhaud/diplomka/dataset_creation/dataset/spikes/V1_Inh_L23/"
)


# Check for all parts of dataset. If some missing -> start extracting it.
for path in "${paths[@]}"; do
    for (( i=50000; i<100000; i+=100 ));  do

        # Part of dataset already exists -> skip it
        if ls "${path}"*"${i}"* 1> /dev/null 2>&1; then
            continue
        fi

        # Missing part -> extract it.
        filename="NewDataset_Images_from_${i}*"

        sbatch export_test.sh $filename
    done
done