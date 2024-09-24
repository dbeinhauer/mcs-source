#!/bin/bash

# Script that runs dataset extraction for the missing part of dataset 
# (parts that failed during proper extraction)

# Ensure the script receives the necessary arguments
if [ "$#" -ne 2 ]; then
    echo "Run the wintermute batch job to export dataset for missing data in generated dataset."
    echo "Usage: $0 <sheet> <dataset_variant>"
    echo ""
    echo "Sheet variants: [V1_Exc_L4, V1_Inh_L4, X_ON, X_OFF, V1_Exc_L23, V1_Inh_L23]"
    echo "Dataset vaiants: [train, test]"
    exit 1
fi

base_path="/home/beinhaud/diplomka/mcs-source/dataset/" # Base directory.
spikes_dir="spikes/"    # Spikes subdirectory.
sheet=$1    # Sheet to process.

# Variables for train/test selection.
train_index=0   # Train dataset subdirectory.
test_index=1    # Test dataset subdirectory.

# Variant index (whether train or test).
variant_index=$train_index
if [[ "$2" == "test" ]]; then
    variant_index=$test_index
fi

# Variants of dataset subdirectory (train/test).
dataset_variants=(
    "train_dataset"
    "test_dataset"
)
# Start, end and steps for the train/test variants.
start_indices=(
    50000
    300050
)
end_indices=(
    100000
    301000
)
step_sizes=(
    100
    50
)


# Specify dataset directory (to check the existing values and store the rest).
# paths=(
#     "/home/beinhaud/diplomka/mcs-source/dataset/spikes/X_ON/"
#     "/home/beinhaud/diplomka/mcs-source/dataset/spikes/X_OFF/"
#     "/home/beinhaud/diplomka/mcs-source/dataset/spikes/V1_Exc_L4/"
#     "/home/beinhaud/diplomka/mcs-source/dataset/spikes/V1_Inh_L4/"
#     "/home/beinhaud/diplomka/mcs-source/dataset/spikes/V1_Exc_L23/"
#     "/home/beinhaud/diplomka/mcs-source/dataset/spikes/V1_Inh_L23/"
# )

# Example of path for train variant index and X_ON layer:
#   /home/beinhaud/diplomka/mcs-source/dataset/train_dataset/spikes/X_ON/
path="$base_path${dataset_variant[$variant_index]}$spikes_dir$sheet/"


# Check for all parts of dataset. If some missing -> start extracting it.
# for path in "${paths[@]}"; do
# for (( i=50000; i<100000; i+=100 ));  do
for (( i=${start_indices[$variant_index]}; i<${end_indices[$variant_index]}; i+=${step_sizes[$variant_index]} ))
    if [ $i -eq 300150]; then
        # Skip the part 300150-300200 as it is corrupted.
        continue
    fi
    # Part of dataset already exists -> skip it
    if ls "${path}"*"${i}"* 1> /dev/null 2>&1; then
        continue
    fi

    # Missing part -> extract it.
    filename="NewDataset_Images_from_${i}*"

    sbatch export_test.sh $filename
done
# done