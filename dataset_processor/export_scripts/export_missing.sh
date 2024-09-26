#!/bin/bash

# Script that runs dataset extraction for the missing part of dataset 
# (parts that failed during proper extraction)

# Ensure the script receives the necessary arguments
if [ "$#" -ne 2 ]; then
    echo "Run the wintermute batch job to export dataset for missing data in generated dataset."
    echo "Usage: $0 <sheet> <dataset_variant>"
    echo ""
    echo "Sheet variants: [V1_Exc_L4, V1_Inh_L4, X_ON, X_OFF, V1_Exc_L2/3, V1_Inh_L2/3]"
    echo "Dataset vaiants: [train, test]"
    exit 1
fi

base_path="/home/beinhaud/diplomka/mcs-source/dataset/" # Base directory.
spikes_dir="spikes/"    # Spikes subdirectory.

# Define sheet directory (need to tackle the L23 layers).
sheet_dir=$1
if [[ "$1" == "V1_Exc_L2/3" ]]; then
    sheet_dir="V1_Exc_L23"
elif [[ "$1" == "V1_Inh_L2/3" ]]; then
    sheet_dir="V1_Inh_L23"
fi

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
    "train_dataset/"
    "test_dataset/"
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

# Example of path for train variant index and X_ON layer:
#   /home/beinhaud/diplomka/mcs-source/dataset/train_dataset/spikes/X_ON/
path="$base_path""${dataset_variants[$variant_index]}""$spikes_dir""$sheet_dir/"

# Check for all parts of dataset. If some missing -> start extracting it.
for (( i=${start_indices[$variant_index]}; i<${end_indices[$variant_index]}; i+=${step_sizes[$variant_index]} )); do
    if [ $i -eq 300150 ]; then
        # Skip the part 300150-300200 as it is corrupted.
        continue
    fi
    # Part of dataset already exists -> skip it
    if ls "${path}"*"${i}"* 1> /dev/null 2>&1; then
        continue
    fi

    # Missing part -> extract it.
    filename="NewDataset_Images_from_${i}*"

    sbatch /home/beinhaud/diplomka/mcs-source/dataset_processor/export_scripts/export_job.sh $filename $1 $2
done