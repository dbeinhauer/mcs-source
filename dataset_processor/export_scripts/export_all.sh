#!/bin/bash

# Script to run all export jobs for the given pattern.

# Ensure the script receives the necessary arguments
if [ "$#" -ne 2 ]; then
    echo "Run the wintermute batch jobs to export dataset from raw data for specific sheet."
    echo "Usage: $0 <sheet> <dataset_variant>"
    echo ""
    echo "Sheet variants: [V1_Exc_L4, V1_Inh_L4, X_ON, X_OFF, V1_Exc_L2/3, V1_Inh_L2/3]"
    echo "Dataset vaiants: [train, test]"
    exit 1
fi

# Variables for train/test selection.
train_index=0   # Train dataset subdirectory.
test_index=1    # Test dataset subdirectory.

# Variant index (whether train or test).
variant_index=$train_index
if [[ "$2" == "test" ]]; then
    variant_index=$test_index
fi

# Start, end and steps for the train/test variants.
start_indices=(
    500
    30005
)
end_indices=(
    1000
    30100
)
step_sizes=(
    1
    5
)

for (( i=${start_indices[$variant_index]}; i<${end_indices[$variant_index]}; i+=${step_sizes[$variant_index]} )); do
    if [ $i -eq 30015 ]; then
        # Skip dataset part 300150-300200, because the data are corrupted.
        continue
    fi
    # Form the string with the current number
    filename="NewDataset_Images_from_${i}*"
    sbatch export_job.sh $filename $1 $2
done