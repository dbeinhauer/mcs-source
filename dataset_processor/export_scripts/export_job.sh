#!/bin/bash

#SBATCH --job-name=export_dataset
#SBATCH --output=output_dir/output_%j.txt   
#SBATCH --ntasks=4  # Requesting n processors
#SBATCH --nodes=1
#SBATCH --hint=nomultithread

#SBATCH --exclude=w[1-2,10-12]


# Job for srun command which runs dataset extractions in loop for specified interval of data and sheets.

# Ensure the script receives the necessary arguments
if [ "$#" -ne 3 ]; then
    echo "Run the wintermute batch job to export dataset from specific raw data for specific sheet and dataset variant (train/test)."
    echo "Usage: $0 <input_subdirectory> <sheet> <dataset_variant>"
    exit 1
fi


# Define an array of base folders
base_folders=(
    # L2/3
    "/CSNG/baroni/mozaik-models/LSV1M/20240117-111742[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}" 
    # L4 and LGN
    "/CSNG/baroni/mozaik-models/LSV1M/20240124-093921[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}"
    # Test dataset (multiple trials).
    "/CSNG/baroni/mozaik-models/LSV1M/20240911-181115[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:20}"
    )

# Define an array of sheets to export
# other_sheets=(
#     "V1_Exc_L4"
#     "V1_Inh_L4"
#     "X_ON"
#     "X_OFF"
#     )

l23_sheets=( 
    "V1_Exc_L2/3"
    "V1_Inh_L2/3"
    )

# Set default to train L4 and LGN layer folder.
base_folder=${base_folders[1]}
if [[ "$3" == "test" ]]; then
    # Set test base folder.
    base_folder=${base_folders[2]}
else
    # Set train base folder.
    for sheet in "${l23_sheets[@]}"; do
        if [[ "$2" == "$sheet" ]]; then
            # The sheet is in L23 layer ()
            base_folder=${base_folders[0]}
            break
        fi
    done
fi

echo "New experiment"
echo $base_folder
echo $2
echo "---------------------------------"
echo


# Loop through each directory starting with given name $1
find "$base_folder" -type d -name $1 | while read folder; do
    python3 /home/beinhaud/diplomka/mcs-source/dataset_processor/export_scripts/export_dataset.py --input_path=$folder --sheet=$2
done
