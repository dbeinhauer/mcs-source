#!/bin/bash

#SBATCH --job-name=export_dataset
#SBATCH --output=output_dir/output_%j.txt   
#SBATCH --ntasks=4  # Requesting n processors
#SBATCH --nodes=1
#SBATCH --hint=nomultithread

#SBATCH --exclude=w[1-2,10-12]


# Job for srun command which runs dataset extractions in loop for specified interval of data and sheets.


# Define an array of base folders
base_folders=(
    # L2/3
    # "/CSNG/baroni/mozaik-models/LSV1M/20240117-111742[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}" 
    # L4 and LGN
    "/CSNG/baroni/mozaik-models/LSV1M/20240124-093921[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}" 
    )

# Define an array of sheets to export
sheets=(
    "V1_Exc_L4"
    # "V1_Inh_L4"
    # "X_ON"
    # "X_OFF"
    # "V1_Exc_L2/3"
    # "V1_Inh_L2/3"
    )


echo "New experiment"
echo $base_folders
echo $sheets
echo "---------------------------------"
echo


# Loop over each base folder
for base_folder in "${base_folders[@]}"; do
    for sheet in "${sheets[@]}"; do 
        # Loop through each directory starting with given name $1
        find "$base_folder" -type d -name $1 | while read folder; do
            python3 /home/beinhaud/diplomka/dataset_creation/export_dataset.py --input_path=$folder --sheet=$sheet
        done
    done
done



