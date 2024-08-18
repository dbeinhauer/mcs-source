#!/bin/bash

#SBATCH --job-name=compress_dataset
#SBATCH --output=output_dir/output_%j.txt   
#SBATCH --ntasks=4  # Requesting n processors
#SBATCH --nodes=1
#SBATCH --hint=nomultithread

#SBATCH --exclude=w[1-2,9-12]

# Ensure the script receives the necessary arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_paht> <output_path> <sheet>"
    exit 1
fi

python3 /home/beinhaud/diplomka/mcs-source/dataset_processor/time_trimmer/time_trimmer.py \
    $1 $2 --sheet=$3
