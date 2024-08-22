#!/bin/bash

#SBATCH --job-name=compress_dataset
#SBATCH --output=output_dir/output_%j.txt   
#SBATCH --ntasks=4  # Requesting n processors
#SBATCH --nodes=1
#SBATCH --hint=nomultithread

#SBATCH --exclude=w[1-2,9-12]


# Ensure the script receives the necessary arguments
if [ "$#" -ne 4 ]; then
    echo "Run wintermute batch job to merge time intervals for the given sheet"
    echo "Usage: $0 <input_paht> <output_path> <time_interval> <sheet>"
    exit 1
fi

python3 /home/beinhaud/diplomka/mcs-source/dataset_processor/time_merger/time_interval_merger.py \
    $1 $2 --time_interval=$3 --sheet=$4
