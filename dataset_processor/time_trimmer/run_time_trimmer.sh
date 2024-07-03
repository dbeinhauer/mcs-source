#!/bin/bash

#SBATCH --job-name=compress_dataset
#SBATCH --output=output_dir/output_%j.txt   
#SBATCH --ntasks=4  # Requesting n processors
#SBATCH --nodes=1
#SBATCH --hint=nomultithread

#SBATCH --exclude=w[1-2,10-12]

# Ensure the script receives the necessary arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_paht> <output_path> <time_interval> <sheet>"
    exit 1
fi

# INPUT_DIRECTORY="/home/beinhaud/diplomka/dataset_creation/dataset/spikes"
# OUTPUT_DIRECTORY="/home/beinhaud/diplomka/dataset_creation/dataset/compressed_data"

# interval_size=$1

# OUTPUT_PATH="$OUTPUT_DIRECTORY/size_$interval_size"

python3 /home/beinhaud/diplomka/dataset_creation/time_interval_trimmer.py \
    $1 $2 --time_interval=$3 --sheet=$4
