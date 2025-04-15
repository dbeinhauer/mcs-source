#!/bin/bash

#SBATCH --job-name=histogram_creation
#SBATCH --output=output_dir/output_%j.txt   
#SBATCH --ntasks=4  # Requesting n processors
#SBATCH --nodes=1

#SBATCH --exclude=w[1-2,9-12]

SUBSET=$1

VARIANT="train"
# VARIANT="test"

TIME_STEP=20
ACTION="subset_dataset"
FILENAME="$ACTION-$VARIANT-$SUBSET.pkl"

RESULTS_BASE_DIR="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/"
RESULTS_SUBDIR="subset_dataset_analysis/"
RESULTS_SAVE_PATH="$RESULTS_BASE_DIR$RESULTS_SUBDIR$FILENAME"


python3 evaluation_tools/evaluation_processor.py \
    --action=$ACTION \
    --dataset_variant=$VARIANT \
    --dataset_subset_id=$SUBSET \
    --results_save_path=$RESULTS_SAVE_PATH \
    --time_step=$TIME_STEP \
    --num_data_workers=8 

