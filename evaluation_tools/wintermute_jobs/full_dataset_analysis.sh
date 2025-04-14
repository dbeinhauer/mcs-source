#!/bin/bash

#SBATCH --job-name=histogram_creation
#SBATCH --output=output_dir/output_%j.txt   
#SBATCH --ntasks=8  # Requesting n processors
#SBATCH --nodes=1

#SBATCH --exclude=w[1-2,9-12]

TIME_STEP=20

VARIANT="train"
# VARIANT="test"

ACTION="full_dataset"
FILENAME="$ACTION-$VARIANT-$TIME_STEP.pkl"

RESULTS_BASE_DIR="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/"
RESULTS_SUBDIR="full_dataset_analysis/"
RESULTS_SAVE_PATH="$RESULTS_BASE_DIR$RESULTS_SUBDIR$FILENAME"


python3 evaluation_tools/evaluation_processor.py \
    --action=$ACTION \
    --dataset_variant=$VARIANT \
    --results_save_path=$RESULTS_SAVE_PATH \
    --time_step=$TIME_STEP \
    --num_data_workers=8 
