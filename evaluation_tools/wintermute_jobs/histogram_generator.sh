#!/bin/bash

#SBATCH --job-name=histogram_creation
#SBATCH --output=output_dir/output_%j.txt   
#SBATCH --ntasks=8  # Requesting n processors
#SBATCH --nodes=1

#SBATCH --exclude=w[1-2,9-12]

TIME_STEP=20

VARIANT="train"
# VARIANT="test"
ACTION="histogram_$VARIANT"
# ACTION="histogram_test"
RESULTS_SAVE_DIR="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/histograms/"
FILENAME="$ACTION-$TIME_STEP.pkl"

python3 evaluation_tools/response_analyzer.py \
    --action=$ACTION \
    --results_save_path="$RESULTS_SAVE_DIR$FILENAME" \
    --time_step=$TIME_STEP \
    --num_data_workers=8 
