#!/bin/bash

#SBATCH --job-name=full_dataset_analysis
#SBATCH --output=output_dir/output_%j.txt   
#SBATCH --ntasks=8  # Requesting n processors
#SBATCH --nodes=1

#SBATCH --exclude=w[1-2,9-12]



ACTION="prediction_analysis"
MODEL="dnn_joint_evaluation"
FILENAME="$ACTION-$MODEL.pkl"

RESULTS_BASE_DIR="/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/"
RESULTS_SUBDIR="evaluation_analysis/"
RESULTS_SAVE_PATH="$RESULTS_BASE_DIR$RESULTS_SUBDIR$FILENAME"


python3 evaluation_tools/evaluation_processor.py \
    --action=$ACTION \
    --model_evaluation_variant=$MODEL \
    --results_save_path=$RESULTS_SAVE_PATH

