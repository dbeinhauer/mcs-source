#!/bin/bash
#PBS -N VISUAL_CORTEX_MODEL
#PBS -l walltime=8:0:0
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=40gb:mem=100gb:scratch_local=100gb

#PBS -j oe

set -e

# Ensure clean_scratch runs on exit, even on error
cleanup() {
	echo "Running clean_scratch at $(date)"
	clean_scratch
}
trap cleanup EXIT

echo "Task started at $(date)"

PROJECT_NAME="<insert root project name>"
SERVER_LOCATION="praha1"
DATADIR="/storage/$SERVER_LOCATION/home/$USER/$PROJECT_NAME"
export TMPDIR=$SCRATCHDIR


test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }

WANDB_API_KEY=$(<"$DATADIR/.wandb_api_key")
if [ -z "$WANDB_API_KEY" ]; then
	echo "wandb key is not set or empty"
	exit 1
fi

cd "$DATADIR" || { echo >&2 "Failed to enter DATADIR"; exit 1; }

module load mambaforge

echo "Creating conda environment at $(date)"
mamba env create -p "$SCRATCHDIR/tmp_env" -f environment.yaml || { echo >&2 "Failed to create Conda environment"; exit 1; }
source activate "$SCRATCHDIR/tmp_env" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
echo "Environment created at $(date)"

wandb login "$WANDB_API_KEY" || { echo >&2 "Failed to log into wandb"; exit 1; }

echo "Logged in wandb at $(date)"

echo "Starting model execution at $(date)"

python execute_model.py --model=rnn_separate --num_epochs=100 --num_data_workers=8 --num_backpropagation_time_steps=10 --learning_rate=0.000031 --neuron_num_layers=3 --neuron_residual --save_all_predictions --parameter_reduction

echo "Task finished at $(date)"

clean_scratch
