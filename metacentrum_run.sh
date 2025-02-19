#!/bin/bash
#PBS -N VISUAL_CORTEX_MODEL
#PBS -l walltime=15:0:0
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=40gb:mem=100gb:scratch_local=100gb:spec=8.0:gpu_cap=compute_86:osfamily=debian

#PBS -m ae
#PBS -j oe

set -e

# Ensure clean_scratch runs on exit, even on error
cleanup() {
    echo "Running clean_scratch at $(date)"
    clean_scratch
}
trap cleanup EXIT

nvidia-smi

echo "Task started at $(date)"

PROJECT_NAME="mcs-source"
SERVER_LOCATION="praha1"
DATADIR="/storage/$SERVER_LOCATION/home/$USER/$PROJECT_NAME"
export TMPDIR=$SCRATCHDIR


test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }

# echo "Copying data to $SCRATCHDIR at $(date)"
# cp -r "$DATADIR" "$SCRATCHDIR" || { echo >&2 "Error copying data to scratch"; exit 1; }
# echo "Data copied at $(date)"

# cd "$SCRATCHDIR/$PROJECT_NAME" || { echo >&2 "Failed to enter scratch directory"; exit 1; }
cd "$DATADIR" || { echo >&2 "Failed to enter DATADIR"; exit 1; }

module load mambaforge

echo "Creating conda environment at $(date)"
mamba env create -p "$SCRATCHDIR/tmp_env" -f environment.yaml || { echo >&2 "Failed to create Conda environment"; exit 1; }
source activate "$SCRATCHDIR/tmp_env" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
echo "Environment created at $(date)"

wandb login "<your_api_key>" || { echo >&2 "Failed to log into wandb"; exit 1; }

echo "Logged in wandb at $(date)"

echo "Starting model execution at $(date)"

python execute_model.py --model="dnn_separate" --num_epochs=10 --neuron_num_layers=5 --num_data_workers=8  \
      	|| { echo >&2 "Python script failed"; exit 1; }

# TODO copy results

echo "Task finished at $(date)"

clean_scratch