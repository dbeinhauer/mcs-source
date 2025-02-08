#!/bin/bash
#PBS -N VISUAL_CORTEX_MODEL
#PBS -l walltime=10:0:0
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=39gb:mem=100gb:scratch_local=100gb
#PBS -m abe
#PBS -j oe

# This script should be run from the your home directory on a frontend server
# Fill these variables in order for the script to work
PROJECT_NAME="mcs-source"
SERVER_LOCATION="redacted" # for example "plzen1". Look for it in ($pwd)
USERNAME="redacted"
WANDB_API_KEY="redacted"

########################################################################################################################
echo "Task started at $(date)"

DATADIR="/storage/$SERVER_LOCATION/home/$USERNAME/$PROJECT_NAME"
export TMPDIR=$SCRATCHDIR

test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }

echo "Copying data to $SCRATCHDIR at $(date)"
cp -r "$DATADIR" "$SCRATCHDIR" || { echo >&2 "Error copying data to scratch"; exit 1; }
echo "Data copied at $(date)"

cd "$SCRATCHDIR/$PROJECT_NAME" || { echo >&2 "Failed to enter scratch directory"; exit 1; }

module load mambaforge

echo "Creating conda environment at $(date)"
mamba env create -p "$SCRATCHDIR/tmp_env" -f environment.yaml || { echo >&2 "Failed to create Conda environment"; exit 1; }
source activate "$SCRATCHDIR/tmp_env" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
echo "Environment created at $(date)"

wandb login $WANDB_API_KEY || { echo >&2 "Failed to log into wandb"; exit 1; }

echo "Logged in wandb at $(date)"

echo "Starting model execution at $(date)"

python execute_model.py --subset_dir="" --train_dir="testing_dataset/train/size_20" --test_dir="testing_dataset/test/size_20" --debug --num_epochs=1 || { echo >&2 "Python script failed"; exit 1; }

# TODO copy results

echo "Task finished at $(date)"