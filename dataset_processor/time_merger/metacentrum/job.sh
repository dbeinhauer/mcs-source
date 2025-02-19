#!/bin/bash
#PBS -N TIME_MERGER
#PBS -l walltime=15:0:0
#PBS -l select=1:ncpus=1:mem=24gb:scratch_local=30gb
#PBS -j oe

set -e

SERVER_LOCATION="praha1"
PROJECT_DIR="/storage/$SERVER_LOCATION/home/$USER/mcs-source"
TIME_MERGER_DIR="$PROJECT_DIR/dataset_processor/time_merger"
TIME_MERGER="$TIME_MERGER_DIR/time_interval_merger.py"

echo "Task started at $(date)"

export TMPDIR=$SCRATCHDIR
test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }
cd "$SCRATCHDIR" || { echo >&2 "Failed to enter scratch directory"; exit 1; }

module load mambaforge
echo "Activating conda environment at $(date)"
source activate "$TIME_MERGER_DIR/metacentrum/tmp_env" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
echo "Running time merger at $(date)"

python "$TIME_MERGER" "$VAR_IN" "$VAR_OUT" --time_interval="$VAR_SIZE" --sheet="$VAR_SHEET" || { echo >&2 "Time merger failed"; exit 1; }

echo "Task finished at $(date)"