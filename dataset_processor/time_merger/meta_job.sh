#!/bin/bash
#PBS -N TIME_MERGER
#PBS -l walltime=15:0:0
#PBS -l select=1:ncpus=1:mem=24gb:scratch_local=100gb
#PBS -m abe
#PBS -j oe

PROJECT_NAME="mcs-source"
SERVER_LOCATION="praha1"
TIME_MERGER="/storage/$SERVER_LOCATION/home/$USER/$PROJECT_NAME/dataset_processor/time_merger/time_interval_merger.py"

echo "Task started at $(date)"

export TMPDIR=$SCRATCHDIR

test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }

python "$TIME_MERGER $1 $2 --time_interval=$3 --sheet=$4"

echo "Task finished at $(date)"