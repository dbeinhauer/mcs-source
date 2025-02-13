#!/bin/bash
#PBS -N PREPARE_ENV
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=1:mem=24gb:scratch_local=100gb
#PBS -j oe

set -e

SERVER_LOCATION="praha1"
PROJECT_DIR="/storage/$SERVER_LOCATION/home/$USER/mcs-source"
TIME_MERGER_DIR="$PROJECT_DIR/dataset_processor/time_merger/metacentrum"
ENV_DIR="$TIME_MERGER_DIR/tmp_env"


echo "Task started at $(date)"

export TMPDIR=$SCRATCHDIR
test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }
cd "$SCRATCHDIR" || { echo >&2 "Failed to enter scratch directory"; exit 1; }

module load mambaforge
echo "Creating conda environment at $(date)"
rm -fr "$ENV_DIR"
mkdir -p "$ENV_DIR"
mamba env create -p "$ENV_DIR" -f "$TIME_MERGER_DIR/environment.yaml" || { echo >&2 "Failed to create Conda environment"; exit 1; }
source activate "$ENV_DIR" || { echo >&2 "Failed to activate Conda environment"; exit 1; }
echo "Environment created at $(date)"

echo "Task finished at $(date)"