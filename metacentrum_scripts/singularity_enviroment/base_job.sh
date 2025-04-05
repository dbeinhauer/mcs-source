#!/bin/bash
#PBS -N VISUAL_CORTEX_MODEL
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=8:ngpus=1

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
SIF_IMAGE="/storage/$SERVER_LOCATION/home/$USER/visual_cortex.sif"  # or .sandbox

export TMPDIR=$SCRATCHDIR

test -n "$SCRATCHDIR" || {
    echo >&2 "SCRATCHDIR is not set!"
    exit 1
}

cd "$DATADIR" || {
    echo >&2 "Failed to enter DATADIR"
    exit 1
}

# Module (optional if singularity already in path)
# module load singularity

# Copy image to scratch (optional, improves performance)
cd "$SCRATCHDIR"

# Create the runner script for inside container
cat << 'EOF' > run_inside.sh
#!/bin/bash
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate metacentrum_env

WANDB_API_KEY=$(cat /mnt/.wandb_api_key)
wandb login "$WANDB_API_KEY"

python /mnt/execute_model.py --debug
EOF

chmod +x run_inside.sh

echo "Running inside Singularity at $(date)"

# Bind your project directory as /mnt inside container
singularity exec -nv -B "$DATADIR":/mnt "$SIF_IMAGE" bash run_inside.sh || {
    echo >&2 "Python script failed"
    exit 1
}

clean_scratch

echo "Task finished at $(date)"
