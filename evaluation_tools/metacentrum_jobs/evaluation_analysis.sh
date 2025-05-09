#!/bin/bash
#PBS -N EVALUATION_ANALYSIS
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=24gb:scratch_local=30gb

#PBS -m ae
#PBS -j oe

set -euo pipefail

# === CONFIGURATION ===
PROJECT_NAME="mcs-source"
SERVER_LOCATION="praha1"
DATADIR="/storage/$SERVER_LOCATION/home/$USER/$PROJECT_NAME"
EVALUATION_TOOLS_DIR="$DATADIR/evaluation_tools"
SIF_IMAGE="/storage/$SERVER_LOCATION/home/$USER/visual_cortex.sif"  # or .sandbox

# === ARGUMENT VALIDATION ===
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_variant>"
    exit 1
fi

MODEL="$1"
RESULTS_SAVE_PATH="$EVALUATION_TOOLS_DIR/evaluation_results/evaluation_analysis/$MODEL"

# === PREPARATION ===
echo "Task started at $(date)"

export TMPDIR=$SCRATCHDIR
test -n "$SCRATCHDIR" || {
echo >&2 "SCRATCHDIR is not set!"
exit 1
}

cd "$SCRATCHDIR" || { echo "Failed to enter scratch directory" >&2; exit 1; }

# Ensure results directory exists
mkdir -p "$RESULTS_SAVE_PATH"

# Create the runner script for inside container
cat << EOF > run_inside.sh
#!/bin/bash
export PATH=/opt/conda/bin:\$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate metacentrum_env

echo "Starting evaluation analysis at \$(date)"

python /mnt/evaluation_tools/evaluation_processor.py \\
    --action=prediction_analysis \\
    --model_evaluation_variant="$MODEL" \\
    --results_save_path="/mnt/evaluation_tools/evaluation_results/evaluation_analysis/$MODEL" || {
    echo >&2 "Python script failed"
    exit 1
}
EOF

chmod +x run_inside.sh

echo "Running inside Singularity at $(date)"

# Bind your project directory as /mnt inside container
singularity exec --nv -B "$DATADIR":/mnt "$SIF_IMAGE" bash run_inside.sh || {
    echo >&2 "Singularity execution failed"
    exit 1
}

# === CLEANUP ===
echo "Task finished at $(date)"
clean_scratch
