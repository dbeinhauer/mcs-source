#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_variant>"
    exit 1
fi

MODEL="$1"

# Path to the job script template (will be created dynamically)
JOB_SCRIPT="job_evaluation_${MODEL}.sh"

cat << EOF > "$JOB_SCRIPT"
#!/bin/bash
#PBS -N EVALUATION_ANALYSIS_${MODEL}
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=8:mem=100gb:scratch_local=100gb
#PBS -m ae
#PBS -j oe

set -euo pipefail

PROJECT_NAME="mcs-source"
SERVER_LOCATION="praha1"
DATADIR="/storage/\$SERVER_LOCATION/home/\$USER/\$PROJECT_NAME"
SIF_IMAGE="/storage/\$SERVER_LOCATION/home/\$USER/visual_cortex.sif"

MODEL="${MODEL}"
FILENAME="prediction_analysis-\$MODEL.pkl"
RESULTS_SAVE_DIR="\$DATADIR/analysis_results/prediction_analysis/"
RESULTS_SAVE_PATH="\$RESULTS_SAVE_DIR\$FILENAME"

echo "Task started at \$(date)"

export TMPDIR=\$SCRATCHDIR
test -n "\$SCRATCHDIR" || {
  echo >&2 "SCRATCHDIR is not set!"
  exit 1
}

cd "\$SCRATCHDIR" || { echo "Failed to enter scratch directory" >&2; exit 1; }

mkdir -p "\$RESULTS_SAVE_DIR"

cat << EOI > run_inside.sh
#!/bin/bash
export PATH=/opt/conda/bin:\$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate metacentrum_env

echo "Starting evaluation analysis at \$(date)"

python /mnt/evaluation_tools/evaluation_processor.py \\
    --action=prediction_analysis \\
    --model_evaluation_variant="\$MODEL" \\
    --results_save_path="\$RESULTS_SAVE_PATH" || {
    echo >&2 "Python script failed"
    exit 1
}
EOI

chmod +x run_inside.sh

echo "Running inside Singularity at \$(date)"
singularity exec --nv -B "\$DATADIR":/mnt "\$SIF_IMAGE" bash run_inside.sh || {
    echo >&2 "Singularity execution failed"
    exit 1
}

echo "Task finished at \$(date)"
clean_scratch
EOF

# Make the job script executable
chmod +x "$JOB_SCRIPT"

# Submit it
qsub "$JOB_SCRIPT"
