#!/bin/bash

# Input/output files:
TEMPLATE="metacentrum_scripts/job_template.pbs"
FILENAME="metacentrum_scripts/prepared_jobs/base_job.sh"
# Machine setup:
WALLTIME="24:00:00"
NCPUS=4
NGPUS=1
GPU_MEM="40gb"
MEM="100gb"
SCRATCH_LOCAL="100gb"
# Optional machine arguments. For example:
# OPT_MACHINE_ARGS=":spec=8.0:gpu_cap=compute_86:osfamily=debian"

# Parse arguments - subset variant if we work with multiple of them.
SUBSET_VARIANT=${1:--1} # Default to -1 if no argument is provided

# Model parameters:
MODEL_PARAMS="--num_data_workers=8 \\
--learning_rate=0.00001 \\
--model=dnn_separate \\
--num_epochs=10 \\
--save_all_predictions"

# Add --subset_variant to MODEL_PARAMS if SUBSET_VARIANT is not -1
if [ "$SUBSET_VARIANT" -ne -1 ]; then
    MODEL_PARAMS="$MODEL_PARAMS \\
--subset_variant=$SUBSET_VARIANT"
fi

script_dir="${BASH_SOURCE[0]%/*}"  # Get script's directory
WANDB_NAME="${script_dir##*/}"       # Extract last directory

MODEL_PARAMS="$MODEL_PARAMS \\
--wandb_project_name=$WANDB_NAME"

# Run the generate_script.py with the specified parameters and submit the job
python metacentrum_scripts/generate_script.py \
    --template $TEMPLATE \
    --filename $FILENAME \
    --walltime $WALLTIME \
    --ncpus $NCPUS \
    --ngpus $NGPUS \
    --gpu_mem $GPU_MEM \
    --mem $MEM \
    --scratch_local $SCRATCH_LOCAL \
    --model_params "$MODEL_PARAMS" \
    # --submit_job
# --use_opt_arguments \
# --opt_machine_args $OPT_MACHINE_ARGS \
