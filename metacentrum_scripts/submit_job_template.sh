#!/bin/bash

# Read the API key from the .wandb_api_key file
# WANDB_API_KEY=$(cat .wandb_api_key)

# Define default parameters
TEMPLATE="metacentrum_scripts/job_template.pbs"
FILENAME="metacentrum_scripts/job_script.sh"
WALLTIME="15:0:0"
NCPUS=8
NGPUS=1
GPU_MEM="40gb"
MEM="100gb"
SCRATCH_LOCAL="100gb"
SPEC="8.0"
GPU_CAP="compute_86"
OPT_MACHINE_ARGS=":spec=8.0:gpu_cap=compute_86:osfamily=debian"
# DEFAULT_PARAMS_PATH="default_params.json"
MODEL_PARAMS="--learning_rate=0.00001 --num_epochs=10 --neuron_residual"

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
    --spec $SPEC \
    --gpu_cap $GPU_CAP \
    --opt_machine_args $OPT_MACHINE_ARGS \
    --model_params "$MODEL_PARAMS" # --wandb_api_key $WANDB_API_KEY \
# --submit_job
