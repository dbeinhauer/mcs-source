#!/bin/bash

# Input/output files:
TEMPLATE="metacentrum_scripts/job_template.pbs"
FILENAME="metacentrum_scripts/prepared_jobs/job_script.sh"
# Machine setup:
WALLTIME="<hours>:<minutes>:<seconds>"
NCPUS=<ncpus>
NGPUS=<ngpus>
GPU_MEM="<gpu_mem>gb"
MEM="<ram_mem>gb"
SCRATCH_LOCAL="<scratch_mem>gb"
# Optional machine arguments.
# For example: ":spec=8.0:gpu_cap=compute_86:osfamily=debian"
OPT_MACHINE_ARGS=<opt_machine_args>

# Model parameters:
MODEL_PARAMS="--learning_rate=0.00001 \\
--num_epochs=10 \\
--neuron_residual"

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
    --opt_machine_args $OPT_MACHINE_ARGS \
    --model_params "$MODEL_PARAMS"
# --submit_job
