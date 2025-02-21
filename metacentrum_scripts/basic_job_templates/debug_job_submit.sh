#!/bin/bash

# Input/output files:
TEMPLATE="metacentrum_scripts/job_template.pbs"
FILENAME="metacentrum_scripts/prepared_jobs/debug_job.sh"
# Machine setup:
WALLTIME="0:10:00"
NCPUS=4
NGPUS=1
GPU_MEM="5gb"
MEM="20gb"
SCRATCH_LOCAL="100gb"
# Optional machine arguments. For example:
# OPT_MACHINE_ARGS=":spec=8.0:gpu_cap=compute_86:osfamily=debian"

# Model parameters:
MODEL_PARAMS="--learning_rate=0.00001 \\
--num_epochs=2 \\
--model=dnn_separate \\
--neuron_num_layers=5 \\
--neuron_layer_size=10 \\
--neuron_residual \\
--synaptic_adaptation_size=10 \\
--synaptic_adaptation_time_steps=1 \\
--num_data_workers=8 \\
--debug "
# --synaptic_adaptation"

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
    --submit_job
# --use_opt_arguments \
# --opt_machine_args $OPT_MACHINE_ARGS \
