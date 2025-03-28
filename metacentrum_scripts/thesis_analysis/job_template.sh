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


# Default values for model parameters
LEARNING_RATE=0.00001
MODEL="dnn_separate"
NUM_EPOCHS=10
SUBSET_VARIANT=-1
NUM_BACKPROP_STEPS=1
NEURON_NUM_LAYERS=5
NEURON_LAYER_SIZE=10
NEURON_RESIDUAL=false
NEURON_RNN_VARIANT="gru"
SYN_ADAPT_USE=false
SYN_ADAPT_NUM_LAYERS=1
SYN_ADAPT_SIZE=10
SYN_ADAPT_LGN=false
TRAIN_SUBSET=-1
WANDB_NAME=""


# Parse long options
while [[ $# -gt 0 ]]; do
    case $1 in
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --subset_variant)
            SUBSET_VARIANT="$2"
            shift 2
            ;;
        --num_backprop_steps)
            NUM_BACKPROP_STEPS="$2"
            shift 2
            ;;
        --neuron_num_layers)
            NEURON_NUM_LAYERS="$2"
            shift 2
            ;;
        --neuron_layer_size)
            NEURON_LAYER_SIZE="$2"
            shift 2
            ;;
        --neuron_residual)
            NEURON_RESIDUAL=true  # Boolean flag
            shift 1
            ;;
        --neuron_rnn_variant)
            NEURON_RNN_VARIANT="$2"
            shift 2
            ;;
        --syn_adapt_use)
            SYN_ADAPT_USE=true  # Boolean flag
            shift 1
            ;;
        --syn_adapt_num_layers)
            SYN_ADAPT_NUM_LAYERS="$2"
            shift 2
            ;;
        --syn_adapt_size)
            SYN_ADAPT_SIZE="$2"
            shift 2
            ;;
        --syn_adapt_lgn)
            SYN_ADAPT_LGN=true  # Boolean flag
            shift 1
            ;;
        --train_subset)
            TRAIN_SUBSET="$2"
            shift 2
            ;;
        --wandb_name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --wandb_name value [--learning_rate value] [--model value] [--num_epochs value] [--subset_variant value] [--num_backprop_steps value] [--neuron_num_layers value] [--neuron_layer_size value] [--neuron_residual] [--neuron_rnn_variant value] [--syn_adapt_use] [--syn_adapt_num_layers value] [--syn_adapt_size value] [--syn_adapt_lgn] [--train_subset value]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure WANDB_NAME is provided
if [ -z "$WANDB_NAME" ]; then
    echo "Error: --wandb_name is required."
    exit 1
fi



# Model parameters:
MODEL_PARAMS="--num_data_workers=8 \\
--learning_rate=$LEARNING_RATE \\
--model=$MODEL \\
--num_epochs=$NUM_EPOCHS \\
--num_backpropagation_time_steps=$NUM_BACKPROP_STEPS \\
--neuron_num_layers=$NEURON_NUM_LAYERS \\
--neuron_layer_size=$NEURON_LAYER_SIZE \\
--neuron_rnn_variant=$NEURON_RNN_VARIANT \\
--synaptic_adaptation_size=$SYN_ADAPT_SIZE \\
--synaptic_adaptation_num_layers=$SYN_ADAPT_NUM_LAYERS \\
--wandb_project_name=$WANDB_NAME \\
--save_all_predictions"

# Add boolean flags to MODEL_PARAMS if they are set to true
for flag in "NEURON_RESIDUAL:--neuron_residual" "SYN_ADAPT_USE:--synaptic_adaptation" "SYN_ADAPT_LGN:--synaptic_adaptation_only_lgn"; do
    VAR_NAME="${flag%%:*}"  # Extract variable name
    FLAG_NAME="${flag##*:}" # Extract flag name
    if [ "${!VAR_NAME}" = true ]; then
        MODEL_PARAMS="$MODEL_PARAMS \\
$FLAG_NAME"
    fi
done

# Add --subset_variant to MODEL_PARAMS if SUBSET_VARIANT is not -1
if [ "$SUBSET_VARIANT" -ne -1 ]; then
    MODEL_PARAMS="$MODEL_PARAMS \\
--subset_variant=$SUBSET_VARIANT"
fi

# Add --subset_variant to MODEL_PARAMS if SUBSET_VARIANT is not -1
if [ "$TRAIN_SUBSET" -ne -1 ]; then
    MODEL_PARAMS="$MODEL_PARAMS \\
--train_subset=$TRAIN_SUBSET"
fi

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
