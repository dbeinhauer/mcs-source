#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <output_dir> [wandb_api_key]"
    exit 1
fi

# Set Weights and Biases API Key if not specified
if [ "$#" -eq  1 ]; then
    echo "Running .wandb_api_key.sh because wandb_api_key wasn't specified."
    source ./.wandb_api_key.sh || exit 1
fi

# Directory to store the output log
OUTPUT_DIR=$1

# Set Parameters
SUBSET_ID=0

# Set wandb api key if specified
if [ "$#" -eq 5 ]; then
    echo "Setting wandb api key based on passed argument"
    API_KEY=$5
    export WANDB_API_KEY=$API_KEY
    wandb login
fi

# Command to execute, including the learning rate and number of epochs
COMMAND="python execute_model.py --num_data_workers=4 \
--learning_rate=0.00001 \
--model=dnn_joint \
--num_epochs=10 \
--num_backpropagation_time_steps=1 \
--neuron_num_layers=5 \
--neuron_layer_size=10 \
--neuron_rnn_variant=gru \
--synaptic_adaptation_size=10 \
--synaptic_adaptation_num_layers=1 \
--wandb_project_name=eval_dnn_joint \
--save_all_predictions \
--neuron_residual \
--subset_variant=$SUBSET_ID \
--best_model_evaluation"

# Print the command to be run
echo "Running command: $COMMAND"

# Run the command with nohup, but redirect output to a temporary log file
nohup $COMMAND >"$OUTPUT_DIR/output_temp.log" 2>&1 &

# Capture the process ID of the background job
PROCESS_ID=$!

# Rename the temporary log file to include the process ID
mv "$OUTPUT_DIR/output_temp.log" "$OUTPUT_DIR/output_$PROCESS_ID.log"

# Display the process ID and log file location
echo "Process ID: $!"
echo "Log file: $OUTPUT_DIR/output_$!.log"
