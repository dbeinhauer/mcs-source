#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <output_dir> <model> <learning_rate> <num_epochs> [wandb_api_key]"
    exit 1
fi

# Set Weights and Biases API Key if not specified
if [ "$#" -eq 4 ]; then
    echo "Running .wandb_api_key.sh because wandb_api_key wasn't specified."
    source ./.wandb_api_key.sh || exit 1
fi

# Directory to store the output log
OUTPUT_DIR=$1
# Assign command-line arguments to variables
MODEL=$2
LEARNING_RATE=$3
NUM_EPOCHS=$4

SUBSET_ID=3

# Set wandb api key if specified
if [ "$#" -eq 5 ]; then
    echo "Setting wandb api key based on passed argument"
    API_KEY=$5
    export WANDB_API_KEY=$API_KEY
    wandb login
fi

# Command to execute, including the learning rate and number of epochs
COMMAND="python execute_model.py \
--model=$MODEL \
--learning_rate=$LEARNING_RATE \
--num_epochs=$NUM_EPOCHS"

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
