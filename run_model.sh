#!/bin/bash

# Set Weights and Biases API Key
source ./.wandb_api_key.sh

# Check if the correct number of arguments is passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <output_dir> <model> <learning_rate> <num_epochs>"
    exit 1
fi

# Assign command-line arguments to variables
MODEL=$2
LEARNING_RATE=$3
NUM_EPOCHS=$4

# Directory to store the output log
OUTPUT_DIR=$1

# Command to execute, including the learning rate and number of epochs
COMMAND="python execute_model.py --model=$MODEL --learning_rate=$LEARNING_RATE --num_epochs=$NUM_EPOCHS"

# Print the command to be run
echo "Running command: $COMMAND"


# Run the command with nohup, but redirect output to a temporary log file
nohup $COMMAND > "$OUTPUT_DIR/output_temp.log" 2>&1 &

# Capture the process ID of the background job
PROCESS_ID=$!

# Rename the temporary log file to include the process ID
mv "$OUTPUT_DIR/output_temp.log" "$OUTPUT_DIR/output_$PROCESS_ID.log"

# Display the process ID and log file location
echo "Process ID: $!"
echo "Log file: $OUTPUT_DIR/output_$!.log"
