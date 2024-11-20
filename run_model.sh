#!/bin/bash

# Set Weights and Biases API Key
source ./.wandb_api_key.sh

# Check if the correct number of arguments is passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <output_dir> <model> <learning_rate> <num_epochs>"
    exit 1
fi

# Function to display usage

# usage() {

#     echo "Usage: $0 -m mandatory_arg [-o optional_arg] [-t] [-f]"

#     echo "  -m mandatory_arg   Mandatory argument"

#     echo "  -o optional_arg    Optional argument"

#     echo "  -t                 True/False switch (on if present)"

#     echo "  -f                 Another True/False switch (on if present)"

#     exit 1

# }

# # Initialize variables

# MANDATORY_ARG=""

# OPTIONAL_ARG=""

# TRUE_SWITCH=0

# FALSE_SWITCH=0

# # Parse command line options

# while getopts ":m:o:tf" opt; do

#     case ${opt} in

#     m)

#         MANDATORY_ARG=$OPTARG

#         ;;

#     o)

#         OPTIONAL_ARG=$OPTARG

#         ;;

#     t)

#         TRUE_SWITCH=1

#         ;;

#     f)

#         FALSE_SWITCH=1

#         ;;

#     \?)

#         echo "Invalid option: -$OPTARG" 1>&2

#         usage

#         ;;

#     :)

#         echo "Option -$OPTARG requires an argument." 1>&2

#         usage

#         ;;

#     esac

# done

# # Shift processed options away

# shift $((OPTIND - 1))

# # Check for mandatory argument

# if [ -z "$MANDATORY_ARG" ]; then

#     echo "Mandatory argument is missing."

#     usage

# fi

# # Output the parsed arguments for demonstration purposes

# echo "Mandatory Argument: $MANDATORY_ARG"

# if [ -n "$OPTIONAL_ARG" ]; then

#     echo "Optional Argument: $OPTIONAL_ARG"

# fi

# if [ $TRUE_SWITCH -eq 1 ]; then

#     echo "True Switch: ON"

# fi

# if [ $FALSE_SWITCH -eq 1 ]; then

#     echo "False Switch: ON"

# fi

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
nohup $COMMAND >"$OUTPUT_DIR/output_temp.log" 2>&1 &

# Capture the process ID of the background job
PROCESS_ID=$!

# Rename the temporary log file to include the process ID
mv "$OUTPUT_DIR/output_temp.log" "$OUTPUT_DIR/output_$PROCESS_ID.log"

# Display the process ID and log file location
echo "Process ID: $!"
echo "Log file: $OUTPUT_DIR/output_$!.log"
