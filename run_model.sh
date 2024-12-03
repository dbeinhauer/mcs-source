#!/bin/bash

# Set Weights and Biases API Key
source ./.wandb_api_key.sh

# Check if the correct number of arguments is passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <output_dir> <model> <learning_rate> <num_epochs>"
    exit 1
fi

# Directory to store the output log
OUTPUT_DIR=$1
# Assign command-line arguments to variables
MODEL=$2
LEARNING_RATE=$3
NUM_EPOCHS=$4

# usage() {
#     echo "Usage: $0 output_dir model_type learning_rate num_epochs [-]"
#     echo "  --required          Multiple required arguments"
#     echo "  --optional          Multiple optional arguments"
#     echo "  --verbose           Verbose output"
#     echo "  -d, --dry-run       Dry run, won't execute commands"
#     exit 1
# }

# # Default values
# VERBOSE=false
# DRY_RUN=false
# REQUIRED_ARGS=()
# OPTIONAL_ARGS=()

# # Parse command line arguments

# while [ $# -gt 0 ]; do
#     case "$1" in
#     --required)
#         shift
#         while [[ $# -gt 0 && "$1" != --* ]]; do
#             REQUIRED_ARGS+=("$1")
#             shift
#         done
#         ;;
#     --optional)
#         shift
#         while [[ $# -gt 0 && "$1" != --* ]]; do
#             OPTIONAL_ARGS+=("$1")
#             shift
#         done
#         ;;
#     --verbose)
#         VERBOSE=true
#         shift
#         ;;
#     -d | --dry-run)
#         DRY_RUN=true
#         shift
#         ;;
#     -*)
#         echo "Error: Invalid option '$1'"
#         usage
#         ;;
#     *)
#         echo "Error: Invalid argument '$1'"
#         usage
#         ;;
#     esac
# done

# # Check if required arguments are set

# if [ ${#REQUIRED_ARGS[@]} -eq 0 ]; then
#     echo "Error: At least one required argument is missing."
#     usage
# fi

# # Display the parsed arguments

# echo "Required Arguments: ${REQUIRED_ARGS[*]}"

# if [ ${#OPTIONAL_ARGS[@]} -gt 0 ]; then
#     echo "Optional Arguments: ${OPTIONAL_ARGS[*]}"
# fi

# echo "Verbose: $VERBOSE"
# echo "Dry Run: $DRY_RUN"

# # Logic to execute based on the flags and arguments

# if [ "$VERBOSE" = true ]; then
#     echo "Running in verbose mode..."
# fi

# if [ "$DRY_RUN" = true ]; then
#     echo "This is a dry run, commands will not be executed."
# else
#     echo "Executing commands..."
#     # Place your command execution logic here
# fi

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
