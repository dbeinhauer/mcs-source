#!/bin/bash

set -e

[ -z "$1" ] && { echo "Error: Missing interval size"; exit 1; }

ROOT="$HOME/mcs-source"
SCRIPT="$ROOT/dataset_processor/time_merger/metacentrum/job.sh"
SIZE=$1
VARIANTS=("train_dataset" "test_dataset")
SHEETS=("V1_Exc_L23" "V1_Exc_L4" "V1_Inh_L23" "V1_Inh_L4" "X_OFF" "X_ON")

for VARIANT in "${VARIANTS[@]}"; do
    DIR="$ROOT/dataset/$VARIANT"
    IN="$DIR/trimmed_spikes"
    OUT="$DIR/compressed_spikes/trimmed/size_$SIZE"

    [ -d "$IN" ] || { echo "Error: $IN missing"; continue; }
    mkdir -p "$OUT"

    for SHEET in "${SHEETS[@]}"; do
        qsub -v VAR_IN="$IN",VAR_OUT="$OUT",VAR_SIZE="$SIZE",VAR_SHEET="$SHEET" "$SCRIPT"
    done
done