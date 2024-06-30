#!/bin/bash

# Script to run all export jobs for the given pattern.

for i in $(seq 50 100); do
    # Form the string with the current number
    filename="NewDataset_Images_from_${i}*"
    
    sbatch export_job.sh $filename
done