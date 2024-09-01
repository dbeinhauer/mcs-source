# About
This directory contains all the tools used for dataset preparation and modification.

# Directory Content
- `copy_subset.sh` - script for copying subset of the whole dataset (it is usually used only for testing the dataset preparation tools)
- `dataset_renamer.py` - script for renaming the spikes dataset to format used in the model
- `export_scripts/` - directory containing tools for exporting the dataset from raw data that are store in wintermute cluster (from CSNG)
- `time_merger/` - directory containing tools for merging the spikes dataset to larger time intervals
- `time_trimmer/` - directory containing tools for trimming the spikes dataset to smaller time intervals (usually to separate experiments) 