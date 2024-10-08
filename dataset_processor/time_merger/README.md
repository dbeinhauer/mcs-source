# About
This directory contains python script for merging spikes for given 
time interval. Additionally, there are bash scripts for running it 
on Wintermute server (CSNG).


## Running Jobs 
In order to run the Slurm job we need to connect to Wintermute server.
To run the time merger we need to execute the command:
```bash
./run_parallel_time_merger.sh <merged_interval_size> <dataset_variant>
```
Where `merged_interval_size` is the size of the interval we want to merge,
`dataset_variant` is either train or test (difference in multitrials).


# Directory Content
- `run_parallel_time_merger.sh` - script for merging all sheets on Wintermute
- `run_time_merger.sh` - script for merging specific sheet on Wintermute
- `time_interval_merger.py` - main python script for merging