# About
This directory contains python script for trimming time intervals to 
smaller chunks. Additionally, there are bash scripts for running it 
on Wintermute server (CSNG).


## Running Jobs 
In order to run the Slurm job we need to connect to Wintermute server.
To run the time trimmer we need to execute the command:
```bash
./run_parallel_time_trimmer.sh <dataset_variant>
```
Where `dataset_variant` is either train or test (difference in multitrials).


# Directory Content
- `run_parallel_time_trimmer.sh` - script for trimming all sheets on Wintermute
- `run_time_trimmer.sh` - script for trimming specific sheet on Wintermute
- `time_trimmer.py` - main python script for time trimming