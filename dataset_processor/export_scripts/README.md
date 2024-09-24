# About
This directory contains scripts for extraction of the raw spikes data to the 
format used in the model. Namely it stores IDs of images and neurons used 
in the experiments. The main purpose of these scripts is to convert the spikes information 
into the format of multidimensional matrix specifiying when and in which
neuron the spike happened.

## Result dataset format
The scripts store the extracted data to the selected directory with 
predifined structure which is exploited in the future steps.
The structure is the following:
```bash
|- image_ids/
|- neuron_ids/
|- spikes/
    |_______
    |- X_ON/
    |- X_OFF/
    |- V1_Exc_L4/   
    |- V1_Inh_L4/
    |- V1_Exc_L23/
    |- V1_Inh_L23/ 
```

# Directory Content
- `export_all.sh` - script to submit Slurm jobs on Wintermute node for extracting whole dataset of given sheet stored in the provided path 
- `export_dataset.py` - main python script doing the exact extraction 
- `export_job.sh` - bash script to submit Slurm job on Wintermute node for extracting subset of dataset from given directory
- `export_missing.sh` - bash script used to additionaly extract the missing parts of dataset (in case regular extraction using script `export_all.sh` fails in some examples)  
- `export_parameters.md` - markdown file containing basic information about the raw dataset storage
