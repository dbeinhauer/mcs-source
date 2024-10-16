# About
This directory contains scripts for extraction of the raw spikes data to the 
format used in the model. Namely it stores IDs of images and neurons used 
in the experiments. The main purpose of these scripts is to convert the spikes information 
into the format of multidimensional matrix specifying when and in which
neuron the spike happened.


## Running Jobs 
To export the dataset we need to connect to Wintermute (CSNG) server.
To export all dataset of specific layer we need to run the command:
```bash
./export_all.sh <sheet> <dataset_variant>
```
This script runs multiple jobs on wintermute cluster that export the dataset
The parameter `<sheet>` specifies which sheet we want to export, the `dataset_variant`
specifies whether we want to export train or test dataset (in test dataset we 
export multiple trials).


If some part of the dataset exported wrongly (typically problem with Wintermute
job), we can run the following script to determine missing part of the dataset
and export only those (if we do not want to export the whole layer again). We
run it as follows:
```bash
./export_missing.sh <sheet> <dataset_variant>
```
The parameters are the same as they are in the `export_all.sh`.


## Result dataset format
The scripts store the extracted data to the selected directory with 
predefined structure which is exploited in the future steps.
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
- `export_missing.sh` - bash script used to additionally extract the missing parts of dataset (in case regular extraction using script `export_all.sh` fails in some examples)  
- `export_parameters.md` - markdown file containing basic information about the raw dataset storage
