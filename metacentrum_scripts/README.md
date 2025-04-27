# About
This directory contains scripts and templates for creating and submitting jobs on Metacentrum service.

# Usage
Main interface script for running and preparing the metacentrum jobs is located in the root directory 
in the file `run_metacentrum_experiments.py`. For further understanding of the Metacentrum functionality 
please refer to documentation of this script.

# Directory Structure
The directory contains the following:
- `run_generate_script.sh` - script that serves as first entrypoint of main `run_metacentrum_experiments.py` it runs proper job generation and submit in script `generate_script.py`
generate_script.py  job_template.pbs  prepared_jobs singularity_enviroment
- `generate_script.py` - Python script for generating Metacentrum PBS job definition (or submit) for running the model with specified parameters. It generates the job based on template `job_template.pbs`.
- `job_template.pbs` - Template of the bash script for running the PBS job running the model. It defines the environment and all necessary variables.
- `singularity_environment/` - Directory that contains script that creates singularity environment which is up-to-date the best approach how to run the model. NOTE: That in order to create the singularity image one needs to have access to `builder.metacentrum.cz` (see: https://docs.metacentrum.cz/en/docs/software/containers#custom-singularity-build).
- `settings_template.conf` - Example of configuration file used for metacentrum job execution with the specific model setup.

# Usage
IMPORTANT NOTE: In order to run the experiment correctly, it is expected have singularity image properly
set and installed on the Metacentrum (see `singularity_environment/` and `job_template.pbs` to properly
understand the Metacentrum job workflow). 

There are several approaches how to run the model on Metacentrum. The ideal one and customized also
for advanced grid search and multiple model execution using setup files is the following (including moving to the base repository directory):

```bash
cd ..
python run_metacentrum_experiments.py {custom_settings_file}.conf
```

The script above starts the metacentrum job based on the provided experiment configuration file. 
Example of such configuration file can be seen either in this directory in file 
`settings_template.conf` or in the directory `thesis_experiment_setups/` of the repository where
the configurations for all experiments executed during the analysis of the thesis results are stored.

## Weights and Biases API Key
IMPORTANT NOTE: The current implementation of the project model does not allow running the model
without Weights and Biases account specified. This means that when correct Weights and Biases key
is not specified the jobs will not be executed correctly.

The model requires Weights and Biases API key to run properly. In case one want to run the 
predefined metacentrum scripts they need to store their API key to file named 
`.wandb_api_key` in root directory of this project.

SECURITY WARNING: Make sure to set the correct permissions of this file 
containing API key to assure no one except of you have the access to 
it.