# About
This directory contains scripts and templates for creating
(and submitting) jobs on Metacentrum service.

# Directory Structure
The directory contains the following files:
- `generate_script.py` - Python script for generating Metacentrum PBS job definition for running the model with specified resources
- `job_template.pbs` - template of the bash script for running the PBS job running the model. It defines the environment and all necessary variables.
- `submit_job_template.sh` - OUTDATED!: Will be deleted. ~~template of the script that runs the `generate_script.py` for job definition and optionally submits the job~~
- `basic_job_templates/` - OUTDATED!: Will be deleted. ~~Directory containing few examples of the scripts for different experiment setups (inspired by `submit_job_template.sh`)~~
- `thesis_analysis/` - Directory containing the most updated version of script that runs the `generate_script.py` alongside with the template settings file that defines the experiment setup (and allows running grid search of experiments). The `job_template.sh` is expected to be executed by `run_experiments_from_settings_metacentrum.py`. 
- `singularity_environment/` - Directory that contains script that creates singularity environment which is up-to-date the best approach how to run the experiment. NOTE: That in order to create the singularity image one needs to have access to `builder.metacentrum.cz` (see: https://docs.metacentrum.cz/en/docs/software/containers#custom-singularity-build).

# Usage
IMPORTANT NOTE: In order to run the experiment correctly, it is expected to run the following commands (and submit jobs) from the root directory of this project.

To create (and/or submit) the job use either `generate_script.py`. For more information about the usage run:

```bash
python metacentrum_scripts/generate_script.py --help
```

There are then few advanced approaches. The best working and recommended one is to run the experiments through the command:

```bash
cd ..
python run_experiments_from_settings_metacentrum.py {custom_settings_file}.conf
```

This will create and submit multiple jobs on metacentrum based on 
the provided config file (example is 
`thesis_analysis/settings_template.conf`). Additionally, in root 
there is a directory `thesis_experiment_setups` where the specific 
configuration files for the experiments are defined.


~~Or you can use predefined examples (`basic_job_templates/`) or 
template of the bash script that creates the job script 
(`submit_job_template.sh`).~~

# Weights and Biases API Key
The model requires Weights and Biases API key to run properly. In 
case you are running the model using script above, you need to
store your API key to file named `.wandb_api_key` in root project
directory.

WARNING: Make sure to set the correct permissions of this file 
containing API key to assure no one except of you have the access to 
it.