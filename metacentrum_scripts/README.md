# About
This directory contains scripts and templates for creating
(and submitting) jobs on Metacentrum service.

# Directory Structure
The directory contains the following files:
- `generate_script.py` - Python script for generating Metacentrum PBS job definition for running the model with specified resources
- `job_template.pbs` - template of the bash script for running the PBS job running the model. It defines the environment and all necessary variables.
- `submit_job_template.sh` - template of the script that runs the `generate_script.py` for job definition and optionally submits the job
- `basic_job_templates/` - directory containing few examples of the scripts for different experiment setups (inspired by `submit_job_template.sh`)

# Usage
IMPORTANT NOTE: In order to run the experiment correctly, it is expected to run the following commands (and submit jobs) from the root directory of this project.

To create (and/or submit) the job use either `generate_script.py`. For more information about the usage run:

```bash
python metacentrum_scripts/generate_script.py --help
```

Or you can use predefined examples (`basic_job_templates/`) or 
template of the bash script that creates the job script 
(`submit_job_template.sh`).

# Weights and Biases API Key
The model requires Weights and Biases API key to run properly. In 
case you are running the model using script above, you need to
store your API key to file named `.wandb_api_key` in root project
directory.

WARNING: Make sure to set the correct permissions of this file 
containing API key to assure no one except of you have the access to 
it.