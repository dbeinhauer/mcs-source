# About
This directory contains script that should define the workflow how to create singularity image
that should serve for execution of the model on Metacentrum.

NOTE: In order to create the singularity image one needs to have elevated builder rights on Metacentrum
for more information please refer to Metacentrum documentation.

# Directory Content
This directory contains:

- `base_job.sh` - Testing Metacentrum job to check whether the singularity image has been installed correctly and model execution is possible inside of it.
- `build_singularity_script.sh` - This script should define the workflow necessary to properly install the singularity image on metacentrum to successfully run model inside of it and to run all the Metacentrum jobs.