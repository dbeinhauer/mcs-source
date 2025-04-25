# About
This directory contains small sample of both train and test dataset alongside with 
all model subset indices per each variant that has been used during experiments
in our thesis.

NOTE: In case one renames this directory to "dataset/" the model should load it
without problems if other parameters, mainly model size and time step size, are set 
properly. One thus can test the model functionality even locally without the access 
to full dataset using this toy example.

# Directory Structure
This directory contains:
- `train_dataset/` - small example of the train dataset of time step 20
- `test_dataset/` - small example of the test dataset of time step 20
- `model_subsets/` - all model subset variants indices used throughout our thesis analyses and development
- `neuron_ids/` - Mapping of neuron IDs from raw SNN model responses dataset to our model (in our model the index is basically index of the array).