# About

This repository contains source code for my master's thesis on 
the topic 'Modeling spatio-temporal dynamics in primary visual
cortex using deep neural network model'. It was done during my studies in
Bioinformatics study program in Faculty of Science in Charles University in 
Prague. The thesis is a part of the project from Computational Systems 
Neuroscience Group (CSNG) group from Faculty of Mathematics and Physics in 
Charles University in Prague.

# Repository structure
The structure of the repository is split into several Directories by their
functionalities. The main part of the code is in the Directory `nn_model/` 
where the source code that defines and runs the model is located. 

Main Components of the repository:
- `execute_model.py` - This file serves as an user interface for execution the model of V1.
- `nn_model/` - Main project directory. There is all implementation of the model of V1 located.
- `dataset_processor/` - Directory containing definition of all tools used for dataset processing and preparation.
- `evaluation_tools/` - Directory containing implementation of the evaluation processing tools that preprocesses raw model predictions or dataset for further analysis. There is also a default location where the evaluation results and best model parameters are stored.
- `results_analysis_tools/` - Directory containing tools for processing results generated from the `evaluation_tools/`. It entail for the exact statistical analysis and plotting of the results.
- `testing_dataset/` - Small example of the model dataset and model subset indices that has been used in our thesis. If one renames it to "dataset/", it should be in format in which the model can correctly load this toy example using default parameters without user modifications of the arguments. It may serve for testing correct installation of the model.
- `run_metacentrum_experiments.sh` - Interface script that entails model execution on Metacentrum server.
- `metacentrum_scripts/` - Directory necessary scripts to run model on the Metacentrum server.
- `environment.yaml` - File containing the environment setup necessary for running the model on the Metacentrum server. 
- `requirements_metacentrum.txt` - Requirements file used for correct installation of the environment on Metacentrum.
- `thesis_experiment_setups/` - Setup of all experiments that has been run in the thesis analysis.
- `pyproject.toml` - File for correct project definition while using poetry Poetry package manager.- `poetry.lock` - Poetry lock file for easier environment installation.
- `neural_simulation/` - Auxiliary directory for correct installation of the Poetry project.
- `requirements.txt` - Files listing the requirements for correct execution of the model.
- `run_model.sh` - Script used to execute model training as a background process on the CGG servers.
- `run_evaluation.sh` - Script used to execute model evaluation as a background process on the CGG servers.

# Installation
In order to run the model properly it is necessary to work on the machine with GPU available. 
Currently, it is not supported to run the project on CPUs as it also does not make sense in
regard to complexity of the network.

Apart from that current version also does not support model execution without the [Weights and 
Biases](https://wandb.ai/site/) account logged in.

## Metacentrum Installation
The recommended installation approach is 
installation on the [Metacentrum server](https://metavo.metacentrum.cz/). This computational 
cluster has been used more majority of our experiments and this repository contains several tools
facilitating model run on this server such as customized model execution using config files 
that also enables grid search analysis of the hyperparameters. More information can be found in 
proper files or directories described in the repository structure part.

## Local Installation
It is also possible to run model locally using [Poetry](https://python-poetry.org/) package 
manager. Once Poetry is installed successfully it should be sufficient to run the following 
commands from the root directory to install and activate the virtual environment:

```bash
poetry install
poetry add wandb
poetry shell
```

## CGG Servers Installation
The model has been also tested on the CGG MFF CUNI machines providing various GPU machines.
For more info please refer to the [documentation](https://docs.google.com/document/d/14CQrXu_OyqsMmzB67pJ4B1ALnhUaVJZZHYwrFupxEug/edit?tab=t.0#heading=h.z1jj4ypggaiz).

For these machines we have been using python version `python=3.8.9`. Other tested
versions has been problematic on these servers. To install the environment we use Conda 
(as it is recommended in the documentation for the CGG machines). It should be enough 
to run only the following commands to install the proper environment:

```bash
conda create --name neural_model python=3.8
conda activate neural_model
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Also note that there is a target cuda device defined in the file `execute_model.py`.
To change the cuda device one needs to change the following environment variable or 
set appropriate environment variable:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "{target_device}"
```

## Installation of the Mozaik
In order to run properly the scripts for dataset preparation located in `dataset_processor/` one
needs to install Mozaik project from CSNG MFF CUNI. For more information please see: [Mozaik](https://github.com/CSNG-MFF/mozaik)

IMPORTANT NOTE: If one would like to generate additional dataset data it is needed to do so 
using the Mozaik environment and on the Wintermute cluster of the Neuroscience Group (CSNG).


# Execution of the Model
To run the model (either for training or evaluation) it should be enough just to 
run the following command in the working environment:

```bash
python execute_model.py [all_additional_arguments]
```

To see all possible arguments run:

```bash
python execute_model.py --help
```

Optionally one can test the model runs properly even when they do not have dataset available
by renaming the `testing_dataset/` directory and running the model (either in debug or full 
format). The steps are the following:

```bash
mv testing_dataset/ dataset
python execute_model.py [--debug]
```

## Execution of the Model in Metacentrum
In order to run the model in Metacentrum cluster one would first need to properly setup the 
environment described in `metacentrum_scripts/`. After everything is properly setup then execution
of the jobs using config files is available. One can run multiple jobs using config file in a way:

```bash
python run_metacentrum_experiments.py {path_to_config_file}
```

Example of the config file can be found in `metacentrum_scripts/` or in `thesis_experiment_setups/`
directories. These config files also facilitates grid search running and different model variants
runs.

## Execution of the on CGG Machines

In case one would like to run the program using background job on CGG machines, it
is possible to do so while running the command:

```bash
./run_model.sh {required_arguments}
```

Or potentially run evaluation on CGG machine as background job as:
```bash
./run_evaluation.sh {required_arguments}
```

# Dataset and Other Required Files
In order to run the model properly there needs to be several files provided to execute
the model. Especially there needs to be proper paths to dataset defined 
(for more information about the dataset structure please inspect documentation in 
`dataset_processor/` directory). Ideally one would locate the dataset and model subset files
to the default paths to facilitate the whole workflow.

The required files are specified using the following arguments:
- `--train_dir` - The directory containing the train dataset.
- `--test_dir` - The directory containing the test dataset.
- `--subset_dir` - File containing the indices of the subset of each neuronal layer that are used in the model (as it is computational challenging to use all neurons).
- `--model_dir` - Directory where the best performing model parameters throughout 
the training (in terms of cc_norm metric) parameters are stored (for further 
evaluation).

The rest of the paths are used in optional tools of the model. In case you would
like to use these, please inspect the functionality closer in the source code.

## Dataset Location
Currently the dataset is stored in Wintermute cluster in location:
```bash
/home/beinhaud/diplomka/mcs-source/dataset
```

## Useful Arguments and Setup
Apart from the already mentioned path arguments there are several other parameters
of the model. The most important and less understandable are listed above:

- `--model` - Probably the most important argument. It specifies which type of 
shared neuron representation should be applied in the model. See the section `Model Types` for the comprehensive description of the different model types.
- `--debug` - This flag serves to run only a few time batches in both training and evaluation phasis (serves to debug the model correctness).
- `--best_model_evaluation` - Flag to run only evaluation on the selected model. There needs to be appropriate model stored parameters stored in `--model_dir` path.
- `--save_all_predictions` - Using this flag all model evaluation results of would be stored to appropriate file for further analysis.

Alongside with the model arguments there are few global setup 
parameters that are defined in `nn_model/globals.py` and needs to
be changed directly in the source code (as it is not expected to 
change them often). Those parameters are:

- `DEVICE` - On which device we would like to run the model.
- `SIZE_MULTIPLIER` - Subset of the model in terms of ratio of the whole number of neurons from each layer we want to use. This variable is also possible to change setting environment variable `SIZE_MULTIPLIER={selected_value}`. For example for value `0.1` we want to use 10% of all provided neurons. NOTE: There needs to be corresponding list of IDs of selected neurons provided to run correctly (see argument `--subset_dir`),
- `TIME_STEP` - Size of the time step interval used in the model in milliseconds. This variable is also possible to change setting environment variable `TIME_STEP={selected_value}`.NOTE: There needs to be corresponding dataset generated. In case required dataset is missing it is possible to generate it using tool `dataset_processor/time_merger/` (please see additional documentation there).
- `TRAIN_BATCH_SIZE`, `TEST_BATCH_SIZE` - Batch sizes (these are hardcoded as they are optimized for a given dataset and CGG machines). There are separate train and test batch sizes as test batch size is typically larger (the test dataset contains multiple trials) and it might be challenging to use same batch size as for train dataset.

For the rest of the variables from the file `nn_model/globals`, it is 
not expected to change their values unless we want to do some major 
modification in the program functionality.

# Evaluation Tools and Result Analysis Tools
In case one is interested in execution of the evaluation tool and results analysis tools
please refer to corresponding directories `evaluation_tools/` and `results_analysis_tools/`
where more detailed description is provided.

What is worth noting though is option for running only evaluation on the model parameters
and of the best performing model in terms of normalized CC and storing the evaluation predictions
to files for further analysis. For this it is necessary to set the
exactly same parameters as the best performing model and add the following two switched while
executing `execute_model.py`. The switches to include are:

```bash
--best_model_evaluation
--save_all_predictions
```

