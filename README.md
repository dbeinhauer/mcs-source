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
- `dataset_processor/` - Directory containing definition of all tools used for dataset processing and preparation.
- `evaluation_tools/` - Directory containing definition of all tools used for evaluation of the model results.
- `neural_simulation/` - Auxiliary directory for correct installation of the Poetry project.
- `nn_model/` - Directory containing the major part of the source code. It contains the definition of the model and code for its execution. There is also description of the model architecture and dataset structure.
- `execute_model.py` - The source code used for execution of the model (training/evaluation).
- `requirements.txt` - Requirements file to execute the model.
- `run_model.sh` - Script used to execute the model training as a background process on the CGG servers.
- `metacentrum_scripts/` - directory containing useful scripts and templates for the Metacentrum experiment computing
- `thesis_experiment_setups/` - list of experiment setups for experiments of the master thesis
- `subset_generator.py` - script that generates model subsets variants
- `metacentrum_run_all_subsets_job.sh` - script that runs grid search on metacentrum

# Run Metacentrum Grid Search
First, we need to have grid search settings file defined. 
Example of such settings file can be found in 
`metacentrum_scripts/thesis_analysis/settings_template.conf`

We typically store these config files in directory:
`thesis_experiment_setups/`

To run the metacentrum grid search just run command:
```bash
python run_experiments_from_settings_metacentrum.py {experiment_config_file_path}
```

# Installation

## Model and Evaluation Tools
Since now, we have run the model only on the CGG machines (more info: 
https://docs.google.com/document/d/14CQrXu_OyqsMmzB67pJ4B1ALnhUaVJZZHYwrFupxEug/edit?tab=t.0#heading=h.z1jj4ypggaiz).
The installation steps are relevant only for those machines, and it is not assured 
they will work on other machines. Also note that the model has been executed only on
the machines with GPUs (as it almost does not any sense to run it on CPUs). In case
anyone would like to install the model on other machine, additional steps in the 
installation procedure might be needed.

Currently, we run the project using python version `python=3.8.9`. Newer python 
versions should work too (not tested though).

We run the model using the Conda environment (as it is recommended in the 
documentation for the CGG machines). It should be enough to run only the following
commands to install the proper environment:

```bash
conda create --name neural_model python=3.8
conda activate neural_model
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Also note that there is a target cuda device defined in the file `execute_model.py`.
To change the cuda device one needs to change the following environment variable:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "{target_device}"
```

In the repository, there is also `requirements.txt` file that specifies all 
requirements to run the model (and evaluation tools) correctly. This file might be 
helpful in case there are problems with the installation.

## Dataset Processor
IMPORTANT NOTE: If one would like to generate additional dataset data it is needed 
to do so using the Mozaik environment 
(more info: https://github.com/CSNG-MFF/mozaik) and on the Wintermute cluster 
of the Neuroscience Group (CSNG).

# Executing Model
To run the model (either for training or evaluation) it should be enough just to 
run the following command in the working environment:

```bash
python execute_model.py [all_additional_arguments]
```

To see all possible arguments run:

```bash
python execute_model.py --help
```

In case one would like to run the program using background job on CGG machines, it
is possible to do so while running the command:

```bash
./run_model.sh {required_arguments}
```

## Required Files
Alongside with the source code there needs to be several files provided to execute
the model. Especially there needs to be proper paths to dataset defined (for more information about the dataset structure please inspect documentation in `dataset_processor/` directory). If you run the model on CGG machine with already generated setups, it 
should not be problem using the default paths. Otherwise, the appropriate files
should be provided.

The required files are specified using the following arguments:
- `--train_dir` - The directory containing the train dataset.
- `--test_dir` - The directory containing the test dataset.
- `--subset_dir` - File containing the indices of the subset of each neuronal layer that are used in the model (as it is computational challenging to use all neurons).
- `--model_dir` - Directory where the best performing model parameters throughout 
the training (in terms of cc_norm metric) parameters are stored (for further 
evaluation).

The rest of the paths are used in optional tools of the model. In case you would
like to use these, please inspect the functionality closer in the source code.

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
- `SIZE_MULTIPLIER` - Subset of the model in terms of ratio of the whole number of neurons from each layer we want to use. For example for value `0.1` we want to use 10% of all provided neurons. NOTE: There needs to be corresponding list of IDs of selected neurons provided to run correctly (see argument `--subset_dir`) 
- `TIME_STEP` - Size of the time step interval used in the model in milliseconds. NOTE: There needs to be corresponding dataset generated. In case required dataset is missing it is possible to generate it using tool `dataset_processor/time_merger/` (please see additional documentation there).
- `TRAIN_BATCH_SIZE`, `TEST_BATCH_SIZE` - Batch sizes (these are hardcoded as they are optimized for a given dataset and CGG machines). There are separate train and test batch sizes as test batch size is typically larger (the test dataset contains multiple trials) and it might be challenging to use same batch size as for train dataset.

For the rest of the variables from the file `nn_model/globals`, it is 
not expected to change their values unless we want to do some major 
modification in the program functionality.

## Weights and Biases tool
During training it is also possible to use `Weights and Biases` tool 
to track the training procedure and model performance. To use such
tool one needs to create an account in for this tool and properly 
log in. For further information please check the source code 
(look specifically in the `wand` parts).

# Dataset Processing and Evaluation Tools
In case you are interested in dataset processing including the 
description of the dataset structure please see documentation in 
directory `dataset_processor/`. For all kinds of evaluation tools 
please inspect the `evaluation_tools/` directory.

## Running evaluation predictions
To plot the evaluation results you need to first run the full
evaluation and store its predictions for future analysis. 
It can be done by running the model with the appropriate 
arguments and two additional switches:

```bash
--best_model_evaluation
--save_all_predictions
```

To plot the results you can inspect the implementation of the
current version in the `evaluation_tools/` directory. To
have a notion of the usage of these tools please have a look 
at `evaluation_tools/response_analysis.ipynb` Jupyter notebook,
where a few examples are located.
