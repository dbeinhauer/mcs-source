# About
This directory contains all parts of the source code used for model 
definition and execution.

# Directory Content
In the following lines there is a list of all source code files and a
brief description of their content:

- `custom_rnn.py` - Linear part of the RNN model layer.
- `dataset_loader.py` - Dataset class definition used in the model.
- `dictionary_handler.py` - Auxiliary functionalities used for complex operations with dictionaries.
- `evaluation_metrics.py` - Definition of evaluation metrics.
- `evaluation_results_saver.py` - Auxiliary functionalities for storing the evaluation results.
- `globals.py` - Global setup of the model (see `README.md` from root directory for more information).
- `layers.py` - Definition of RNN model layer with non-linearity application (either function of neuron module).
- `logger.py` - Logging of the model execution.
- `model_executer.py` - Main source code that controls model initialization, training, and evaluation.
- `models.py` - Definition of the whole model.
- `neurons.py` - Definition of the shared models of the neurons.
- `type_variants.py` - Definition of all the variants and types used in the model (as Enums).
- `weights_constraints.py` - Definition of weight constraints (excitatory/inhibitory neurons).

# Model description
The whole model is an RNN. The input of the network consist of the 
neuronal responses from the LGN layer (`X_ON`, `X_OFF`) of the current 
time step alongside with neuronal responses of all V1 layers from the 
previous time step. The information is propagated based on the model 
architecture. Each layer of the model represents different 
neuronal population (`V1_Exc_L4`, `V1_Inh_L4`, `V1_Exc_L23` and
`V1_Inh_L23`), and output of each layer represents part of the whole 
model output (the whole output sequence is an output). 

There is an optional possibility to replace the final non-linearity 
function of each RNN layer with the shared trainable DNN module that 
should represent activation function of the neurons from the 
corresponding population. In that case the outputs are the outputs of 
the neuron module.

## Layers
The model is separated into the 4 layers that represent different 
neuronal population. The connections between these layers are defined in 
`models.py`. Each layer is defined in `layers.py`.

Each layer consist of two parts linear RNN part that is defined in 
`custom_rnn.py` and final non-linearity neuronal module that is defined
in `neurons.py` (with exception to `simple` model that contains only
function tanh instead of the shared neuronal model).

### Architecture Structure
The layer setup is the following (Note: We also specify the appropriate
input time step (the current time is `t`)):

- `V1_Exc_L4`:
    - input:
        - current time step (`t`): `X_ON`, `X_OFF`
        - previous time (`(t-1)`): `V1_Exc_L4`, `V1_Inh_L4`, `V1_Exc_L23`
- `V1_Inh_L4`:
    - input:
        - current time step (`t`): `X_ON`, `X_OFF`
        - previous time (`(t-1)`): `V1_Exc_L4`, `V1_Inh_L4`, `V1_Exc_L23`
- `V1_Exc_L23`:
    - input:
        - current time step (`t`): `V1_Exc_L4`
        - previous time (`(t-1)`): `V1_Exc_L23`, `V1_Inh_L23`
- `V1_Inh_L23`:
    - input:
        - current time step (`t`): `V1_Exc_L4`
        - previous time (`(t-1)`): `V1_Exc_L23`, `V1_Inh_L23`

## Shared Neuron Modules
The shared modules of the neurons are defined in `neurons.py`. 
These neurons take inputs of size either `1` or `2` based on the return
type of one item from the linear RNN part. Their output is always of 
size 1 (one number) that should represent the neuronal response in the
corresponding time step of the given neuron.

### Types of the neuron modules
Currently, there are 2 variants of the neurons Feed-Forward DNN neurons
(marked as `dnn_`) and RNN neurons using cells (marked as `rnn_`). 
We also differentiate between different types of inputs of these 
models. In case the output of the preceding linear RNN part returns
joined output for inhibitory and excitatory layers (marked as `joint`)
the input is of size 1. Otherwise, the input size is 2 (one value for 
each excitatory and inhibitory layers). Such a model is marked as 
`separated`.

Overall there are currently 5 types of models:
- `simple` - Simple model without the neuron module (tanh non-linearity is applied after linear RNN step).
- `dnn_joint` - Feed-forward neuron is used as neuron module and linear part of the rnn outputs joint value for both excitatory and inhibitory input part.
- `dnn_separate` - Feed-forward neuron is used as neuron module and linear part of the rnn outputs separate values for both excitatory and inhibitory input part.
- `rnn_joint` - RNN neuron is used as neuron module and linear part of the rnn outputs joint value for both excitatory and inhibitory input part.
- `rnn_separate` - RNN neuron is used as neuron module and linear part of the rnn outputs separate value for both excitatory and inhibitory input part.

## Weight constraints
In our model we differentiate also between the excitatory and inhibitory
neurons. We do this using the constraints on the weights. For each
excitatory neurons we enforce that their weights are always non-negative
(it is clipped to 0 if this condition is violated). For inhibitory 
neurons we enforce the non-positivity of the weights (in case of 
violation of this condition they are clipped to 0).

# Dataset
The dataset is extracted from the raw data from the Mozaik model. There
is one significant difference between the training and testing dataset.
Testing dataset consists of multiple trials of one experiment 
(used to compute mean neuronal responses for evaluation). On the other 
hand, training dataset consist of only 1 trial (or more but does not 
need them).

## Dataset Structure
Because of the large size but sparseness of the data they are stored as
sparse Scipy matrices. Each layer data are stored in its own 
subdirectory each experiment in separate file. 
Those subdirectories are `X_ON/`, `X_OFF`, `V1_Exc_L4`, `V1_Inh_L4`,
`V1_Exc_L23` and `V1_Inh_L23`. 

These files are loaded at the point when they should be used in batch
(dataloader loads batch of subset of these files). It is done this way 
because it is not feasible to load all dataset at once. In case there 
is test dataset loaded there are all trial files loaded conjoinly.

## Experiment data format
Each experiment data (data from one file) are 2D tensors of shape 
`(num_neurons, num_time_steps)`. Since these data are loaded in batches,
the input size is typically `(batch_size, num_neurons, num_time_steps)`.
In case we do testing model evaluation there is additional trials 
dimension that is treated properly during the function execution.

## Model subsets
Since it is very computationally demanding work with the model that 
consist all neurons from our dataset, we use only subset of these neurons
for debugging, model refinement etc. These neurons need to be the same
in each experiment (to repeat the experiments with the same setup and
to perform the evaluation on the same subset of neurons). Subset of these
neurons are stored in the appropriate files with their indices specified
and are selected in each data loading. These subsets need to be provided
during each model execution (see `README.md` from root directory for 
more information).
