# Title
Simulating neural inputs from the 
Jan Antolik (CNSG)

# Intro
- computational neuroscience - combines math and computer science to better understand neural system
- recently more data -> usage of DNN models to simulate the function of the neural system
- the knowledge can be used e.g. to develop better visual prosthetis

## Early visual system
- light goes through eye to retina -> converts to neural signal -> goes through specialized
neurons to LGN (hypothalamus) -> V1
- LGN - first preprocessing of the data
    - input from eye + also significant part from V1 (signal modulation)

### Primary Visual Cortex (V1)
- divided into layers (we are interested in IV and II/III)
- majority from LGN -> IV (also reccurent from IV) -> II/III
- several other connection (intra-layer), from other layer and areas
- architecture much more complex (different cell types (naive/complex, excitatory/inhibitory), etc.)

## Neural Networks (Deep NN)
- type of machine learning models
- train data used to train (modification) of the model parameters to perform the best results
- architecture:
    1. input layer
    2. hidden layers
    3. output layer
- output of neuron:
    y_j = f(\Sigma_{i} x_i w_{ij})
- Deep NN - uses high number of hidden layers (it is deep)

## Recurrent DNN
- uses also recurrent connections (not just forward)
    - connections inside the layer as well as to the previous layers

## State-of-the-art models
- typical current approach is to use feed-forward NN
- discrete spikes (spike rate in given time interval)
    - spike is excited while signal reach some treshold
- train data:
    images + neural responses to the image in each layer (just spikes)
- response -> spikes


# Aims
- desing model exploiting the "continuous" input infomation of the neural response
    - not number of spikes but rather values of voltage in the neuron through given time interval
    - more information -> more constrained -> better corresponds to real-life -> possibility of better performance
- possible aims:
    1. design model using continuous input which results are comparable to typical discrete approach (on train data from the cat model simulator developed by the CNSG)
    2. modify the model architecture to better correspond the real-life architecture
        - add layers (LGN, IV, II/III), differenciate excitatory/inhibitory, simple/complex cells
    3. design model corresponding to experimental results (add more hidden neurons (not detected one))
    4. Final results (not done as the master thesis) -> incorporate the model into the existing cat model from CNSG

# Methods
- using RNN (continuous information -> the model needs context)
- train data -> from cat's primary visual cortex (LGN, IV, II/III)
    - images + neural activity of each neuron through the experiment
    - basically we know all values from the neurons
        - train it gradually (reveal just around 10% of values and train the rest let hidden)
- after successfully trained model -> try to incorporate the real architecture into it
- final approach would be to try to fit also the real data (not jsut the one from the simulator)

# Results
- we will compare the results of the model with the classical approach model (we have the one trained on exactly the same data)