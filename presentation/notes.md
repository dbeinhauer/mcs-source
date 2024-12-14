# Modeling spatio-temporal dynamics in primary visual cortex using deep neural network model
David Beinhauer

Mgr. Ján Antolík, Ph.D.
(CSNG)

# Introduction to Computation Neuroscience
- to fully understand the purpose of this project we need to understand some related problems of the general computational neuroscience

## Computational Neuroscience Intro
- interdisciplinary field (neurobiology + computer science)
- to enhance understanding of the brain with help of computer science approach
    - models of neurons, brain parts, whole brain networks
- Why is it useful?
    - much more data than real life data
        - almost infinite possibility to enlarge dataset
        - we can study and look at much larger number of neurons than in real-life experiments (in real-life we are very limited)
    - possibility to understand the brain more, find new connection properties
    - thanks to these we can gradually improve our understanding of the brain and apply it to general neuroscience fields
    - example of possible usage:
        - neural prosthesis - e.g. for blind people in our CSNG group
        - overall if we understand the brain better -> better treatments overall

## Spiking models
- classical method in Computational neuroscience
- we tracked thousands of neurons -> we want to describe their function, interconnections, behavior using exact mathematical description
- we can describe them at various levels of complexity
    - e.g. neurons, smaller parts, larger parts etc.
- they work fairly well
    - e.g. model of cat visual cortex from CSNG
        - around 70 thousand neurons that matches the real-life neurons and its properties
        - quite high CC with the experimental data
        - we can retrieve almost limitless number of data from it
- Cons:
    - we need to know the properties of the neurons to model it well
        - we tracked them for approx. 70 years till it started to be technically feasible
    - we need to ideally model each of the neuron individually
    - it is computationally and time demanding
    -> there is not large possibility to apply this approach to large models (human has approx 2.000.000 neurons, we can model 70.000)

- What might be the improvement method? -> Using DNNs

## Deep Neural Networks
- the buzz word method nowadays
- bunch of thousand mathematical operations that are interconnected in the structured way to kind of "simulate" real life NN
- great in dealing with missing information
- it needs to have large training dataset
    - we do have it! - the data from the biologically plausible spiking model
- Cons:
    - Nowadays typical usage is just black box
        - usually its architecture is without any deeper motivation
        - we usually cannot infer useful information from it
    - especially Feed-Forward NN (no recurrent connections) -> partially solved using RNNs  
        - RNNs - temporal data (sequences), it has recurrent connections

- possible improvements -> define NN architecture using known properties of the studied (modeled) network -> Our Approach!


# The studied case

## Primary Visual Cortex
- visual information: eye -> LGN -> primary visual cortex (V1) -> other higher cortices
- LGN - in hippocampus (evolutionary old)
    - first information modulations (light/dark...)
- V1 - first information processing is happening there
- we want to focus on modeling of the V1 part with inputs from LGN

### Architecture of V1
- layered structure
- each layer has its inhibitory (lowering) and excitatory (enhancers) sub-layers
- we just want to focus on L4 and L2/3 (those we have in our spiking model)

# Our approach
- we take the data from the spiking network -> convert them to just spike rates in small time interval bins (currently 20ms)
    - data consist of stimulus -> following blank period (there we can learn the spontaneous stimuli)
- model architecture to emulate real-life architecture (inhibitory/excitatory, layers interconnected the way as in real life)
    - we also try to train the shared small DNN model of the neuron instead of some non-linearity in the model
        - the model can learn the typical responses of the neurons in different layers

# Results
- quickly explain the evaluation metrics
- Pearson's CC + CC_NORM (Pearson's CC whose range we diminish based on the data noisiness)
    - the cc_norm should be then better spread across the values (0; 1)
- best model currently 0.91 cc_norm (cc_abs 0.75)
    - better using DNN than simple (a bit but indeed it is)
- show plots of the mean neural responses to demonstrate the model performance

# Possible next steps?
- after the model quality is reasonably high -> try to partially hide the visible parameters
    - simulation of the real-life example
    - after this is successful -> try to apply it on the real-life macaque data
- compare neuronal parameters to spiking network (it might be good metric how to know that we can spread our model using DNNs)
- apply local connectivity constraints (the probability that the neurons are connected lowers with the distance)