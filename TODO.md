# Now
- inspect different neuron models
    - different number of layers/layer sizes
- code refinement of the new features
    - evaluation on the subset of experiments and neurons
- adding additional hidden time steps between each target
    - the model might learn the dynamics better
- make plot of DNN responses

- it is very probably that there has been bug and I have been using residual as not residual 


# Future steps
- possibly change the RNNs with some LSTM (should be same for all neurons in the layer)
    - might be better than some small NN instead of simple neuron
    - need to share the weights (otherwise it would not be sufficient to run it (too large model))
- for the training with smaller numbers (float16)
    - it might be good idea to lower the training step
- try to find better measure
    - something for correlation of binary values
- tools for summarizing training results
- different evaluation metrics
- different losses
- try different models for neurons
- make plotting more convenient

# In longer time period:
- install the model to computational cluster
    - need to create docker image


# When is time:
- add documentation to existing code
- add documentation for the repository structure
- add globals for common paths and prefixes for extraction tools
- documentation for random subset selector
- create global variables for paths used in the model 
- probably separate model executer and argparse to separate source
- think about model_executor architecture
    - it might be useful to change it in order to make the source readable
- disable weights and biases for evaluation of best model
- make it possible to change all model parameters from `run_model.sh` script
    - some of the parameters might be optional
    - also change residual to optional
- improve response analysis 
    - add option to select neurons for analysis
    - add option to select images for analysis
- option for selecting the device (GPU) where we want to run the experiment


# Notes for the meeting
- look at the time back-propagation
    - how to it works in pytorch
    - additional step might be adding this between time steps
        - artificial time steps between the targets
            - model might learn more the dynamics
                - it is pretty complicated to learn the dynamics that is so sharp
- inspect what happens at the end
    - there should be slight increase at the end
    - same problem also with trained responses
        - predictions went downwards and targets upwards (strange)

