# Now
- think about model hyper-parameters
- try other bins (now trying on size 20)
- code refinement of the new features
    - evaluation on the subset of experiments and neurons
- loading the best model and running evaluation option
    - this connects to asking whether we want to save the best model
        - or/and move the trained models somewhere else in to ensure not overwriting
    - option only for analyzing selected experiments and neurons
- check the time of blank/stimulus for evaluation set


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