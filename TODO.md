# Now
- fix evaluation step
    - looks like problem is in V1_Inh_L23
- try to find better measure
    - something for correlation of binary values
- think about model hyperparameters
- inspect the training works fine (it looks like some the predicting the time series does not work fine) 
- tools for summarizing training results
- improve output of the training to facilitate follow-up work with it
- class for different metrics

# Future steps
- possibly change the RNNs with some LSTM (should be same for all neurons in the layer)
    - might be better than some small NN instead of simple neuron
    - need to share the weights (othewise it would not be sufficient to run it (too large model))
- for the training with smaller numbers (float16)
    - it might be good idea to lower the training step

# In longer time period:
- install the model to computational cluster
    - need to create docker image

# When is time:
- add documentation to existing code
- add documentation for the repository structure
- add globals for common paths and prefixes for extraction tools
- refine the model code
- documentation for random subset selector


# Notes for the meeting
- does it make sense to comput Pearson's CC for integers (or can predictions stay floats)