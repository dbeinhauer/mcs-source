# Now
- add validation using correlation between samples
- fix evaluation step
    - looks like problem is in V1_Inh_
- study multitrial validation

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


# Notes for the meeting