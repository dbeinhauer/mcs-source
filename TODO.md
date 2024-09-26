# Now
- add validation using correlation between samples
- fix evaluation step
- study multitrial validation
- correct weights for LGN (should be only positive)
    - all LGN neurons are excitatory
- start the job for test dataset creation
    - after that:
        - trimming
        - merging
- check model can load multiple trials data for evaluation

# Future steps
- possibly change the RNNs with some LSTM (should be same for all neurons in the layer)
    - might be better than some small NN instead of simple neuron
    - need to share the weights (othewise it would not be sufficient to run it (too large model))

# In longer time period:
- install the model to computational cluster
    - need to create docker image

# When is time:
- add documentation to existing code
- add documentation for the repository structure
- add globals for common paths and prefixes for extraction tools 


# Notes for the meeting