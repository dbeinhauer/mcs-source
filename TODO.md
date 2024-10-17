# Now
- think about model hyper-parameters
- different losses
- training: only one step (not sequence) - weights to real data (target)
- start on the first time step (not with the zeros)
- figures + random neurons - plot responses
    - for the meeting to see what the neurons does in the response
- neuron NN 
    - smaller size of the layer
    - higher number of layers
    - input should be one neuron -> output one neuron (not the whole population)
- larger bins
    - bin of size 10 and 20


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


# In longer time period:
- install the model to computational cluster
    - need to create docker image

# When is time:
- add documentation to existing code
- add documentation for the repository structure
- add globals for common paths and prefixes for extraction tools
- documentation for random subset selector


# Notes for the meeting
