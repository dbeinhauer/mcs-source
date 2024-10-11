# Now
- try to find better measure
    - something for correlation of binary values
- think about model hyperparameters
- tools for summarizing training results
- different evaluation metrics
- different losses

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
- documentation for random subset selector


# Notes for the meeting
- does it make sense to compute Pearson's CC for integers (or can predictions stay floats)
- how to improve the model training
    - why the learning is so inconsistent