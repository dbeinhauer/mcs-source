# Now
- think about model hyper-parameters
- figures + random neurons - plot responses
    - for the meeting to see what the neurons does in the response
    - select random neurons
    - select random images
    - select best CC results
    - plot these for each layer
- neuron NN 
    - smaller size of the layer
    - higher number of layers
- try other bins (now trying on size 20)
- saving the best model based on the CC metric


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
