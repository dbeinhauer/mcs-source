# Now
- split the training data to experiments and load them as batches
- add validation using correlation between samples
- generate data for test dataset (for correlation validation)
- add documentation to existing code
- add documentation for the repository structure
- look at loss (is still `nan`)
- fix evaluation step
- get validation data from Luca
- install the model to computational cluster
    - need to create docker image
- get access to bioinfo partition
- find out how to trasnfer training data from wintermute to gpulab

# Future steps
- possibly change the RNNs with some LSTM (should be same for all neurons in the layer)
    - might be better than some small NN instead of simple neuron
    - need to share the weights (othewise it would not be sufficient to run it (too large model))


# Notes for the meeting
- assure LGN needs Inh/Exc weights