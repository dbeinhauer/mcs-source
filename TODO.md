# Now
- split the training data to experiments and load them as batches
- reduce layer size
    - randomly selecting subset of neurons for training
- generate dataset for each size of interval size for trimmed dataset
- correct the model architecture
    - Inh should be only within layer
    - Exc should also be from L23 -> L4
- add validation using correlation between samples
- train/test split implementation
- generate data for test dataset (for correlation validation)

# Future steps
- possibly change the RNNs with some LSTM (should be same for all neurons in the layer)
    - might be better than some small NN instead of simple neuron
    - need to share the weights (othewise it would not be sufficient to run it (too large model))


# Notes for the meeting
- assure LGN needs Inh/Exc weights