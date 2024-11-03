# Now
- think about model hyper-parameters
- figures + random neurons - plot responses
    - for the meeting to see what the neurons does in the response
    - plot these for each layer
        - probably do in form of jupyter notebook for now
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
- plots for:
    - histogram of number of spikes & number of neurons across the model
    - plot of time evolution of spiking per layer
    - plot of average neuron response per all images
        - also generally for population
    - select very active & not-active neurons and plot their responses together
        - add also the points where the blank/stimulus parts are
- look at the start and end of the activity (there should be drops down)
- the activity should go from low to peak at the beginning of the stimulus then drop down at the end
- add also classical correlation alongside with CC_NORM
- implement with `weights and biases` -> to better orient during training
