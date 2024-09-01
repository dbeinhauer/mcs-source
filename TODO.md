# Now
- add validation using correlation between samples
- generate data for test dataset (for correlation validation)
- add documentation to existing code
- add documentation for the repository structure
- fix evaluation step
- get validation data from Luca
- install the model to computational cluster
    - need to create docker image
- study multitrial validation
- check multiple trials in extraction tools (I should be sure it detect number of neurons and trials).
    - now we have the solution that for each trial creates file:
        `spikes_trial_{trial_id}_{layer}_{image_id}.npz`
- maybe spikes prefixes to some globals python file (to share between the sources)
- check multitrial for trimmer and merger
    - it should be ok with the new variant for multitrials
    - there should be only additional tool to concatenate multitrials to one array
    - in the trimmer I only changed the filename prefix (merger not changed)

# Future steps
- possibly change the RNNs with some LSTM (should be same for all neurons in the layer)
    - might be better than some small NN instead of simple neuron
    - need to share the weights (othewise it would not be sufficient to run it (too large model))


# Notes for the meeting
- assure LGN needs Inh/Exc weights
- change to float32 (loss is othervise `nan`)