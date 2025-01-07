# TODO:
- probably wrong manipulation with neuron RNN model hidden states (we reset them each time step 
    - we probably want to let them (at least in evaluation)
    - we probably want them original in both training and evaluation


# Questions for meeting


# Notes from meeting
- separate RNN time steps to propagate across all time steps
- add signal modulation RNN after output of each neuron
    - it should be shared across the tuples input+output layers
    - it should be also added before LGN input is passed to the input layers
        - and after each output of the neurons too
    - this part should correspond to signal modulation when multiple spikes happen in short time period
        - the neurons diminish the signal in that case (needs to regenerate)
    - as output of the model we want to still have unmodulated signal (same as we have)
        - we just want to adjust the input signal to the other neurons
            - the change happens in real example in the output neuron 