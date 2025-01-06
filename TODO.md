# TODO:
- probably wrong manipulation with neuron RNN model hidden states (we reset them each time step 
    - we probably want to let them (at least in evaluation)
    - we probably want them original in both training and evaluation


# Questions for meeting
- loss calculation
    - is it better to calculate the loss separately for each layer or together concatenated?



# Notes from meeting
- the problematic part still stays the dynamics during stimulus 
    - in lately time steps there is not enough inhibition
        - we want ideally to be under stable spiking
- we want plots for different neurons
    - not only the population of them
    - to see there how it behaves on the neurons
- we want to plot the trained model only on blank stimuli
    - to see whether the spontaneous activity is ok
- train on smaller time steps
    - may better capture the dynamics
- train on larger model size
- add ReLU after DNN module (or other condition to have positive outputs)
    - we want the responses of the neuron to be spikes
- maybe make the DNN neuron module more complex to have some memory
    - create RNN of the DNN module
        - not sure if it works because there is already RNN (and memory) captured in the model architecture
