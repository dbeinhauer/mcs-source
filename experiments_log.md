# General Notes
- omitting `bee` cluster as the jobs fail there

# Simple Grid Search
- it seems that the best learning rate is `0.000075`
    - other comparable are `0.00001` and `0.000005`
    - with respect to high amount of experiments we will focus only on the learning rate `0.0000075`

# Simple evaluation

# DNN Grid Search
- based on

# RNN Grid Search
- selected multiple learning rates as we tested it before and the model proven to be better with higher learning rate with increasing number of backpropagation time steps
- need to run for 48 hours and pretty memory demanding
    - we decided to focus only backpropagation time steps and learning rate
    - rest taken from previous experiments as we expect the rnn layers and sizes kind of treat similar
        - less number of layers in order to save memory
        - GRU instead of LSTM - to save memory
            - LSTM is implemented though
                - if time allows, we can test it too
