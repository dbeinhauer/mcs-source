# 18.1.2024
- theme of the thesis
- analysis of the model based on the continuous data from the signaling spikes (e.g. 10ms)
    - different from classical approach
        * just discrete spikes
        * just feed-forward network (our approach would be use RNN)
- it would be neccessary to use RNN
    - train on the data from the simulator 
        - these I should get from Luca at the end of the next week
        - anotated data together with "data parser"
            - I would just need to rewrite it to continuous data (instead of the spikes)
        - data consist of neural spikes from the LGN, V4 and V2/3 (both inhibitory and excitatory layers)
        
- first step:
    - design RNN that would work with the given data
    - it depends what types of neurons should we use
        - in reality none of the neurons from the simulator is hidden
        - ideally uncover just 10% of neurons and try to train the model the way it appropriately predicts the "hidden neurons"
            - good baseline for the real data (in them we do not see the all the neurons (there are hidden ones))
    - probably the good idea would be to design the model based on the layers (learn how!)
    - the model has to be more complex than the one for the spikes
        - although the continuous information also adds amount of info in the train data
            * it is possible that the resulting model would be more constained than the easier one -> better performance 
            (might be the goal of the whole thesis, also good result for the good academic paper)

- final result (not mine) of the project is to merge it together with the simulator
