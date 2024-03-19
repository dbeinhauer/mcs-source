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



# 12.2.2024
- sheet is the layer
- github1s - VSCOde github

- mozaik experiments - how to convert image to spikes
- the last 2 folders with 500 have the experiments (one for l2/3 the other for l4 and LGN)


# 19.3.2024
- the mozaik should be used only for data extraction
- data consist of:
    - the image
    - experiment lenght (and blank/stimuli parts)
    - spike trains for neurons in LGN, L4 and L3/2
        - we want to split them into smaller bins
        - each bin has number of spikes inside the bin (typical case might be just changing 0/1)

## Architecture of our RNN
- step 1: LGN (input) x rest of the neurons (output layer)
    - completely dismiss the images (just use results from LGN)
    - neurons inside the output layer has fully recurrent and inter/intra connections
        - might be too many parameters
            - solution would be to get rid of long range connections and connect only neurons close to each other
    - layers based on the architecture + inhibitory/excitatory layers
        - should correspond to its function (inhibitory should have negative? outputs)

- step 2: Add NN instead of the neurons
    - probably convolutionary (just one NN for all neurons inside the layer)
        - the layers should probably have different NN (slightly different function)
    - because the neurons are in reality much complicated than just some linear (or non-linear function)

- step 3: Simulate real-life situation
    - try to hide neurons and train the model (simulation of the real data)
    - should decide how to hide the neurons - probably choose as much uniform distribution as possible
        - if we let only the neurons close to each other (from one region) it would probably result in loss of significant amount of information
    - maybe just fine-tune the pre-trained model (decide how to set the rigit neurons etc.)