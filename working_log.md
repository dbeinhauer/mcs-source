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
- mozaik should be used only for data extraction
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


# 26.3.2024
- the subdirecories in the Luca's dataset are due the memory managment (too large data)
- each segment is 1 experiment (1 image)
- I am also interested in blank parts
- trials are not interesting (they are here in case we have experiments with multiple trials)
- basicaly what I want is to take the times of each spike train in each neuron
    - need to manage the order (both experiments and neurons)
- I want to create from the data multidimensional matrix
    - neurons matrix architecture:
        experiment * neurons * time
            - also, somehow deal with neuron populations
    - images matrix architecture:
        images * time
- I would need to somehow deal with long chunks of information
    - I also want to interconnect both previous and next blank images after stimuli
- there should be also information about how the neurons are interconnected
    - we would like to use it later in the work
    - now, we are not interested in this
        

# 28.3.2024
- stimuli should be in ordered way using command:
    `segs_stimuli = dsv2.get_segments(ordered=True)`
- blank image stimuli:
    `segs_blank = dsv2.get_segments(null=True,ordered=True)`
- in separe folders there are different experiments (the data are not interconnected between the directories)
- I have created functions to separate stimuli to blank and images
- I have created function to sort segment neurons

# 5.5.2024
- created dataset export
- I have np.arrays of shape:
    `(num_imgs, num_neurons, time_duration)`
    * each `blank` and `image` has separate array (durations 151 and 561)
- Need to iterate through all directories and all sheets
    - should be 500 iterations of the function
- Does not consider the images (asked Luca if needed)
- Also storing neuron IDs and Image Ids into separate np.arrays


# 10.5.2024
- using sparse representation to drastically reduce size 
    - from 1.96 GB -> 0.27 MB
    - need to convert from 3D matrix -> 2D
        - hopefully not very time consuming (both convertion to sparse and dimensions)
- assuming we have only 1 trial (got rid of the checking trials and iterating through them)
- We probably do not need to extract images because Luca has provided them to us (hopefully in correct format)
- We do not sort the neurons (we assume all of them are in the correct order and the same)
- We assume that all experiments last the same time (150ms blank -> 560ms image)
- we would probably want using 10ms time windows


# 30.5.2024
- input data from the LGN layer -> output - rest of layers
- we should use 10-20ms time windows (for better results and faster training)
- maybe good idea to somehow spread the digital spike data to surroundings (maybe some normal distribution around the spike?)


# 28.6.2024
- exporting dataset on wintermute
- trying to create the first prototype of the model
- I do need neuron IDs


# 30.6.2024
- dataset finalized
- problem with the size of the dataset and work with it
    - possible memmory problems to have all the dataset in the RAM
- solution would be:
    - prepare the dataset for the given time interval size
    - cache the preprocessed dataset with the interval size
        - to not do it repeteadely
        - might not be a good idea (too large dataset)
            - in my oppinion in sparse representation it would be ok
- dataset extraction finished
- problematic work with the size of 1 data (too large -> won't fit into local memory)
    - lets try it on metacentrum
    - approx. 24 GB needed for 1 data
- working on first version of model
    - loads each batch from the directory
        - otherwise won't fit into memory
    - we do not need any modification of dataset
    - just need to create acumulated time steps (maybe some caching before run)


# 1.7.2024
- trying work with time steps 10 ms
    - succesfully started training -> killed because of memory
        - possibly might work on metacentrum?


# 2.7.2024
- creating scripts for dataset trimming
    - for execution on wintermute
- now it is possible to trim the dataset to arbitrary window size
- for timesize 20 it is possible to run the model on local


# 25.7.2024
- we want to train the model with architecture:
    - input (LGN) -> L4 (with residual connections to itself) -> L23 (with residual connections to itself)
- we do know all the expected values of the model neurons
- inhibitory vs. excitatory
    - just clip the values to 0 (if positive/negative)


# 26.7.2024
- hopefully working architecture using LGN -> L4 -> L23


# 27.7.2024
- used architecture that uses `nn.RNN` with hidden neurons corresponsing to number of neurons in each layer is too large
- our architecture:
    - RNNs for each layer (without LGN (it is only input))
        - RNN layer:
            input: input size (previous layer size)
            hidden (output): num_neurons in the layer
            interlayer connections: in RNN module definition
            inhibitory/excitatory: constrain weights to non-positive/negative
- problem:
    - too large model it does not fit in the GPU memory
        - too much weights
        -> we need to design smaller model


# 28.7.2024
- try using `nn.RNNCell` instead
    - better for custom model definition
    - we need to define passing sequence information
    - we can better look at how the information is processed through the network
    - same architecture as before:
        - too large -> need to do the model smaller
        - possible solution:
            - usage of FC layers from RNN with less hidden neurons
                - FC created outputs for each layer that will be passed further
- variant 1:
    - FC only for generating outputs
    - problem with memory alocation
        - problematic part in weights constrains (for Inh/Exc) -> too much memory consuming
        - after that solved -> still problems with large time intervals
            - it would be probably necessary to trim the time intervals to smaller chunks
            - we may try to optimize the memory usage in GPUs


# 29.7.2024
- memory optimization
    - it is able to run the training on `size_5` dataset
- iteration per example:
    ~45s
- approximate iteration per epoch:
    ~6 hours and 30 minutes


# 30.7.2024
- Notes from the consultation:
- I should write an email to try to get the access to the GPUs with large memory (computing clusters)
    - different from CGI group that has only regular GPUs

## Possibilities how to make the model smaller:
- converting the tensors from `float32` to `float16`
    - disadvantage: possible loss of information (not severe)
- splitting the dataset samples based on the image
    - needed overlap of the sequences or cut it in the middle of the blank sequence
    - it would significantly reduce the output size
        -  the sequences would be only 1 experiment (not 100)
    - disadvantage: there might be loss of context (not significant)
        - the majority of the effect of the image should disapear in during the blank sequences (because of that we want to split it in the middle of blank)
- reducing the layer size
    - randomly select subset of neurons for the training the model
    - it should have the largest impact on the size of the model
    - it should help us in definition using only RNNs (without FC)
    - try to play with the size till we reach the point where we can use RNNs
    - disadantages:
        - significant loss of information (lack of neurons)
- I should definitely implement variant of using multiple GPUs

## Questions regarding the model
- the number of epochs should not be probably large
    - because of the number of sequences
- to validate that the model is correct
    - compute the correlation of the model output and targets
    - if the correlation is above 0.5 it might be the good model
- also implement the Inh/Exc weights for LGN input layer 
    - forward connections should always have appropriate weight constraints


# 6.8.2024
- shared memory - works for small GPUs
- using half precision - possible only in forward step
    - optimizer needs to work with float32
- reducing number of timesteps to 900 (should be 1 example)


# 7.8.2024
- using the GPU with 48 GB memory (largest possible)
    - after optimizations I am able to run 
- the model is tested the way that all possible time step sizes will fit (even the 1 ms timestep)
- we would use only one experiment for data
    - second part of blank + image + first part of next blank
        - first example also have first part of the blank
        - last example have first part of next blank missing
- finished time_trimmer -> I will have each example in separate file

# 9.8.2024
- consultation notes:
- need to change the architecture
    - inhibitory should be only in its layer
    - excitatory should be also from L23 -> L4 (both Inh, Exc)
    - not sure about LGN (how are the Inh, Exc weights?)
- for validation:
    - generate data in multiple trials (for correlation calculation)

# 18.8.2024
- changed the architecture
- changed the weights definition
- created functions for train/test split and model subset generation
    - for generating the indices (for repeated usage)
- train/test split done
- model subset selection done

# 27.8.2024
- obtained access to computing cluster
- need to create docker image in order to run it on the cluster