# 18.1.2024
- theme of the thesis
- analysis of the model based on the continuous data from the signaling spikes (e.g. 10ms)
    - different from classical approach
        * just discrete spikes
        * just feed-forward network (our approach would be use RNN)
- it would be necessary to use RNN
    - train on the data from the simulator 
        - these I should get from Luca at the end of the next week
        - annotated data together with "data parser"
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
            * it is possible that the resulting model would be more constrained than the easier one -> better performance 
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
    - probably convolution (just one NN for all neurons inside the layer)
        - the layers should probably have different NN (slightly different function)
    - because the neurons are in reality much complicated than just some linear (or non-linear function)

- step 3: Simulate real-life situation
    - try to hide neurons and train the model (simulation of the real data)
    - should decide how to hide the neurons - probably choose as much uniform distribution as possible
        - if we let only the neurons close to each other (from one region) it would probably result in loss of significant amount of information
    - maybe just fine-tune the pre-trained model (decide how to set the rigit neurons etc.)


# 26.3.2024
- the sub-directories in the Luca's dataset are due the memory management (too large data)
- each segment is 1 experiment (1 image)
- I am also interested in blank parts
- trials are not interesting (they are here in case we have experiments with multiple trials)
- basically what I want is to take the times of each spike train in each neuron
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
    - possible memory problems to have all the dataset in the RAM
- solution would be:
    - prepare the dataset for the given time interval size
    - cache the preprocessed dataset with the interval size
        - to not do it repeatedly
        - might not be a good idea (too large dataset)
            - in my opinion in sparse representation it would be ok
- dataset extraction finished
- problematic work with the size of 1 data (too large -> won't fit into local memory)
    - lets try it on metacentrum
    - approx. 24 GB needed for 1 data
- working on first version of model
    - loads each batch from the directory
        - otherwise won't fit into memory
    - we do not need any modification of dataset
    - just need to create accumulated time steps (maybe some caching before run)


# 1.7.2024
- trying work with time steps 10 ms
    - successfully started training -> killed because of memory
        - possibly might work on metacentrum?


# 2.7.2024
- creating scripts for dataset trimming
    - for execution on wintermute
- now it is possible to trim the dataset to arbitrary window size
- for time-size 20 it is possible to run the model on local


# 25.7.2024
- we want to train the model with architecture:
    - input (LGN) -> L4 (with residual connections to itself) -> L23 (with residual connections to itself)
- we do know all the expected values of the model neurons
- inhibitory vs. excitatory
    - just clip the values to 0 (if positive/negative)


# 26.7.2024
- hopefully working architecture using LGN -> L4 -> L23


# 27.7.2024
- used architecture that uses `nn.RNN` with hidden neurons corresponding to number of neurons in each layer is too large
- our architecture:
    - RNNs for each layer (without LGN (it is only input))
        - RNN layer:
            input: input size (previous layer size)
            hidden (output): num_neurons in the layer
            inter-layer connections: in RNN module definition
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
    - problem with memory allocation
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
        - the majority of the effect of the image should disappear in during the blank sequences (because of that we want to split it in the middle of blank)
- reducing the layer size
    - randomly select subset of neurons for the training the model
    - it should have the largest impact on the size of the model
    - it should help us in definition using only RNNs (without FC)
    - try to play with the size till we reach the point where we can use RNNs
    - disadvantages:
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
- reducing number of time-steps to 900 (should be 1 example)


# 7.8.2024
- using the GPU with 48 GB memory (largest possible)
    - after optimizations I am able to run 
- the model is tested the way that all possible time step sizes will fit (even the 1 ms time-step)
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

# 29.8.2024
- obtained access to bio-informatics cluster partition
- will need to debug and work with the parallelism of the GPUs probably
- discussed the multitrial data with Luca and we agreed that he might generate some data for me
- we should use normalized cross-correlation for the data evaluation
- we agreed with Luca that he will generate new 1000 images with 10 trials for test dataset till I come back from vacation

# 30.8.2024
- work on the evaluation computation

# 31.8.2024
- have to change from float16 back to float32
    - otherwise there are `nan`

# 1.9.2024
- working variant using batches 
    - much faster than just 1 example
    - also no memory problems
    - can easily apply evaluation
- evaluation still returns `nan` -> need to be corrected
- changes of the raw extraction and dataset tools to work with also with the multiple trials
    - not completed yet

# 19.9.2024
- Meeting notes:
    - evaluation through whole vector concatenated through the time-steps
    - there are techniques how to prevent `nan` during the training (using smaller learning rate)
    - correlation should be for the vectors of all neurons in all time-steps
    - from LGN it is all positive weights

# 29.9.2024
- LGN weights are corrected (all are excitatory)
- loading of multi-trials corrected
    - training loads always single trial
    - testing loads all available trials

# 30.9.2024
- all training on:
    - `original_model_size=0.1`
- training 1:
    - `learning_rate=0.001`
    - `batch_size=50`
    - `epochs=1`
    - the CC was equal to approx:   `-0.02`
- training 2:
    - `learning_rate=0.001`
    - `batch_size=50`
    - `epochs=2`
    - the CC was equal to approx:   `0.03`
- training 3:
    - `learning_rate=0.0001`
    - `batch_size=50`
    - `epochs=2`
    - the CC was equal to approx:   `-0.02`

- conclusion:
    - maybe increase learning rate
    - probably more epochs
    - might be good idea to change the metrics (better correlation coefficient)


# 1.10.2024
- another testing on size 0.10 of original size
- it looks that reasonable learning rate is between `0.00075-0.003`
- reasonable number of trials is around `6-10`
- the best result was only 0.25 correlation


# 3.10.2024
- restructure of the model code for better and easier work
- testing training on size 0.25

# 4.10.2024
- adding small shared NN instead of simple neuron in RNNCell
    - for each layer its own
    - were not able to implement LSTM layer (might do manually)
- finished results of 0.25
    - looks like the best is of the learning rate 0.00075
- testing training of complex NN (with NN instead of neuron)

# 15.10.2024
Notes after the meeting:
- the training that reached 0.34 CC is quite good (although normally it might go to 0.9)
    - pros:
        - the data are much different (we are predicting the sequence not the number of spikes)
            - also the CC is not designed for sequential data
        - the training makes much more sense when each step starts on the previous target
        - it might increase when using larger time bins (reaching to 20)
- when using the NN instead of neuron it should be replaces by each neuron, it does not have to be super large in case of the width (layer size) but it should be deeper
    - basically we want 1 input and 1 output -> the neuron should add complexity
    - first thing that comes to my mind is using iterating through each input and returning output (loop through number of layer neurons)
- when predicting each step with predicted hidden state it does not make much sense in future time steps
    - it might in fact misled the training (it trains on wrong inputs)
- when starting the new sequence it should start with the first time step not zeros
- interval sizes: 5, 10, 15, and 20 are available for learning (other sizes have to be generated)


# 19.10.2024
- mistake in evaluation encountered - I was adding CC for layers -> it overflows 1 (max value is 4 then)
    - now corrected - evaluation is computed for concatenated tensors of the predictions for all layers
- training steps corrected - now the hidden states are the previous ones for the targets
    - in case of evaluation we assign only the first step (the rest should predict the model)
- model architecture corrected 
    - the inhibitory layers should get excitatory input from the previous state (not the current)
- we also started training on size of time intervals 20 instead of 5 as it looks 5 is very hard to train and training on 20 takes much less time
    - also now training on size 0.1 of the original model (since we fine-tune the training procedure and resolve the bugs)

# 21.10.2024
- it seems that using residual connections inside complex neuron is much better than not using it

# 24.10.2024
- after several tests of model training on the size 0.25 we consider complex model much better than simple
    - reached max CC `0.84` - almost our target
    - it seems that best learning rate is `1e-05`
    - it seems that `20` epochs is ideal number for training
    - the complex model does not seem to be overtraining yet
        - while model trained it stayed at similar CC
    - training is much stable than using simple model
- the residual connections are not probably better
    - till now model without residual connections are better a little bit
- CC over `0.7` is stable reachable

# 25.10.2024
- computation of loss changed - now computing loss across all layers at once
    - also I was printing only last step loss -> because of that I cannot check whether the average loss is descending
- training with small learning rate `8e-06` seem to be too small
    - learning very slowly and probably will stop lower than `1e-05`
- adding saving the best model and using it in the last evaluation

# 26.10.2024
- changed naming without `_summed`
    - it might be possible to merge time slots while loading the file
        - probably not use it -> much slower (have to load bigger data)
- it is needed to have different model sizes preprocessed before running
    - unfortunately it lead to higher memory storage demand

# 1.11.2024
Notes after meetings:
- the training acts strangely
    - it fluctuates in CC_NORM - should not if training properly
    - we should add classical CC_ABS (Pearson's CC) to check whether the normalization does not lead to strange results
- it seems that the model does not learn that good spatio-temporal behavior of the neurons
- we should check whether the data processing is in correct format:
    - it should be some blank time step -> stimulus -> additional blank
    - should check for test dataset -> we are sure that for train it is correctly defined
        - it looks it is correct
    - we should check the histogram of the number of neurons per number of spikes
        - it should exponentially decrease
- create plot of mean spatio-temporal response through all neurons in the layer
    - to see the mean behavior of the neuronal response makes sense
        - it should go from down (blank) to stimuli (steeply up) -> lower -> down (blank)
- select few neurons and plot it together to show its behavior throughout the images
    - also add its mean response
- implement `weights and biases` tool to the training - to see what it does during training


# 8.11.2024
- we should get rid of leading blank stage in the data
    - these might be the reasons of strange behavior at the beginning of the prediction
        - the responses might be still influenced by the previous one
    - I will also solve the problem of padding
        - I will just need to pad the last stimuli with blanks
- Also need to inspect the behavior at the end
    - it should slightly spike at the stimulus/blank border
- change blank/stimulus order
    - start with stimulus -> whole blank
        - it might be problematic that when starting with blanks it learns random spikes and do not perform well
            - we can see it from the data
    - also good because the padding would be needed only for the end examples
        - only 1% of examples with padded zeros at the end
    - need to redo trimming process
        - also need to run time merger again for regenerating the data
- add initial time step to plots (to see where it all starts)
- look at the time back-propagation
    - how to it works in pytorch
    - additional step might be adding this between time steps
        - artificial time steps between the targets
            - model might learn more the dynamics
                - it is pretty complicated to learn the dynamics that is so sharp
- inspect what happens at the end
    - there should be slight increase at the end
    - same problem also with trained responses
        - it predictions went downwards and targets upwards (strange)


# 15.11.2024
- strangely after dataset correction it looks that simple model is at least as good as complex one
    - the highest correlation of simple model is `0.86`
        - the best complex has correlation `0.84`
        - it helps to use more complex neuron model (for now 7 layers is the best)
- the training looks stable
    - once it starts to continually grow in correlation it stays at that level
        - major improvement in comparison to previous data
- while training the model with time step 10 -> I get smaller loss
    - the correlation seems to be comparable to time step 20
- residual connections seems be still worse than model without it
    - it trains quickly but it reaches the correlation about `0.05` smaller in average

- it looks that I wrongly defined residuals (I interchanged residuals for not-residuals)
    - residuals are indeed better than non-residuals
- we need to focus on DNN module
    - the model should be able to transfer the information between inhibitory and excitatory layers
    - now it looks that the model DNN module does not learn anything
        - based on the plot of rnn_output -> dnn_output function it looks that the model just does addition
            - it is almost identity with slight shift    

# 19.11.2024
- now implementing the hidden time steps to improve the model learning
    - we want to do back-propagation through time
        - it is not clear what it does when we have each time step visible

# 20.11.2024
- as it seems now it looks that there is backpropagation through time problem
    - it seems the state reset during training to target is the problem

# 26.11.2024
- as far as I get the information I would say it is essential to perform the optimizer step before each target state reset
- the optimizer step and backward step should be performed for all layers together
    - not sure about calculation of the loss (whether separate the losses or not)

# 3.12.2024
- model DNN module was applied in the wrong way
    - first it was applied non-linearity -> DNN module
        - we wanted DNN module instead of non-linearity
    - this might be the reason of very strange behavior of the DNN module
    - also might be the reason of the bad model dynamics
    -> solution was create custom RNNCell module
- also code refined

# 5.12.2024
- possible future steps after model captures the dynamics same as the spiking model:
    - checking that the connections are similar (the spiking model is bio-inspired)
    - not-showing some neurons -> predicting them from others
        - natural transition to real-data (in the future)
    - constraining the neurons with some connectivity rules
        - neurons that are further has less probability to be connected

# 13.12.2024
- implemented model variant where we split the inhibitory and excitatory inputs and pass two values to DNN
    - inputs to DNN are then sum of excitatory and inhibitory layer (separate 2 values)
- there was a problem caused by weight clipping mainly probably
    - we had a bug in code so the original model did not clip weights
    - after application of weight clipping we were encountering NaNs mainly in predictions of evaluation
    - after some inspection of gradients it looks that exploding gradients are the main cause of this problem
        - we applied gradient clipping (to prevent it)
    - we also applied bounded ReLU (max value 20) for last neuron activation to prevent large spike predictions
        - does not make sense to predict larger values (it is not realistic to have such many spikes in 20 ms).
        - due our model resolution (1 ms) we set the max value to 20

# 17.12.2024
- currently we are encountering issues with the training the model caused by the weight constraints
    - clipping to absolute value might help (instead of clipping to 0)
    - better initialization of the inhibitory/excitatory weights
        - as there are approximately 4 times less inhibitory neurons the weights should be 4 times higher
        - it might be also good to use adaptive learning rate for each layer
            - inhibitory should have 4 time lower (because they are in average 4 times higher)
        - also we know that the weights are generally distributed using Gauss distribution
            - try to apply this too
    - generally dealing with gradients might improve the training
    - maybe study gradient exploding problem and its improvements on weights in general case and try to apply this to constraint example (find the analogies)

# 30.12.2024
- changed activation non-linearity to combination of sigmoid and tanh
    - sigmoid - applied for output values of DNN module smaller than 1
    - `5 * torch.tanh(x / 5)` - for values greater than 1
    - this should make the non-linearity smoother in comparison to hardtanh (bounder ReLU)
    - the predictions are now bounded to interval (0, 5)
        - should not be a problem (in training dataset there are only few data with 4 spikes in 20ms window)
        - it is very improbable to have more spikes
- using grading clipping seems to be problematic
    - we are not using it now (only with very large boundary)
- using different weight initialization or adaptive learning rate for inh/exc layers does not seem to work
- using hidden time steps does not seem to be working
- using separate DNN module produces slightly better results than variant without it
    - currently, we are reaching 0.9 cc_norm (joint variant only 0.85)
- long-term problems with the model still persist

# 6.1.2025
- we have added the possibility to run the model on separate excitatory and inhibitory layers
    - seems to perform slightly better than having both joint
- we also added RNN variant of the neuron model that used LSTM model
    - currently it seems to be as good as classical feed-forward DNN approach
    - although, we encountered bug in the implementation that does not propagate the states of the neuron to other time steps 
        - this might significantly improve the model performance

# 7.1.2025