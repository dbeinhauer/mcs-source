# TODO:

## Data Aggregation from the dataset for analysis
1. histogram neuron spike rate - `HISTOGRAM_NEURON_SPIKE_RATES`
    - `num bins: 712` (highest possible spike rate)
2. histogram time bin spike rate - `HISTOGRAM_TIME_BIN_SPIKE_RATES`
    - `num bins: 20` (highest possible spike rate)
3. sum of spikes in each time bin - `TIME_BIN_SPIKE_COUNTS`
    - shape:    `[num_time_bins]`
        - maximal size: 712

4. sum spikes in each trial and experiment: - `EXPERIMENT_SPIKE_COUNTS`
    - shape:    `[num_experiments, num_trials]`
    - max size: 50000
5. mean and variance of spikes per neuron in separate trials - `NEURON_MEAN_VARIANCE_EXPERIMENT`
    - shape: `[num_experiments, num_trials]`
    - max size: 50000
Dimension `[num_experiments, num_trials]` experiments:
1. spike counts - `TOTAL_COUNT`
2. mean across neurons - `MEAN`
3. variance across neurons - `VARIANCE`
4. spike density - `DENSITY`
    - count non-zero time bins -> mean across time
    - how sparse is the dataset
5. average Per-bin synchrony - how much neurons spiked across one time bin - `SYNCHRONY`

6. for each neuron total spikes: 
    - shape `[num_neurons]`
    - max size: 33000


7. b) Fano Factor across trials - makes sense only for test
    - shape: `[experiment]`
    - need:
        - sum across spikes and time bins -> fano = vars / mean (over trials)
    - how consistent or unpredictable the neuron firing is (higher variant -> more unpredictable)


9. Temporal Synchrony:
    - shape: `[experiment, time_bins]`
        - max size: 35600000
            - might be problem
    - needs:
        - mean across trials -> `[time_bins, neurons]`
            - sum across neurons -> synchrony curve
10. Per trial Synchrony:
    - shape: `[experiment, trials, time_bins]`
        - max size: 35600000
            - might be problem
            - may overflow in test with multiple trials (probably not really)
    - same as 9. just do not mean across trials

- We know that shape: `[experiment, 35, neurons]` is not suitable
    - approx max size: ~57750000000



# Experiments
## Dataset
- I want to show that time binning is not a problem
    - we have lost some spikes -> show that not that big problem
    - show change in temporal behavior
    - show spiking rates
- I want to show that subseting is not a problem
    - show that the parameters stays roughly the same for subset and full dataset
    - maybe try to train the model on different subset sizes (probably won't make it)
- compare how omitting part of training data influence model performance -> show that more data would not really help

- I want to compare the model variants
    - based on CC_norm
    - based on temporal behavior across population
        - maybe difference of spike peaks?
        - maybe blank/stimuli difference?


# Notes from the meeting
- it would be nice to try to run the evaluation with the next time step resetting
    - to compare results of the model while we predict the time steps and while it one needs to predict the following time step (the thing that it did see during the training)
- it might be good to train the model with only 1 neuron in each layer to check whether the architecture can capture at least one neuron behavior